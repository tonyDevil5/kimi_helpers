r"""
DM Government Bond Performance Comparison
==========================================
Based on D:\python\notebook\DM_GOVY.ipynb.

Pulls developed-market government bond yields (US, France, Germany, Japan)
across key tenors and produces a compact PDF comparison report:
    - yield levels and changes over multiple horizons
    - z-scores of yield changes
    - RSI and Bollinger Band indicators
    - country-by-tenor performance tables
    - yield-curve snapshot chart

Run:
    python dm_govy_report.py

Output:
    ./dm_govy_output/DM_GOVY_Report.pdf
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "dm_govy_output"

COUNTRIES = {
    "us": "United States",
    "frf": "France",
    "dem": "Germany",
    "jpy": "Japan",
}
TENORS = ["2y", "5y", "10y", "20y", "30y"]

TICKERS = {
    "us": ["GT02 Govt", "GT05 Govt", "GT10 Govt", "GT20 Govt", "GT30 Govt"],
    "frf": ["GTFRF2Y Govt", "GTFRF5Y Govt", "GTFRF10Y Govt", "GTFRF20Y Govt", "GTFRF30Y Govt"],
    "dem": ["GTDEM2Y Govt", "GTDEM5Y Govt", "GTDEM10Y Govt", "GTDEM20Y Govt", "GTDEM30Y Govt"],
    "jpy": ["GTJPY2Y Govt", "GTJPY5Y Govt", "GTJPY10Y Govt", "GTJPY20Y Govt", "GTJPY30Y Govt"],
}

# Map each ticker to a {country}_{tenor} column
TICKER_TO_COL = {}
for country, tickers in TICKERS.items():
    for ticker, tenor in zip(tickers, TENORS):
        TICKER_TO_COL[ticker] = f"{country}_{tenor}"

# Currency / funding rate mapping
CURRENCY = {
    "us": "usd",
    "frf": "eur",
    "dem": "eur",
    "jpy": "jpy",
}
FUNDING_TICKERS = {
    "usd": "SOFRRATE Index",
    "eur": "ESTRON Index",
    "jpy": "MUTKCALM Index",
}
CARRY_WINDOW = 50

# Lookback horizons in business days (approximate)
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

RSI_PERIOD = 14
BB_WINDOW = 20
BB_NSTD = 2.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data_from_bloomberg(start_date: str = "2018-05-21") -> pd.DataFrame:
    """Pull px_last from Bloomberg for all DM gov tickers."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today().strftime("%Y-%m-%d")
    all_tickers = [t for tickers in TICKERS.values() for t in tickers]

    print(f"Downloading {len(all_tickers)} tickers from Bloomberg...")
    df = blp.bdh(tickers=all_tickers, flds=["px_last"], start_date=start_date, end_date=end_date)
    df = rename_bbg_columns(df)
    df = df.ffill().dropna()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_funding_rates(start_date: str = "2018-05-21") -> pd.DataFrame:
    """Pull funding rates (SOFR, ESTR, SONIA) from Bloomberg."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today().strftime("%Y-%m-%d")
    unique_tickers = list(dict.fromkeys(FUNDING_TICKERS.values()))  # preserve order, remove dups

    print(f"Downloading {len(unique_tickers)} funding rate tickers from Bloomberg...")
    df = blp.bdh(tickers=unique_tickers, flds=["px_last"], start_date=start_date, end_date=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    # Rename columns to currency codes
    rename_map = {v: k for k, v in FUNDING_TICKERS.items() if v in df.columns}
    df = df.rename(columns=rename_map)
    df = df.ffill().dropna()
    print(f"Loaded funding rates: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def rename_bbg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Bloomberg-style columns to {country}_{tenor}."""
    if isinstance(df.columns, pd.MultiIndex):
        if "px_last" in df.columns.get_level_values(-1):
            data = df.xs("px_last", axis=1, level=-1)
        else:
            data = df.copy()
            data.columns = data.columns.get_level_values(0)
    else:
        data = df.copy()

    rename_map = {}
    for col in data.columns:
        if col in TICKER_TO_COL:
            rename_map[col] = TICKER_TO_COL[col]
        else:
            rename_map[col] = _parse_bbg_ticker(col)
    return data.rename(columns=rename_map)


def _idx_date(idx_value) -> date:
    """Convert a pandas Timestamp or datetime.date to a date."""
    return idx_value.date() if hasattr(idx_value, "date") else idx_value


def _parse_bbg_ticker(ticker: str) -> str:
    """Fallback parser for Bloomberg ticker -> {country}_{tenor}."""
    import re

    t = str(ticker).upper()
    if "FRF" in t:
        country = "frf"
    elif "DEM" in t:
        country = "dem"
    elif "JPY" in t:
        country = "jpy"
    else:
        country = "us"

    m = re.search(r"(\d+)", t)
    if not m:
        raise ValueError(f"Cannot parse tenor from ticker '{ticker}'")
    years = m.group(1).lstrip("0")
    if years == "":
        years = "0"
    return f"{country}_{years}y"


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Simple RSI."""
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bbands(series: pd.Series, window: int = BB_WINDOW, n_std: float = BB_NSTD):
    """Bollinger bands. Returns mid, upper, lower."""
    s = series.astype(float)
    mid = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def compute_yield_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield changes in bps for each horizon."""
    changes = {}
    for label, days in HORIZONS.items():
        changes[label] = (df - df.shift(days)) * 100  # yield * 100 -> bps
    return pd.concat(changes, axis=1)


def compute_zscores(changes: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Z-score of yield changes over a trailing window."""
    zscores = {}
    for col in changes.columns.get_level_values(1).unique():
        for horizon in HORIZONS.keys():
            series = changes[(horizon, col)]
            mean = series.rolling(window=window, min_periods=60).mean()
            std = series.rolling(window=window, min_periods=60).std()
            zscores[(f"{horizon}_z", col)] = (series - mean) / std.replace(0, np.nan)
    return pd.concat(zscores, axis=1)


def build_summary_table(
    df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a wide summary table: latest yield, changes, RSI, BB position, carry metrics."""
    changes = compute_yield_changes(df)

    rows = []
    for col in df.columns:
        if col not in [f"{c}_{t}" for c in COUNTRIES for t in TENORS]:
            continue

        series = df[col]
        latest = series.iloc[-1]

        # Yield changes (bps)
        change_row = {"yield": latest}
        for horizon in HORIZONS:
            change_row[f"chg_{horizon}"] = changes[(horizon, col)].iloc[-1]

        # RSI and BB position
        rsi = compute_rsi(series)
        mid, upper, lower = compute_bbands(series)
        bb_pos = np.nan
        if pd.notna(upper.iloc[-1]) and pd.notna(lower.iloc[-1]) and upper.iloc[-1] != lower.iloc[-1]:
            bb_pos = (latest - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        change_row["rsi"] = rsi.iloc[-1]
        change_row["bb_pos"] = bb_pos

        # Carry profile
        if funding_df is not None:
            country = col.split("_")[0]
            ccy = CURRENCY.get(country)
            if ccy and ccy in funding_df.columns:
                funding_rate = funding_df[ccy].reindex(series.index).ffill()
                yield_change_bps = series.diff() * 100
                yield_change_std = yield_change_bps.rolling(window=CARRY_WINDOW, min_periods=CARRY_WINDOW).std()
                expected_carry = (series - funding_rate) * 100  # bps
                carry_ratio = expected_carry / yield_change_std.replace(0, np.nan)
                change_row["carry_bps"] = expected_carry.iloc[-1]
                change_row["carry_ratio"] = carry_ratio.iloc[-1]

        rows.append(pd.Series(change_row, name=col))

    summary = pd.DataFrame(rows)
    summary.index.name = "bond"
    return summary


def group_summary_by_tenor(summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split summary into one DataFrame per tenor."""
    groups: Dict[str, pd.DataFrame] = {}
    for tenor in TENORS:
        mask = summary.index.str.endswith(f"_{tenor}")
        if mask.any():
            groups[tenor] = summary.loc[mask].copy()
    return groups


# ---------------------------------------------------------------------------
# PDF report helpers
# ---------------------------------------------------------------------------
def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    fontsize: int = 8,
    header_color: str = "#40466e",
    row_height: float = 0.35,
    gradient_columns: Optional[List[str]] = None,
    compact: bool = False,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib figure (one page)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    df = df.reset_index()
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.86])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.97, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.94, subtitle, fontsize=8, ha="left", transform=fig.transFigure, color="#555555")

    table_data = [display_df.columns.tolist()] + display_df.values.tolist()
    if n_cols > 1:
        name_width = 0.22 if compact and n_cols >= 8 else 0.35
        col_widths = [name_width] + [(1.0 - name_width) / (n_cols - 1)] * (n_cols - 1)
    else:
        col_widths = [1.0]

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )

    total_cells = n_rows + 1
    cell_h = row_height if compact else max(row_height, 0.86 / total_cells)
    auto_font = fontsize if compact else min(fontsize, int(cell_h * 280))
    table.auto_set_font_size(False)
    table.set_fontsize(auto_font)

    gradient_cols = []
    if gradient_columns:
        for gc in gradient_columns:
            if gc in df.columns:
                gradient_cols.append(gc)

    color_funcs = {}
    for col_name in gradient_cols:
        color_funcs[col_name] = _column_color_func(df[col_name])

    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight="bold", color="white")
        cell.set_height(cell_h)

    for i in range(1, n_rows + 1):
        for j, col_name in enumerate(display_df.columns):
            cell = table[(i, j)]
            cell.set_height(cell_h)
            base_color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
            cell.set_facecolor(base_color)
            if col_name in color_funcs:
                val = df[col_name].iloc[i - 1]
                if pd.notna(val):
                    cell.set_facecolor(color_funcs[col_name](val))

    return fig


def _column_color_func(series: pd.Series):
    """Pick coloring function based on data range."""
    import matplotlib.colors as mcolors

    s = series.dropna()
    if s.empty:
        return lambda x: "#ffffff"

    vmin, vmax = float(s.min()), float(s.max())
    if vmin < 0 < vmax:
        limit = max(abs(vmin), abs(vmax), 0.01)

        def diverging(x):
            norm = (float(x) + limit) / (2 * limit)
            norm = max(0.0, min(1.0, norm))
            cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#4575b4", "#ffffff", "#d73027"])
            return mcolors.to_hex(cmap(norm)[:3])

        return diverging

    def sequential(x):
        if vmax == vmin:
            return "#ffffff"
        norm = (float(x) - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26"])
        return mcolors.to_hex(cmap(norm)[:3])

    return sequential


def _add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    gradient_columns: Optional[List[str]] = None,
    fontsize: int = 8,
    row_height: float = 0.35,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    compact: bool = False,
) -> None:
    fig = _df_to_figure(
        df,
        title,
        subtitle=subtitle,
        gradient_columns=gradient_columns,
        fontsize=fontsize,
        row_height=row_height,
        pagesize=pagesize,
        compact=compact,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_yield_curves(df: pd.DataFrame, countries: List[str]) -> plt.Figure:
    """Plot latest yield curve for each country."""
    fig, ax = plt.subplots(figsize=(11, 6))
    tenors_num = [2, 5, 10, 20, 30]
    colors = {"us": "blue", "frf": "red", "dem": "green", "jpy": "orange"}

    for country in countries:
        yields = []
        for tenor in TENORS:
            col = f"{country}_{tenor}"
            if col in df.columns:
                yields.append(df[col].iloc[-1])
            else:
                yields.append(np.nan)
        ax.plot(tenors_num, yields, marker="o", label=country.upper(), color=colors.get(country, "gray"))

    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("DM Government Yield Curves (Latest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_performance_heatmap(changes: pd.DataFrame, horizon: str = "1m") -> plt.Figure:
    """Heatmap of yield changes (bps) for a given horizon."""
    data = changes[horizon].T  # rows = bonds, cols = dates (latest 22)
    data = data.iloc[:, -22:]  # last 22 business days

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([d.strftime("%m-%d") for d in data.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=7)
    ax.set_title(f"Yield Change Heatmap ({horizon}) — last 22 business days (bps)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate the DM GOVY comparison PDF report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        pdf_path = OUTPUT_DIR / "DM_GOVY_Report.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"DM_GOVY_Report_{ts}.pdf"

    summary = build_summary_table(df, funding_df=funding_df)

    # Order by tenor first, then by country (us, frf, dem, jpy)
    country_order = list(COUNTRIES.keys())
    tenor_order = TENORS

    def _sort_key(name: str) -> Tuple[int, int]:
        country, tenor = name.split("_")
        return (tenor_order.index(tenor), country_order.index(country))

    summary_sorted = summary.loc[sorted(summary.index, key=_sort_key)]

    # One compact wide table: all countries/tenors, all metrics
    base_cols = ["yield"] + [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos"]
    display_cols = base_cols + (["carry_bps", "carry_ratio"] if funding_df is not None else [])
    gradient_cols = [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos"]
    if funding_df is not None:
        gradient_cols += ["carry_bps", "carry_ratio"]
    if funding_df is not None:
        rfr_text = "  |  ".join([f"{ccy.upper()} RFR: {funding_df[ccy].iloc[-1]:.3f}%" for ccy in funding_df.columns])
        subtitle = f"Latest: {_idx_date(df.index[-1])}  |  {rfr_text}"
    else:
        subtitle = f"Latest: {_idx_date(df.index[-1])} | Data start: {_idx_date(df.index[0])}"

    with PdfPages(pdf_path) as pdf:
        # Compact combined table (all tenors in one page)
        _add_table_page(
            pdf,
            summary_sorted[display_cols],
            "DM Government Bonds — Yield, Changes, RSI, BB Position & Carry",
            subtitle=subtitle,
            gradient_columns=gradient_cols,
            fontsize=5.5,
            row_height=0.05,
            pagesize=(13.0, 8.5),
            compact=True,
        )

        # Yield curve chart
        fig = plot_yield_curves(df, list(COUNTRIES.keys()))
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def _add_summary_page(pdf: PdfPages, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Cover page with top-level stats."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")

    title = "DM Government Bond Performance Comparison"
    subtitle = f"Latest date: {_idx_date(df.index[-1])}  |  Observations: {len(df)}"

    ax.text(0.5, 0.92, title, fontsize=22, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.86, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    # Best / worst performers by 1m yield change
    chg_1m = summary["chg_1m"].dropna().sort_values()
    worst = chg_1m.head(3)
    best = chg_1m.tail(3)

    perf_text = (
        "Largest Yield Risers (1m, bps)\n"
        + "\n".join([f"  {idx}: +{val:.1f}" for idx, val in best.iloc[::-1].items()])
        + "\n\nLargest Yield Fallers (1m, bps)\n"
        + "\n".join([f"  {idx}: {val:.1f}" for idx, val in worst.items()])
    )
    ax.text(0.15, 0.70, perf_text, fontsize=11, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2ff", edgecolor="#40466e"))

    contents = (
        "Report contents:\n"
        "1. Compact DM bond summary table\n"
        "   (yield, 1d/1w/1m/3m/6m/1y changes, RSI, BB position)\n"
        "2. DM yield curve snapshot"
    )
    ax.text(0.55, 0.70, contents, fontsize=10, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8ee", edgecolor="#8b5a00"))

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data_from_bloomberg()
    funding_df = load_funding_rates()
    pdf_path = generate_pdf_report(df, funding_df=funding_df)

    print("\n" + "=" * 70)
    print("DM Government Bond Performance Comparison")
    print("=" * 70)
    print(f"Latest date : {_idx_date(df.index[-1])}")
    print(f"Observations: {len(df)}")
    print(f"PDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
