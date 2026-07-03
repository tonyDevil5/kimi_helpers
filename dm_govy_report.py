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
    "uk": "United Kingdom",
}
TENORS = ["2y", "5y", "7y", "10y", "20y", "30y"]

TICKERS = {
    "us": ["GT02 Govt", "GT05 Govt", "GT07 Govt", "GT10 Govt", "GT20 Govt", "GT30 Govt"],
    "frf": ["GTFRF2Y Govt", "GTFRF5Y Govt", "GTFRF7Y Govt", "GTFRF10Y Govt", "GTFRF20Y Govt", "GTFRF30Y Govt"],
    "dem": ["GTDEM2Y Govt", "GTDEM5Y Govt", "GTDEM7Y Govt", "GTDEM10Y Govt", "GTDEM20Y Govt", "GTDEM30Y Govt"],
    "jpy": ["GTJPY2Y Govt", "GTJPY5Y Govt", "GTJPY7Y Govt", "GTJPY10Y Govt", "GTJPY20Y Govt", "GTJPY30Y Govt"],
    "uk": ["GUKG2 Index", "GUKG5 Index", "GUKG7 Index", "GUKG10 Index", "GUKG20 Index", "GUKG30 Index"],
}

# Map each ticker to a {country}_{tenor} column
TICKER_TO_COL = {}
for country, tickers in TICKERS.items():
    for ticker, tenor in zip(tickers, TENORS):
        TICKER_TO_COL[ticker] = f"{country}_{tenor}"

# Spread analysis uses the existing US 10Y generic yield (us_10y = GT10 Govt)
# vs Germany 10Y (dem_10y = GTDEM10Y Govt) and France 10Y (frf_10y = GTFRF10Y Govt).

# Currency / funding rate mapping
CURRENCY = {
    "us": "usd",
    "frf": "eur",
    "dem": "eur",
    "jpy": "jpy",
    "uk": "gbp",
}
FUNDING_TICKERS = {
    "usd": "SOFRRATE Index",
    "eur": "ESTRON Index",
    "jpy": "MUTKCALM Index",
    "gbp": "SONIO/N Index",
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

# Recent-window quantile analysis
QUANTILE_WINDOW = 60

# Realized volatility lookback (business days) and annualization factor
REALIZED_VOL_WINDOW = 20
REALIZED_VOL_ANNUALIZE = 252  # business days in a year


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
    if "GUKG" in t:
        country = "uk"
    elif "FRF" in t:
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


def compute_realized_volatility(series: pd.Series, window: int = REALIZED_VOL_WINDOW) -> pd.Series:
    """
    Annualized realized volatility of daily yield changes.

    Yields are assumed in percent.  Returns volatility in bps.
    """
    s = series.astype(float)
    daily_change = s.diff()
    vol = daily_change.rolling(window=window, min_periods=window).std() * np.sqrt(REALIZED_VOL_ANNUALIZE) * 100
    return vol


def compute_yield_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield changes in bps for each horizon."""
    changes = {}
    for label, days in HORIZONS.items():
        if label == "1d":
            # The report is typically run during Asia hours, so the latest
            # observation is a live/Asia price rather than the previous day's
            # close. Compute the true 1-day change as the last completed daily
            # close vs the close before that (t-1 vs t-2).
            changes[label] = (df.shift(days) - df.shift(days + 1)) * 100
        else:
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

        # Realized volatility (annualized, bps)
        rv = compute_realized_volatility(series)
        change_row["rv20"] = rv.iloc[-1]

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
# Spread analysis
# ---------------------------------------------------------------------------
def build_spread_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table for GT10 vs Bund/OAT 10Y spreads.

    Columns: level, 1d/1w/1m/3m/6m/1y change (bps), RSI, BB position, RV20.
    No carry metrics.
    """
    required = {"us_10y", "dem_10y", "frf_10y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for spread analysis: {missing}")

    spread_pairs = {
        "GT10 - Bund10": df["us_10y"] - df["dem_10y"],
        "GT10 - OAT10Y": df["us_10y"] - df["frf_10y"],
    }

    rows = []
    for name, spread in spread_pairs.items():
        # Drop leading NaNs but keep internal ones so the series is continuous
        spread = spread.dropna()
        if spread.empty:
            continue

        latest = spread.iloc[-1]

        # Spread changes in bps
        spread_df = pd.DataFrame({name: spread})
        changes = compute_yield_changes(spread_df)

        change_row = {"level": latest}
        for horizon in HORIZONS:
            change_row[f"chg_{horizon}"] = changes[(horizon, name)].iloc[-1]

        # RSI and BB position
        rsi = compute_rsi(spread)
        mid, upper, lower = compute_bbands(spread)
        bb_pos = np.nan
        if pd.notna(upper.iloc[-1]) and pd.notna(lower.iloc[-1]) and upper.iloc[-1] != lower.iloc[-1]:
            bb_pos = (latest - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        change_row["rsi"] = rsi.iloc[-1]
        change_row["bb_pos"] = bb_pos

        # Realized volatility (annualized, bps)
        rv = compute_realized_volatility(spread)
        change_row["rv20"] = rv.iloc[-1]

        # 1-year (250-day) spread quantiles and empirical percentile
        window_1y = 250
        recent = spread.tail(window_1y)
        if not recent.empty:
            q = recent.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            min_v = float(q[0.0])
            max_v = float(q[1.0])
            # True empirical percentile: share of observations <= current
            emp_pct = (recent <= latest).mean()
            change_row["min_1y"] = min_v
            change_row["p25_1y"] = float(q[0.25])
            change_row["p50_1y"] = float(q[0.5])
            change_row["p75_1y"] = float(q[0.75])
            change_row["max_1y"] = max_v
            change_row["pct_1y"] = emp_pct

        rows.append(pd.Series(change_row, name=name))

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary.index.name = "spread"
    return summary


def build_quantile_summary_table(
    df: pd.DataFrame, window: int = QUANTILE_WINDOW
) -> Dict[str, pd.DataFrame]:
    """
    For each country, compute min/25%/50%/75%/max over the recent window
    and highlight where the current yield sits in that range.
    """
    recent = df.tail(window)
    summaries: Dict[str, pd.DataFrame] = {}
    for country in COUNTRIES:
        rows = []
        for tenor in TENORS:
            col = f"{country}_{tenor}"
            if col not in df.columns:
                continue
            series = recent[col]
            current = float(series.iloc[-1])
            q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            min_v = float(q[0.0])
            max_v = float(q[1.0])
            range_pct = (current - min_v) / (max_v - min_v) if max_v != min_v else np.nan
            rows.append(
                {
                    "tenor": tenor,
                    "current": current,
                    "min": min_v,
                    "p25": float(q[0.25]),
                    "p50": float(q[0.5]),
                    "p75": float(q[0.75]),
                    "max": max_v,
                    "range_pct": range_pct,
                }
            )
        if rows:
            summaries[country] = pd.DataFrame(rows).set_index("tenor")
    return summaries


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
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.84])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.98, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.93, subtitle, fontsize=8, ha="left", transform=fig.transFigure, color="#555555")

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


def plot_spread_history_1y(df: pd.DataFrame) -> plt.Figure:
    """Plot 1-year history of GT10 vs Bund10 and GT10 vs OAT10 spreads."""
    required = {"us_10y", "dem_10y", "frf_10y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for spread history plot: {missing}")

    # Last ~1 year of data
    window = 252
    recent = df.tail(window)

    spreads = {
        "GT10 - Bund10": recent["us_10y"] - recent["dem_10y"],
        "GT10 - OAT10Y": recent["us_10y"] - recent["frf_10y"],
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = {"GT10 - Bund10": "green", "GT10 - OAT10Y": "red"}

    for ax, (name, spread) in zip(axes, spreads.items()):
        spread_bps = spread * 100
        ax.plot(spread_bps.index, spread_bps, color=colors.get(name, "blue"), linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        # Tight y-axis: from min to max with small padding only for the zero line
        ymin, ymax = spread_bps.min(), spread_bps.max()
        pad = (ymax - ymin) * 0.02 if ymax != ymin else 1.0
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel("Spread (bps)")
        ax.set_title(f"{name} 1-Year History")
        ax.grid(True, alpha=0.3)

        # Latest annotation
        latest = spread_bps.iloc[-1]
        ax.text(
            0.02, 0.95,
            f"Latest: {latest:.1f} bps",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

    axes[-1].set_xlabel("Date")
    plt.subplots_adjust(hspace=0.25)
    return fig


def plot_yield_curves(df: pd.DataFrame, countries: List[str]) -> plt.Figure:
    """Plot latest yield curve for each country."""
    fig, ax = plt.subplots(figsize=(11, 6))
    tenors_num = [2, 5, 7, 10, 20, 30]
    colors = {"us": "blue", "frf": "red", "dem": "green", "jpy": "orange", "uk": "purple"}

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
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"DM_GOVY_Report_{report_date}.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"DM_GOVY_Report_{report_date}_{ts}.pdf"

    summary = build_summary_table(df, funding_df=funding_df)

    # Order by tenor first, then by country (us, frf, dem, jpy)
    country_order = list(COUNTRIES.keys())
    tenor_order = TENORS

    def _sort_key(name: str) -> Tuple[int, int]:
        country, tenor = name.split("_")
        return (tenor_order.index(tenor), country_order.index(country))

    summary_sorted = summary.loc[sorted(summary.index, key=_sort_key)]

    # One compact wide table: all countries/tenors, all metrics
    base_cols = ["yield"] + [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos", "rv20"]
    display_cols = base_cols + (["carry_bps", "carry_ratio"] if funding_df is not None else [])
    gradient_cols = [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos", "rv20"]
    if funding_df is not None:
        gradient_cols += ["carry_bps", "carry_ratio"]
    if funding_df is not None:
        rfr_text = "  |  ".join([f"{ccy.upper()} RFR: {funding_df[ccy].iloc[-1]:.3f}%" for ccy in funding_df.columns])
        subtitle = f"Latest: {_idx_date(df.index[-1])}  |  {rfr_text}"
    else:
        subtitle = f"Latest: {_idx_date(df.index[-1])} | Data start: {_idx_date(df.index[0])}"

    with PdfPages(pdf_path) as pdf:
        # Compact summary tables split by tenor group
        tenor_groups = [
            ("2y / 5y / 7y", ["2y", "5y", "7y"]),
            ("10y / 20y / 30y", ["10y", "20y", "30y"]),
        ]
        for group_name, group_tenors in tenor_groups:
            mask = summary_sorted.index.map(lambda x: x.split("_")[1] in group_tenors)
            sub = summary_sorted.loc[mask, display_cols]
            if sub.empty:
                continue
            _add_table_page(
                pdf,
                sub,
                f"DM Government Bonds — {group_name}",
                subtitle=subtitle,
                gradient_columns=gradient_cols,
                fontsize=6.5,
                row_height=0.06,
                pagesize=(13.0, 8.5),
                compact=True,
            )

        # GT10 vs Bund/OAT spread summary table
        try:
            spread_summary = build_spread_summary_table(df)
            if not spread_summary.empty:
                quantile_cols = ["min_1y", "p25_1y", "p50_1y", "p75_1y", "max_1y", "pct_1y"]
                spread_cols = ["level"] + [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos", "rv20"] + quantile_cols
                _add_table_page(
                    pdf,
                    spread_summary[spread_cols],
                    "GT10 Spread Analysis",
                    subtitle=f"Latest: {_idx_date(df.index[-1])}  |  GT10 vs 10Y Bund / OAT  |  changes in bps",
                    gradient_columns=[f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos", "rv20", "pct_1y"],
                    fontsize=8,
                    row_height=0.10,
                    pagesize=(13.0, 5.5),
                    compact=True,
                )

                # 1-year spread history chart
                fig = plot_spread_history_1y(df)
                pdf.savefig(fig)
                plt.close(fig)
        except ValueError as exc:
            print(f"Skipping spread table: {exc}")
        except Exception as exc:
            print(f"Skipping spread history plot: {exc}")

        # Yield curve chart
        fig = plot_yield_curves(df, list(COUNTRIES.keys()))
        pdf.savefig(fig)
        plt.close(fig)

        # Recent-window quantile summary by country
        quantile_summaries = build_quantile_summary_table(df, window=QUANTILE_WINDOW)
        q_subtitle = f"Latest: {_idx_date(df.index[-1])}  |  Window: last {QUANTILE_WINDOW} observations"
        for country, q_df in quantile_summaries.items():
            q_df_sorted = q_df.reindex(TENORS)
            country_name = COUNTRIES.get(country, country)
            _add_table_page(
                pdf,
                q_df_sorted,
                f"{country_name} — {QUANTILE_WINDOW}-Day Yield Quantiles",
                subtitle=q_subtitle,
                gradient_columns=["range_pct"],
                fontsize=9,
                row_height=0.08,
                pagesize=(10.0, 5.0),
                compact=True,
            )

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
        "2. DM yield curve snapshot\n"
        f"3. {QUANTILE_WINDOW}-day yield quantiles by country\n"
        "   (current yield vs min/25%/50%/75%/max)"
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
