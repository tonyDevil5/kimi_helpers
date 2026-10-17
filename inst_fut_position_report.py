r"""
Institutional Futures Position Analysis Report
==============================================
Based on D:\python\notebook\inst_fut_position.ipynb.

Analyzes CFTC institutional futures positions by investor type
(Asset Manager, Hedge Fund, Dealer, Other) for U.S. Treasury futures
and their relationship with reference Treasury yields.

Data sources (Bloomberg via xbbg):
    - TFF2* futures position indices by category (net positions)
    - GT* Govt reference Treasury yields

Output:
    ./inst_fut_position_output/InstFutPosition_Report.pdf

Run:
    python inst_fut_position_report.py
"""

from __future__ import annotations

import textwrap
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from statsmodels.tsa.stattools import adfuller, coint

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
OUTPUT_DIR = Path(__file__).parent / "inst_fut_position_output"

# Treasury futures term codes (Bloomberg TFF family)
TICKER_TERM = {
    "2y": "TFF2G",
    "5y": "TFF2I",
    "10y": "TFF2H",
    "ultra": "TFF2Z",
    "us": "TFF2E",
    "30y": "TFF2F",
}

# Common futures ticker aliases used on the summary pages
CONTRACT_CODE_MAP = {
    "2y": "TU",
    "5y": "FV",
    "10y": "TY",
    "ultra": "UXY",
    "us": "US",
    "30y": "WN",
}

# Investor category suffixes for net positions
INST_CAT_NET = {
    "AM": "AIN",
    "HF": "LFN",
    "Dealer": "DIN",
    "Other": "ORN",
}

# Reference Treasury yield for each futures tenor
REF_BOND_MAP = {
    "2y": "GT02 Govt",
    "5y": "GT05 Govt",
    "10y": "GT07 Govt",
    "ultra": "GT10 Govt",
    "us": "GT20 Govt",
    "30y": "GT30 Govt",
}

START_DATE = "2021-01-01"

# Only keep weeks where the reference yield moved more than this (bps)
YIELD_CHANGE_FILTER_BPS = 10

# Rolling window for correlation/beta summary (weeks)
CORR_WINDOW_WEEKS = 52  # ~1 year of weekly data

# Resampling frequency; CFTC TFF is weekly, so aggregate to Friday closes
WEEKLY_FREQ = "W-FRI"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data_from_bloomberg(
    tenor: str,
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Pull net futures positions by category and reference yield for a tenor."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    term_code = TICKER_TERM[tenor]
    tickers = [f"{term_code}{suffix} Index" for suffix in INST_CAT_NET.values()]
    tickers.append(REF_BOND_MAP[tenor])

    print(f"Downloading {tenor} data: {len(tickers)} tickers from Bloomberg...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)

    # xbbg >= 1.x returns a narwhals DataFrame in long format
    # (ticker / date / field / value); older versions return a pandas
    # DataFrame with MultiIndex columns. Normalize both to a wide frame.
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    elif isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().dropna()

    # Rename columns
    inv_map = {f"{term_code}{suffix} Index": cat for cat, suffix in INST_CAT_NET.items()}
    inv_map[REF_BOND_MAP[tenor]] = "ref_yield"
    df = df.rename(columns=inv_map)

    # Convert reference yield to bps for easier interpretation
    df["ref_yield_bps"] = df["ref_yield"] * 100

    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_weekly_changes(df: pd.DataFrame, freq: str = WEEKLY_FREQ) -> pd.DataFrame:
    """Resample daily data to weekly Friday closes and compute WoW changes."""
    weekly = df.resample(freq).last().ffill().dropna()
    changes = weekly.diff().dropna()
    changes = changes.rename(columns={
        "AM": "AM_chg",
        "HF": "HF_chg",
        "Dealer": "Dealer_chg",
        "Other": "Other_chg",
        "ref_yield_bps": "ref_chg_bps",
    })
    return changes


def filter_by_yield_change(changes: pd.DataFrame, threshold_bps: int = YIELD_CHANGE_FILTER_BPS) -> pd.DataFrame:
    """Keep only observations where the reference yield moved more than threshold."""
    return changes[changes["ref_chg_bps"].abs() > threshold_bps].copy()


def compute_correlations(changes: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of position changes and reference yield changes."""
    cols = ["AM_chg", "HF_chg", "Dealer_chg", "Other_chg", "ref_chg_bps"]
    corr = changes[cols].corr()
    rename_map = {
        "AM_chg": "AM",
        "HF_chg": "HF",
        "Dealer_chg": "Dealer",
        "Other_chg": "Other",
        "ref_chg_bps": "ref_yield_bps",
    }
    corr = corr.rename(index=rename_map, columns=rename_map)
    return corr


def compute_betas(changes: pd.DataFrame) -> pd.DataFrame:
    """
    Regress reference yield change (bps) on each category's position change.
    Returns slope (bps per contract), intercept, R², p-value.
    """
    categories = ["AM", "HF", "Dealer", "Other"]
    rows = []
    for cat in categories:
        x = changes[f"{cat}_chg"].values
        y = changes["ref_chg_bps"].values
        if len(x) < 5 or np.std(x) == 0:
            rows.append({
                "category": cat,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "p_value": np.nan,
                "n": len(x),
            })
            continue
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        rows.append({
            "category": cat,
            "slope": slope,
            "intercept": intercept,
            "r2": r_value ** 2,
            "p_value": p_value,
            "n": len(x),
        })
    return pd.DataFrame(rows).set_index("category")


def compute_predictive_betas(changes: pd.DataFrame) -> pd.DataFrame:
    """
    Regress next-period reference yield change on current position change.
    Tests whether positions lead yields.
    """
    categories = ["AM", "HF", "Dealer", "Other"]
    changes = changes.copy()
    changes["ref_next_chg_bps"] = changes["ref_chg_bps"].shift(-1)
    changes = changes.dropna()

    rows = []
    for cat in categories:
        x = changes[f"{cat}_chg"].values
        y = changes["ref_next_chg_bps"].values
        if len(x) < 5 or np.std(x) == 0:
            rows.append({
                "category": cat,
                "slope": np.nan,
                "r2": np.nan,
                "p_value": np.nan,
                "n": len(x),
            })
            continue
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        rows.append({
            "category": cat,
            "slope": slope,
            "r2": r_value ** 2,
            "p_value": p_value,
            "n": len(x),
        })
    return pd.DataFrame(rows).set_index("category")


def compute_rolling_correlation(changes: pd.DataFrame, window: int = CORR_WINDOW_WEEKS) -> pd.DataFrame:
    """Rolling correlation of each category's weekly position change vs weekly yield change."""
    categories = ["AM", "HF", "Dealer", "Other"]
    corr_df = pd.DataFrame(index=changes.index)
    for cat in categories:
        corr_df[cat] = changes[f"{cat}_chg"].rolling(window=window, min_periods=window // 2).corr(
            changes["ref_chg_bps"]
        )
    return corr_df.dropna()


def adf_test(series: pd.Series, name: str = "Series") -> Tuple[float, bool]:
    """ADF test; returns (p-value, is_stationary)."""
    result = adfuller(series.dropna())
    return result[1], result[1] < 0.05


def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, bool]:
    """Engle-Granger cointegration test; returns (p-value, is_cointegrated)."""
    score, p_value, _ = coint(series_a, series_b)
    return p_value, p_value < 0.05


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def _add_title_page(pdf: PdfPages, tenors: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Title / subtitle positioned high to leave room for the summary box below.
    ax.text(0.5, 0.90, "Institutional Futures Position Analysis", fontsize=22,
            weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.84, "U.S. Treasury futures positioning by investor type vs. reference yields",
            fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    report_date = date.today().strftime("%Y-%m-%d")
    ref_yields = sorted(set(REF_BOND_MAP.values()))
    # Split the reference-yield list across two lines so the box does not get too wide.
    mid = (len(ref_yields) + 1) // 2
    ref_yield_lines = (
        ", ".join(ref_yields[:mid]) + "\n" +
        " " * 18 + ", ".join(ref_yields[mid:])
    )
    summary_text = (
        "Data Summary\n"
        f"  Date range: {START_DATE} to {report_date}\n"
        f"  Tenors analyzed: {', '.join(tenors)}\n"
        f"  Investor categories: {', '.join(INST_CAT_NET.keys())}\n"
        f"  Position type: Net\n"
        f"  Frequency: weekly ({WEEKLY_FREQ})\n"
        f"  Yield-change filter: >{YIELD_CHANGE_FILTER_BPS} bps\n"
        f"  Reference yields:\n{ref_yield_lines}"
    )
    # va="top" means y=0.68 is the *top* of the box, preventing overlap with the subtitle.
    ax.text(0.5, 0.68, summary_text, fontsize=11, family="monospace",
            ha="center", va="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#40466e"))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _df_to_figure(df: pd.DataFrame, title: str, subtitle: str = "",
                  pagesize: Tuple[float, float] = (11.0, 5.5),
                  fontsize: int = 8,
                  header_rotation: int = 0) -> plt.Figure:
    """Render a DataFrame as a matplotlib table figure."""
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold", y=0.96, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.87, subtitle, fontsize=8, ha="left",
                transform=fig.transFigure, color="#555555")
    fig.subplots_adjust(top=0.80)

    display_df = df.reset_index().copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    # Wrap long column names so they remain readable without rotation.
    wrapped_cols = [
        textwrap.fill(str(col), width=12, break_long_words=False, replace_whitespace=False)
        for col in display_df.columns
    ]
    table_data = [wrapped_cols] + display_df.values.tolist()

    # Allocate a wider first column for the row label; distribute the rest.
    name_width = 0.10 if n_cols >= 8 else 0.14
    data_width = (1.0 - name_width) / max(1, n_cols - 1)
    # Slightly widen columns when there are many of them to keep text readable.
    if n_cols > 7:
        data_width *= 1.15
        name_width = max(0.08, 1.0 - data_width * (n_cols - 1))
    col_widths = [name_width] + [data_width] * (n_cols - 1)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    header_height = 0.22 if header_rotation else 0.10
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")
        cell.set_height(header_height)
        if header_rotation:
            # Rotate headers so long column names do not overlap.
            cell.set_text_props(rotation=header_rotation, ha="center", va="bottom")

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")
            cell.set_height(0.06)

    # Shrink table vertically to leave room for rotated headers.
    table.scale(1.0, 1.0)
    return fig


def plot_positions(df: pd.DataFrame, tenor: str) -> plt.Figure:
    """Time series of net positions by investor category."""
    fig, ax = plt.subplots(figsize=(12, 5))
    categories = ["AM", "HF", "Dealer", "Other"]
    for cat in categories:
        ax.plot(df.index, df[cat], label=cat, linewidth=1.2)
    ax.set_title(f"{tenor} Net Futures Positions by Investor Type")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net contracts")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_position_vs_yield(changes: pd.DataFrame, tenor: str) -> plt.Figure:
    """Scatter of weekly position changes vs weekly reference yield changes."""
    categories = ["AM", "HF", "Dealer", "Other"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = {"AM": "blue", "HF": "green", "Dealer": "red", "Other": "orange"}

    for ax, cat in zip(axes, categories):
        x = changes[f"{cat}_chg"]
        y = changes["ref_chg_bps"]
        # Plot in thousands of contracts for readability
        xk = x / 1000.0
        ax.scatter(xk, y, alpha=0.5, color=colors[cat], s=35)

        # Regression line
        if np.std(x) > 0 and len(x) > 5:
            slope, intercept, r_value, p_value, _ = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line / 1000.0, intercept + slope * x_line,
                    color="black", linestyle="--", linewidth=1)
            slope_per_1k = slope * 1000.0
            ax.text(0.05, 0.95,
                    f"slope={slope_per_1k:.3f} bps/1k\nR²={r_value**2:.2f}\np={p_value:.2f}",
                    transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
        ax.set_title(f"{cat}")
        ax.set_xlabel("Position change (1,000 contracts)")
        ax.set_ylabel("Ref yield change (bps)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{tenor}: Weekly Position Changes vs Reference Yield Changes\n"
                 f"(filtered on |ref yield change| > {YIELD_CHANGE_FILTER_BPS} bps)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_rolling_correlation(rolling_corr: pd.DataFrame, tenor: str) -> plt.Figure:
    """Rolling 1-year correlation of position changes vs yield changes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for cat in rolling_corr.columns:
        ax.plot(rolling_corr.index, rolling_corr[cat], label=cat, linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_title(f"{tenor}: Rolling 1Y Correlation (Weekly Position Change vs Weekly Yield Change)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weekly_position_changes(changes: pd.DataFrame, tenor: str) -> plt.Figure:
    """Bar charts of weekly position changes by investor type for a single tenor."""
    categories = ["AM", "HF", "Dealer", "Other"]
    colors = {"AM": "blue", "HF": "green", "Dealer": "red", "Other": "orange"}

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for ax, cat in zip(axes, categories):
        col = f"{cat}_chg"
        if col not in changes.columns:
            ax.set_visible(False)
            continue

        s = changes[col].dropna()
        # Color bars by sign (green = added long / reduced short, red = opposite)
        bar_colors = ["green" if v >= 0 else "red" for v in s.values]

        ax.bar(s.index, s / 1000.0, color=bar_colors, width=5, alpha=0.7)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Change (k contracts)")
        ax.set_title(f"{tenor.upper()} - {cat} Weekly Position Change")
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_recent_changes_by_investor(
    results_by_tenor: Dict[str, Dict], n_obs: int = 10
) -> Dict[str, plt.Figure]:
    """For each investor type, plot the last n weekly position changes across all contracts,
    overlaid with reference Treasury yields on a secondary axis."""
    categories = ["AM", "HF", "Dealer", "Other"]
    contract_colors = {
        "TU": "#1f77b4", "FV": "#ff7f0e", "TY": "#2ca02c",
        "UXY": "#d62728", "US": "#9467bd", "WN": "#8c564b",
    }
    yield_linestyles = {
        "TU": "-", "FV": "--", "TY": "-.",
        "UXY": ":", "US": "-", "WN": "--",
    }

    # Build reference yield panel (weekly Friday closes)
    yield_frames: Dict[str, pd.Series] = {}
    for tenor, res in results_by_tenor.items():
        code = CONTRACT_CODE_MAP[tenor]
        ref_yield = res["df"]["ref_yield"].resample(WEEKLY_FREQ).last().ffill()
        yield_frames[code] = ref_yield
    yield_df = pd.DataFrame(yield_frames)

    figures: Dict[str, plt.Figure] = {}
    for cat in categories:
        frames: Dict[str, pd.Series] = {}
        for tenor, res in results_by_tenor.items():
            code = CONTRACT_CODE_MAP[tenor]
            col = f"{cat}_chg"
            if col in res["weekly_changes"].columns:
                frames[code] = res["weekly_changes"][col]

        if not frames:
            continue

        df_cat = pd.DataFrame(frames).dropna(how="all").tail(n_obs)
        if df_cat.empty:
            continue

        # Align yield data to the same dates used for position changes
        common_dates = df_cat.index
        yield_slice = yield_df.loc[yield_df.index.isin(common_dates)].reindex(common_dates)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(df_cat))
        width = 0.12
        contracts = [c for c in CONTRACT_CODE_MAP.values() if c in df_cat.columns]

        for i, contract in enumerate(contracts):
            offset = (i - (len(contracts) - 1) / 2) * width
            values = df_cat[contract] / 1000.0
            ax.bar(
                x + offset,
                values,
                width,
                label=contract,
                color=contract_colors.get(contract),
                alpha=0.75,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.strftime("%Y-%m-%d") for d in df_cat.index], rotation=45, ha="right"
        )
        ax.set_ylabel("Weekly position change (k contracts)")
        ax.set_title(
            f"{cat}: Last {len(df_cat)} weeks of position changes by contract"
        )
        ax.legend(title="Contract", loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        # Overlay reference Treasury yields on a secondary axis
        ax2 = ax.twinx()
        for contract in contracts:
            if contract in yield_slice.columns:
                ax2.plot(
                    x,
                    yield_slice[contract],
                    color=contract_colors.get(contract),
                    linestyle=yield_linestyles.get(contract, "-"),
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=f"{contract} yield",
                )
        ax2.set_ylabel("Reference Treasury yield (%)")
        ax2.legend(title="Yield", loc="upper right")

        fig.tight_layout()
        figures[cat] = fig

    return figures


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------
def analyze_tenor(tenor: str) -> Dict[str, object]:
    """Run the full analysis for a single tenor."""
    df = load_data_from_bloomberg(tenor)
    changes = compute_weekly_changes(df)
    filtered = filter_by_yield_change(changes)

    # Align to the latest meaningful CFTC report week (skip stale carry-forward weeks)
    weekly = df.resample(WEEKLY_FREQ).last().ffill().dropna()
    activity = changes[["AM_chg", "HF_chg", "Dealer_chg", "Other_chg"]].abs().sum(axis=1)
    nonzero_changes = changes[activity > 0]
    if nonzero_changes.empty:
        latest_change = changes.iloc[-1]
        latest_change_date = changes.index[-1]
    else:
        latest_change = nonzero_changes.iloc[-1]
        latest_change_date = nonzero_changes.index[-1]

    # Position level as of the latest report date, and the prior report date
    latest_weekly = weekly.loc[latest_change_date][["AM", "HF", "Dealer", "Other"]]
    weekly_dates = weekly.index.tolist()
    if latest_change_date in weekly_dates and weekly_dates.index(latest_change_date) > 0:
        prev_idx = weekly_dates.index(latest_change_date) - 1
        prev_week_date = weekly.index[prev_idx]
        prev_weekly = weekly.iloc[prev_idx][["AM", "HF", "Dealer", "Other"]]
    else:
        prev_week_date = latest_change_date
        prev_weekly = latest_weekly

    results = {
        "tenor": tenor,
        "df": df,
        "weekly_levels": weekly,
        "weekly_changes": changes,
        "filtered": filtered,
        "latest_positions": latest_weekly,
        "previous_positions": prev_weekly,
        "latest_changes": latest_change,
        "latest_change_date": latest_change_date,
        "previous_week_date": prev_week_date,
        "correlations": compute_correlations(filtered),
        "betas": compute_betas(filtered),
        "predictive": compute_predictive_betas(changes),
        "rolling_corr": compute_rolling_correlation(changes),
    }

    # Stationarity / cointegration on full sample (optional diagnostics)
    for cat in ["AM", "HF", "Dealer", "Other"]:
        p_value, is_stationary = adf_test(df[cat], name=f"{tenor}_{cat}")
        results[f"{cat}_adf_pvalue"] = p_value
        results[f"{cat}_stationary"] = is_stationary

    return results


def build_cross_tenor_position_tables(results_by_tenor: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """Build latest position and WoW change tables across all tenors/contracts."""
    categories = ["AM", "HF", "Dealer", "Other"]
    tenors = list(results_by_tenor.keys())

    pos_rows = []
    chg_rows = []
    for cat in categories:
        pos_row = {"category": cat}
        chg_row = {"category": cat}
        total_pos = 0
        total_chg = 0
        for tenor in tenors:
            res = results_by_tenor[tenor]
            pos = res["latest_positions"][cat]
            chg = res["latest_changes"][f"{cat}_chg"]
            code = CONTRACT_CODE_MAP[tenor]
            pos_row[code] = pos
            chg_row[code] = chg
            total_pos += pos if pd.notna(pos) else 0
            total_chg += chg if pd.notna(chg) else 0
        pos_row["Total"] = total_pos
        chg_row["Total"] = total_chg
        pos_rows.append(pos_row)
        chg_rows.append(chg_row)

    pos_df = pd.DataFrame(pos_rows).set_index("category")
    chg_df = pd.DataFrame(chg_rows).set_index("category")

    def fmt(x):
        if pd.isna(x):
            return ""
        return f"{int(round(x)):,}"

    for col in pos_df.columns:
        pos_df[col] = pos_df[col].map(fmt)
        chg_df[col] = chg_df[col].map(fmt)

    latest_chg_date = max(
        res["latest_change_date"] for res in results_by_tenor.values()
    )

    # Use the latest meaningful report date for the position table
    data_week = latest_chg_date
    prev_week = max(res["previous_week_date"] for res in results_by_tenor.values())

    return (
        pos_df,
        chg_df,
        latest_chg_date.strftime("%Y-%m-%d"),
        data_week.strftime("%Y-%m-%d"),
        prev_week.strftime("%Y-%m-%d"),
    )


def build_summary_table(results: Dict[str, object]) -> pd.DataFrame:
    """Build a compact summary table across categories for one tenor."""
    tenor = results["tenor"]
    betas = results["betas"]
    pred = results["predictive"]
    corr = results["correlations"]

    rows = []
    for cat in ["AM", "HF", "Dealer", "Other"]:
        rows.append({
            "category": cat,
            "n": betas.loc[cat, "n"] if cat in betas.index else np.nan,
            "corr": corr.loc[cat, "ref_yield_bps"] if cat in corr.index else np.nan,
            "beta_bps/ct": betas.loc[cat, "slope"] if cat in betas.index else np.nan,
            "beta_R2": betas.loc[cat, "r2"] if cat in betas.index else np.nan,
            "beta_p": betas.loc[cat, "p_value"] if cat in betas.index else np.nan,
            "pred_bps/ct": pred.loc[cat, "slope"] if cat in pred.index else np.nan,
            "pred_R2": pred.loc[cat, "r2"] if cat in pred.index else np.nan,
            "pred_p": pred.loc[cat, "p_value"] if cat in pred.index else np.nan,
        })
    return pd.DataFrame(rows).set_index("category")


def generate_pdf_report(tenors: Optional[List[str]] = None,
                        pdf_path: Optional[Path] = None) -> Path:
    """Generate the institutional futures position analysis PDF report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    if tenors is None:
        tenors = list(TICKER_TERM.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"InstFutPosition_Report_{report_date}.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"InstFutPosition_Report_{report_date}_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        _add_title_page(pdf, tenors)

        # Load all tenor data first
        results_by_tenor = {}
        for tenor in tenors:
            print(f"\nAnalyzing {tenor}...")
            results_by_tenor[tenor] = analyze_tenor(tenor)

        # Cross-tenor position summary at the top
        pos_df, chg_df, latest_chg_date, data_week, prev_week = build_cross_tenor_position_tables(results_by_tenor)

        fig = _df_to_figure(
            pos_df,
            title="Latest Net Positions by Investor Type & Contract",
            subtitle=f"Data week: {data_week}; Previous week: {prev_week}; values in contracts",
            pagesize=(15.0, 5.0),
            fontsize=9,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = _df_to_figure(
            chg_df,
            title="Week-over-Week Position Changes by Investor Type & Contract",
            subtitle=f"Latest report week ending {latest_chg_date}; values in contracts",
            pagesize=(15.0, 5.0),
            fontsize=9,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Recent weekly changes by investor type (last 10 observations)
        recent_figs = plot_recent_changes_by_investor(results_by_tenor, n_obs=10)
        for cat, fig in recent_figs.items():
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for tenor, results in results_by_tenor.items():
            # Summary table page
            summary = build_summary_table(results)
            fig = _df_to_figure(
                summary,
                title=f"{tenor.upper()} Summary: Weekly Position Change vs Yield Change",
                subtitle=(
                    f"Beta = bps yield change per contract position change; "
                    f"Pred = next-week yield change vs current position change; "
                    f"filtered on |ref yield change| > {YIELD_CHANGE_FILTER_BPS} bps"
                ),
                pagesize=(15.0, 6.0),
                fontsize=9,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Correlation matrix page
            fig = _df_to_figure(
                results["correlations"],
                title=f"{tenor.upper()}: Correlation Matrix (Weekly Position Changes vs Yield Change)",
                pagesize=(8.0, 5.0),
                fontsize=9,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Position time series
            fig = plot_positions(results["df"], tenor)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Scatter plots
            fig = plot_position_vs_yield(results["filtered"], tenor)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Weekly position-change bar charts by investor type
            fig = plot_weekly_position_changes(results["weekly_changes"], tenor)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Rolling correlation
            if not results["rolling_corr"].empty:
                fig = plot_rolling_correlation(results["rolling_corr"], tenor)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return pdf_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required.")

    pdf_path = generate_pdf_report()

    print("\n" + "=" * 70)
    print("Institutional Futures Position Analysis Report")
    print("=" * 70)
    print(f"PDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
