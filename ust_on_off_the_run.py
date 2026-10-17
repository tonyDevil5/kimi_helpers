r"""
UST On-the-Run / Off-the-Run Treasury Yield Analysis
=====================================================

Reads the 'Data2' tab from:
    D:\python\notebook\data\mbs\mbs_rv.xlsx

For each tenor (2y, 5y, 7y, 10y, 20y, 30y) it plots the current on-the-run
yield together with the old (O), old-old (OO), and old-old-old (OOO) issues.

Output:
    ./mbs_rv_output/MBS_OnTheRun_Report.pdf
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
FNAME = r"D:\python\notebook\data\mbs\mbs_rv.xlsx"
SHEET_NAME = "Data2"
OUTPUT_DIR = Path(__file__).parent / "ust_on_off_the_run_output"

TENORS = ["2y", "5y", "7y", "10y", "20y", "30y"]
SERIES_NAMES = {
    "": "Current",
    " O": "Old (O)",
    " OO": "Old-Old (OO)",
    " OOO": "Old-Old-Old (OOO)",
}

COLORS = {
    "Current": "#1f77b4",
    "Old (O)": "#ff7f0e",
    "Old-Old (OO)": "#2ca02c",
    "Old-Old-Old (OOO)": "#d62728",
}


def load_data(fname: str = FNAME, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Load the Data2 sheet and set the date index."""
    df = pd.read_excel(fname, sheet_name=sheet_name)
    # The first column is the date, usually unnamed.
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def parse_issue_date(val) -> Optional[date]:
    """Convert an Excel issue/maturity date value to a Python date."""
    if pd.isna(val):
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None
        if len(s) == 8 and s.isdigit():
            return date(int(s[:4]), int(s[4:6]), int(s[6:]))
        return pd.to_datetime(s).date()
    # Numeric values in this file are YYYYMMDD integers (e.g. 20260630).
    # Do NOT pass raw integers to pd.to_datetime, which treats them as
    # nanoseconds since 1970-01-01 and returns 1970-01-01.
    try:
        s = str(int(float(val))).zfill(8)
        if len(s) == 8 and s.isdigit():
            return date(int(s[:4]), int(s[4:6]), int(s[6:]))
        return pd.to_datetime(val).date()
    except Exception:
        return None


def parse_coupon(val) -> Optional[float]:
    """Convert a coupon value to float, returning None if missing."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def format_metadata(
    issue: date, cpn: Optional[float], maturity: date
) -> str:
    """Return a short metadata string for a bond series."""
    parts = []
    if issue is not None:
        parts.append(f"Issue {issue.strftime('%Y-%m-%d')}")
    if cpn is not None:
        parts.append(f"Cpn {cpn:.3f}%")
    if maturity is not None:
        parts.append(f"Mat {maturity.strftime('%Y-%m-%d')}")
    return " | ".join(parts) if parts else ""


def get_metadata(df: pd.DataFrame, tenor: str, suffix: str) -> Tuple[str, date, Optional[float], date]:
    """Return label and latest metadata for a tenor/suffix combination."""
    label = SERIES_NAMES[suffix]
    prefix = f"{tenor}{suffix}"
    issue = parse_issue_date(df[f"{prefix} Issue"].iloc[-1])
    cpn = parse_coupon(df[f"{prefix} Cpn"].iloc[-1])
    maturity = parse_issue_date(df[f"{prefix} Mat"].iloc[-1])
    meta = format_metadata(issue, cpn, maturity)
    return label, issue, cpn, maturity, meta


def build_tenor_figure(df: pd.DataFrame, tenor: str) -> plt.Figure:
    """Create a two-panel figure for one tenor: yields and spreads."""
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.suptitle(f"{tenor} Treasury On-the-Run vs Off-the-Run Yields", fontsize=14, fontweight="bold")

    # --- Top panel: yield levels ---
    ax = axes[0]
    meta_lines = []
    latest = {}

    for suffix, label in SERIES_NAMES.items():
        prefix = f"{tenor}{suffix}"
        yield_col = f"{prefix} Yield"
        if yield_col not in df.columns:
            continue

        label, issue, cpn, maturity, meta = get_metadata(df, tenor, suffix)
        color = COLORS[label]
        ax.plot(df.index, df[yield_col], label=label, color=color, linewidth=1.2)
        latest[label] = df[yield_col].iloc[-1]
        if meta:
            meta_lines.append(f"{label}: {meta} (latest {df[yield_col].iloc[-1]:.3f}%)")

    ax.set_ylabel("Yield (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df.index.min(), df.index.max())

    # --- Bottom panel: spreads vs current ---
    ax2 = axes[1]
    current_col = f"{tenor} Yield"
    for suffix in [" O", " OO", " OOO"]:
        prefix = f"{tenor}{suffix}"
        yield_col = f"{prefix} Yield"
        if yield_col not in df.columns:
            continue
        label = SERIES_NAMES[suffix]
        spread = df[yield_col] - df[current_col]
        color = COLORS[label]
        ax2.plot(df.index, spread, label=f"{label} - Current", color=color, linewidth=1.2)

    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Spread vs Current (bps)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df.index.min(), df.index.max())

    # Convert y-axis to bps for spreads
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}"))

    # Metadata text below the plots
    meta_text = "\n".join(meta_lines)
    fig.text(
        0.12,
        0.02,
        meta_text,
        fontsize=8,
        verticalalignment="bottom",
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    return fig


def _rolling_z_score(series: pd.Series, window: int = 252) -> float:
    """Return the latest z-score of `series` over the trailing `window`."""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score.iloc[-1]


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary DataFrame of latest yields, spreads, min/max, and z-scores."""
    rows = []
    for tenor in TENORS:
        current_col = f"{tenor} Yield"
        if current_col not in df.columns:
            continue
        current_yield = df[current_col].iloc[-1]

        row = {
            "Tenor": tenor,
            "Current\n(%)": round(current_yield, 3),
        }
        for suffix in [" O", " OO", " OOO"]:
            prefix = f"{tenor}{suffix}"
            yield_col = f"{prefix} Yield"
            if yield_col in df.columns:
                short = suffix.strip()  # "O", "OO", "OOO"
                off_yield = df[yield_col].iloc[-1]
                spread_bps = (df[yield_col] - df[current_col]) * 100
                row[f"{short}\nYld%"] = round(off_yield, 3)
                row[f"{short}\nSprd"] = round(spread_bps.iloc[-1], 1)
                row[f"{short}\nMin"] = round(spread_bps.min(), 1)
                row[f"{short}\nMax"] = round(spread_bps.max(), 1)
                row[f"{short}\nZ"] = round(_rolling_z_score(spread_bps), 2)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def _z_score_color(z: float, vmax: float = 3.0) -> str:
    """Map a z-score to a matplotlib color string (diverging, centered at 0)."""
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("zscore", ["#2166ac", "#ffffff", "#b2182b"])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    rgba = cmap(norm(z))
    return mcolors.to_hex(rgba)


def build_summary_figure(summary_df: pd.DataFrame, start_date, end_date) -> plt.Figure:
    """Create a summary figure with a table of latest yields/spreads."""
    # Extra-wide landscape so all columns fit without overlapping.
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.axis("off")

    title = f"UST On-the-Run / Off-the-Run Summary\n{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Build table with multi-line column labels for readability.
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#1f77b4"] * len(summary_df.columns),
        bbox=[0.03, 0.22, 0.94, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Identify z-score columns for heatmap coloring.
    z_cols = [j for j, col in enumerate(summary_df.columns) if col.endswith("\nZ")]

    # Determine a sensible z-score cap for the color scale.
    all_z = []
    for j in z_cols:
        all_z.extend([abs(summary_df.iloc[i, j]) for i in range(len(summary_df))])
    z_max = max(all_z) if all_z else 1.0
    z_cap = max(2.0, min(4.0, z_max))

    # Style header: multi-line text, no rotation, large enough cell.
    for i, key in enumerate(summary_df.columns):
        cell = table[(0, i)]
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(
            color="white",
            fontweight="bold",
            fontsize=9,
            rotation=0,
            ha="center",
            va="center",
            linespacing=1.2,
        )
        cell.set_height(0.16)

    # Style data cells
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            cell.set_fontsize(9)
            cell.set_height(0.08)
            # Heatmap z-score columns.
            if j in z_cols:
                z_val = summary_df.iloc[i - 1, j]
                cell.set_facecolor(_z_score_color(z_val, vmax=z_cap))
                cell.set_text_props(fontweight="bold")

    # Widen columns and add borders
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        cell.set_width(0.10)

    # Footnote explaining z-score window.
    footnote = "Z = 252-day rolling z-score of the spread. Min/Max = full-history min/max of the spread in bps."
    fig.text(
        0.03,
        0.02,
        footnote,
        fontsize=9,
        verticalalignment="bottom",
        style="italic",
    )

    plt.tight_layout()
    return fig


def build_issue_date_analysis_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Event-study analysis of 10Y UST spreads after each current-bond issue date.

    For each 10Y issue cycle, plots the current bond's spread to O, OO, and OOO
    as a function of days since issue.
    """
    tenor = "10y"
    current_yield_col = f"{tenor} Yield"
    issue_col = f"{tenor} Issue"

    # Identify issue-date change points.
    issue_dates = df[issue_col].where(df[issue_col].ne(df[issue_col].shift())).dropna()

    # Spreads across the liquidity/age curve for each issue cycle.
    spreads = {
        "Current vs Old (O)": (df[current_yield_col] - df[f"{tenor} O Yield"]) * 100,
        "Old (O) vs Old-Old (OO)": (df[f"{tenor} O Yield"] - df[f"{tenor} OO Yield"]) * 100,
        "Old-Old (OO) vs Old-Old-Old (OOO)": (df[f"{tenor} OO Yield"] - df[f"{tenor} OOO Yield"]) * 100,
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "10Y UST: On-the-Run Premium Across Issue Cycles",
        fontsize=14,
        fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(issue_dates)))

    for ax, (label, spread_series) in zip(axes, spreads.items()):
        for (issue_first_seen, issue_val), color in zip(issue_dates.items(), colors):
            # Use the first date this issue value appears in the data as day 0.
            # This avoids gaps when the issue value appears before the formal issue date.
            issue_ts = issue_first_seen

            # Window: from first appearance to just before the next issue appears.
            next_issue_dates = issue_dates.index[issue_dates.index > issue_first_seen]
            if len(next_issue_dates) > 0:
                end_ts = next_issue_dates[0]
            else:
                end_ts = df.index[-1]

            window = spread_series.loc[issue_ts:end_ts]
            if window.empty:
                continue

            days_since_issue = (window.index - issue_ts).days
            issue_d = parse_issue_date(issue_val)
            legend_label = issue_d.strftime("%Y-%m-%d") if issue_d else str(issue_val)
            ax.plot(
                days_since_issue,
                window.values,
                label=legend_label,
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )
            # Mark the starting point so each cycle is easy to follow.
            ax.scatter(
                days_since_issue[0],
                window.values[0],
                color=color,
                s=20,
                zorder=5,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{label}\nSpread (bps)")
        ax.grid(True, alpha=0.3)

    # Shared horizontal legend below the plots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Issue Date",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=8,
        title_fontsize=9,
    )

    axes[-1].set_xlabel("Days Since First Appearance in Data")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    return fig


def run_report() -> str:
    """Generate the PDF report and return the output path."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = date.today().strftime("%Y%m%d")
    pdf_path = OUTPUT_DIR / f"UST_OnOffTheRun_Report_{report_date}.pdf"

    # Remove any previously generated report for the same date so only one PDF per date remains.
    for old_file in OUTPUT_DIR.glob(f"UST_OnOffTheRun_Report_{report_date}*.pdf"):
        try:
            old_file.unlink()
        except PermissionError:
            print(f"Warning: could not remove open file {old_file}")

    print("Loading MBS RV Data2 sheet...")
    df = load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols, {df.index.min().date()} to {df.index.max().date()}")

    summary_df = build_summary_table(df)

    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig_summary = build_summary_figure(summary_df, df.index.min().date(), df.index.max().date())
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)

        # One page per tenor
        for tenor in TENORS:
            print(f"Building {tenor} chart...")
            fig = build_tenor_figure(df, tenor)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # 10Y issue-date event study
        print("Building 10Y issue-date analysis...")
        fig_issue = build_issue_date_analysis_figure(df)
        pdf.savefig(fig_issue, bbox_inches="tight")
        plt.close(fig_issue)

    print(f"\nPDF report saved to: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    run_report()
