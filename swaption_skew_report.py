r"""
Swaption Skew Analysis Report
=============================
Mimics the existing report style in D:\python\kimi_helpers.

Reads:
    D:\python\notebook\data\mbs\mbs_rv.xlsx  sheet "Data1"

Outputs:
    ./swaption_skew_output/Swaption_Skew_Report_Interactive_<date>.html
    Optional PDF with --format pdf

Contents:
    - Summary table of risk reversals vs CT10
    - Combined risk reversal plot with CT10
    - Individual risk-reversal vs CT10 plots
    - Scatter plots of CT10 vs each risk reversal
    - 1y10y ATM volatility time series

Run:
    python swaption_skew_report.py  # interactive HTML
    python swaption_skew_report.py --format pdf
"""

from __future__ import annotations

import warnings
import argparse
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SOURCE_CANDIDATES = [
    Path(r"D:\python\notebook\data\mbs\mbs_rv.xlsx"),
    Path.home() / 'Nutstore' / '1' / 'python' / 'notebook' / 'data' / 'mbs' / 'mbs_rv.xlsx',
]
FNAME = str(next((p for p in _SOURCE_CANDIDATES if p.exists()), _SOURCE_CANDIDATES[0]))
OUTPUT_DIR = Path(__file__).parent / "swaption_skew_output"

OPTION_TENORS = ["3m10y", "1y10y", "3y10y"]
UNDERLYING_COL = "10y"  # CT10 proxy in Data1 sheet
ATM_TENOR = "1y10y"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(file_path: str = FNAME, sheet: str = "Data1") -> pd.DataFrame:
    """Load and clean the Data1 sheet."""
    fpath = Path(file_path)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")

    df = pd.read_excel(fpath, sheet_name=sheet)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=['date']).set_index("date").sort_index()
    if "10y" not in df.columns:
        # Workbook updates occasionally drop the '10y' header; the yield
        # column sits directly after the date column.
        df = df.rename(columns={df.columns[0]: "10y"})
    required = [UNDERLYING_COL] + [f'{tenor}_{side}' for tenor in OPTION_TENORS for side in ['atm', 'p50', 'r50']]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing swaption source columns: {sorted(missing)}')
    df = df[required].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.index.has_duplicates:
        raise ValueError('Duplicate dates in swaption source; resolve before computing observation changes.')
    if df.empty:
        raise ValueError('No dated swaption observations in the workbook.')

    print(f"Loaded {fpath.name} sheet '{sheet}': {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    # Keep the source calendar: do not shorten horizons or fill missing quotes.
    return df


def compute_risk_reversals(df: pd.DataFrame, tenors: List[str] = OPTION_TENORS) -> pd.DataFrame:
    """Compute risk reversals as payer 50bp minus receiver 50bp."""
    rr_df = pd.DataFrame(index=df.index)
    for tenor in tenors:
        rr_df[f"{tenor}_rr"] = df[f"{tenor}_p50"] - df[f"{tenor}_r50"]
    return rr_df


def build_summary(
    df: pd.DataFrame,
    rr_df: pd.DataFrame,
    tenors: List[str] = OPTION_TENORS,
    underlying_col: str = UNDERLYING_COL,
) -> pd.DataFrame:
    """Summary table: latest RR, mean, std, and correlation with CT10."""
    rows = []
    for tenor in tenors:
        rr = rr_df[f"{tenor}_rr"]
        rows.append(
            {
                "tenor": tenor,
                "latest_rr": rr.iloc[-1],
                "rr_mean": rr.mean(),
                "rr_std": rr.std(),
                "corr_with_ct10": rr.corr(df[underlying_col]),
            }
        )
    return pd.DataFrame(rows).set_index("tenor")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _idx_date(idx_value) -> date:
    """Convert a pandas Timestamp to a Python date."""
    return idx_value.date() if hasattr(idx_value, "date") else idx_value


def _common_title_suffix(df: pd.DataFrame) -> str:
    """Return a subtitle string with date range and observation count."""
    return (
        f"Latest: {_idx_date(df.index[-1])}  |  "
        f"Obs: {len(df)}  |  "
        f"CT10 proxy: {UNDERLYING_COL}"
    )


def plot_combined_rr(
    df: pd.DataFrame,
    rr_df: pd.DataFrame,
    tenors: List[str] = OPTION_TENORS,
    underlying_col: str = UNDERLYING_COL,
) -> plt.Figure:
    """Combined risk-reversal time series with CT10 on secondary axis."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    for tenor in tenors:
        ax1.plot(rr_df.index, rr_df[f"{tenor}_rr"], label=f"{tenor} RR")
    ax1.set_ylabel("Risk Reversal (p50 - r50, vol pts)")
    ax1.set_title("Swaption Risk Reversals vs CT10")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        df.index,
        df[underlying_col],
        color="black",
        linestyle="--",
        label=underlying_col,
        alpha=0.7,
    )
    ax2.set_ylabel(f"{underlying_col} yield (%)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_rr_vs_underlying(
    tenor: str,
    df: pd.DataFrame,
    rr_df: pd.DataFrame,
    underlying_col: str = UNDERLYING_COL,
) -> plt.Figure:
    """Dual-axis time series of one risk reversal and CT10."""
    rr_col = f"{tenor}_rr"
    rr = rr_df[rr_col]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(rr.index, rr, color="tab:blue", label=f"{tenor} Risk Reversal")
    ax1.set_ylabel("Risk Reversal (p50 - r50, vol pts)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df[underlying_col], color="tab:orange", label=underlying_col)
    ax2.set_ylabel(f"{underlying_col} yield (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(f"{tenor} Risk Reversal vs {underlying_col}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame,
    rr_df: pd.DataFrame,
    tenors: List[str] = OPTION_TENORS,
    underlying_col: str = UNDERLYING_COL,
) -> plt.Figure:
    """Scatter plots of CT10 vs each risk reversal with trend lines."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, tenor in zip(axes, tenors):
        x = df[underlying_col]
        y = rr_df[f"{tenor}_rr"]
        ax.scatter(x, y, alpha=0.5, s=20)
        corr = x.corr(y)
        ax.set_xlabel(f"{underlying_col} yield (%)")
        ax.set_ylabel(f"{tenor} RR (vol pts)")
        ax.set_title(f"{tenor} RR vs {underlying_col} (corr={corr:.2f})")
        ax.grid(True, alpha=0.3)

        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = x.sort_values()
        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.7)

    fig.suptitle("Risk Reversal vs CT10 — Scatter", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_atm_vol(
    df: pd.DataFrame,
    atm_tenor: str = ATM_TENOR,
) -> plt.Figure:
    """Standalone ATM volatility time series."""
    fig, ax = plt.subplots(figsize=(12, 5))
    atm_col = f"{atm_tenor}_atm"
    ax.plot(df.index, df[atm_col], color="navy", linewidth=1.5)
    ax.set_title(f"{atm_tenor} ATM Volatility")
    ax.set_ylabel("ATM Volatility")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    latest_val = df[atm_col].iloc[-1]
    latest_date = df.index[-1]
    ax.axhline(latest_val, color="red", linestyle="--", alpha=0.5)
    ax.text(latest_date, latest_val, f"  {latest_val:.2f}", color="red", va="center")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDF helpers (mimic mbs_rv_report.py)
# ---------------------------------------------------------------------------
def _column_color_func(series: pd.Series):
    """Pick coloring function based on data range."""
    s = series.dropna()
    if s.empty:
        return lambda x: "#ffffff"

    vmin, vmax = float(s.min()), float(s.max())
    if vmin < 0 < vmax:
        limit = max(abs(vmin), abs(vmax), 0.01)

        def diverging(x):
            norm = (float(x) + limit) / (2 * limit)
            norm = max(0.0, min(1.0, norm))
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "div", ["#4575b4", "#ffffff", "#d73027"]
            )
            return mcolors.to_hex(cmap(norm)[:3])

        return diverging

    def sequential(x):
        if vmax == vmin:
            return "#ffffff"
        norm = (float(x) - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "seq", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26"]
        )
        return mcolors.to_hex(cmap(norm)[:3])

    return sequential


def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    fontsize: int = 9,
    header_color: str = "#40466e",
    row_height: float = 0.35,
    gradient_columns: Optional[List[str]] = None,
    index_col_width: float = 0.12,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib figure (one page)."""
    df = df.reset_index()
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            )

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.84])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.98, ha="left", x=0.04)
    if subtitle:
        ax.text(
            0.04,
            0.93,
            subtitle,
            fontsize=8,
            ha="left",
            transform=fig.transFigure,
            color="#555555",
        )

    table_data = [display_df.columns.tolist()] + display_df.values.tolist()
    if n_cols > 1:
        col_widths = [index_col_width] + [
            (1.0 - index_col_width) / (n_cols - 1)
        ] * (n_cols - 1)
    else:
        col_widths = [1.0]

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )

    total_cells = n_rows + 1
    cell_h = max(row_height, 0.86 / total_cells)
    auto_font = min(fontsize, int(cell_h * 280))
    table.auto_set_font_size(False)
    table.set_fontsize(auto_font)

    gradient_cols = []
    if gradient_columns:
        for gc in gradient_columns:
            if gc in df.columns:
                gradient_cols.append(gc)

    color_funcs = {col_name: _column_color_func(df[col_name]) for col_name in gradient_cols}

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


def _add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    gradient_columns: Optional[List[str]] = None,
    fontsize: int = 9,
    row_height: float = 0.07,
    pagesize: Tuple[float, float] = (10.0, 4.5),
) -> None:
    """Add a table page to the PDF."""
    fig = _df_to_figure(
        df,
        title,
        subtitle=subtitle,
        gradient_columns=gradient_columns,
        fontsize=fontsize,
        row_height=row_height,
        pagesize=pagesize,
        index_col_width=0.12,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_cover_page(pdf: PdfPages, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Cover page with title, data info, and key highlights."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")

    title = "Swaption Skew Analysis Report"
    subtitle = (
        f"Latest date: {_idx_date(df.index[-1])}  |  "
        f"Observations: {len(df)}  |  "
        f"CT10 proxy: {UNDERLYING_COL}"
    )

    ax.text(0.5, 0.92, title, fontsize=22, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.86, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    # Latest RR highlights
    latest_rr = summary["latest_rr"]
    rr_text = "Latest Risk Reversals (vol pts)\n" + "\n".join(
        [f"  {idx}: {val:.2f}" for idx, val in latest_rr.items()]
    )

    contents = (
        "Report contents:\n"
        "1. Summary table (latest RR, mean, std, corr with CT10)\n"
        "2. Combined risk reversal time series vs CT10\n"
        "3. Individual RR vs CT10 plots\n"
        "4. Scatter plots: CT10 vs each RR\n"
        "5. 1y10y ATM volatility time series"
    )

    ax.text(
        0.15,
        0.70,
        rr_text,
        fontsize=11,
        family="monospace",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2ff", edgecolor="#40466e"),
    )
    ax.text(
        0.55,
        0.70,
        contents,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8ee", edgecolor="#8b5a00"),
    )

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    df: pd.DataFrame,
    rr_df: pd.DataFrame,
    summary: pd.DataFrame,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate the Swaption Skew PDF report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"Swaption_Skew_Report_{report_date}.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"Swaption_Skew_Report_{report_date}_{ts}.pdf"

    subtitle = _common_title_suffix(df)

    with PdfPages(pdf_path) as pdf:
        # Cover page
        _add_cover_page(pdf, df, summary)

        # Summary table
        _add_table_page(
            pdf,
            summary,
            "Swaption Risk Reversal Summary",
            subtitle=subtitle,
            gradient_columns=["corr_with_ct10"],
            fontsize=10,
            row_height=0.08,
            pagesize=(10.0, 4.5),
        )

        # Combined RR plot
        fig = plot_combined_rr(df, rr_df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Individual RR plots
        for tenor in OPTION_TENORS:
            fig = plot_rr_vs_underlying(tenor, df, rr_df)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Scatter plots
        fig = plot_scatter(df, rr_df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ATM vol plot
        fig = plot_atm_vol(df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved PDF report to {pdf_path}")
    return pdf_path


def main(report_format='interactive', file_path=FNAME) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(file_path)
    rr_df = compute_risk_reversals(df)
    summary = build_summary(df, rr_df)

    print("\n" + "=" * 70)
    print("Swaption Skew Analysis Summary")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    if report_format == 'interactive':
        from swaption_skew_interactive import build_interactive_report
        return build_interactive_report(df, rr_df, source_path=Path(file_path))
    pdf_path = generate_pdf_report(df, rr_df, summary)
    print(f"PDF report saved to: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['interactive', 'pdf'], default='interactive')
    parser.add_argument('--source', type=Path, default=Path(FNAME), help='Workbook containing Data1')
    args = parser.parse_args()
    main(args.format, args.source)
