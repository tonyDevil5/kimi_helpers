r"""
UST Curve Spread Fair-Value Analysis
=====================================
Runs the existing UST fair-value framework across Treasury curve spreads:
2s5s, 2s10s, 5s10s, 5s30s, 7s30s, 10s30s.

For each spread:
    Y = (long-end yield - short-end yield) * 100   (bps)
    X = [SOFR 1y1y, 5y5y inflation zero, Fed holdings ratio, post-Apr-2025 dummy]

Outputs a PDF with a compact summary table (mkt/mdl/diff/resid-std/ratio)
and a grid of fitted-vs-actual plots.

Run:
    python ust_curve_report.py

Output:
    ./ust_curve_output/UST_Curve_Report.pdf
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from matplotlib.backends.backend_pdf import PdfPages

# Reuse the data-loading and PDF helpers from the UST fair-value report
from ust_fair_value_report import (
    _df_to_figure,
    build_dat_final,
    load_bloomberg_data,
    load_fed_holdings_data,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "ust_curve_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-05-21"

# Predictor columns in mergedDf by position:
#   2: S0490FS 1Y1Y BLC Curncy  (SOFR 1y1y)
#   6: 5y5y Infl Zero
#   7: FedHoldRatio
#  12: dummy (post 2025-04-09)
PREDICTOR_INDICES = [2, 6, 7, 12]

# Treasury yield tickers to add for curve construction
TSY_TICKERS = ["GT2 Govt", "GT5 Govt", "GT7 Govt", "GT30 Govt"]

# Curve pairs to analyse: (long/end, short/end, display_name)
CURVE_PAIRS: List[Tuple[str, str, str]] = [
    ("GT5 Govt", "GT2 Govt", "2s5s"),
    ("GT10 Govt", "GT2 Govt", "2s10s"),
    ("GT10 Govt", "GT5 Govt", "5s10s"),
    ("GT30 Govt", "GT5 Govt", "5s30s"),
    ("GT30 Govt", "GT7 Govt", "7s30s"),
    ("GT30 Govt", "GT10 Govt", "10s30s"),
]

# Dummy break date (tariff/policy regime change)
DUMMY_DATE = "2025-04-09"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_treasury_yields(start_date: str = START_DATE) -> pd.DataFrame:
    """Download 2Y/5Y/7Y/30Y UST yields and add a post-Apr-2025 dummy."""
    from xbbg import blp

    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(TSY_TICKERS)} Treasury yield tickers from Bloomberg...")
    df = blp.bdh(
        tickers=TSY_TICKERS,
        flds=["px_last"],
        start_date=start_date,
        end_date=end_date,
    )
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df["dummy"] = np.where(df.index >= pd.Timestamp(DUMMY_DATE), 1, 0)
    print(f"Loaded Treasury yields: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def build_merged_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Combine main predictors, Fed-holdings ratio and Treasury yields."""
    df_main = load_bloomberg_data(start_date=start_date)
    fed = load_fed_holdings_data(start_date=start_date)
    dat_final = build_dat_final(df_main, fed)
    tsy = load_treasury_yields(start_date=start_date)

    merged = dat_final.join(tsy, how="left")
    merged = merged.ffill().dropna()
    print(f"Merged dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
    return merged


# ---------------------------------------------------------------------------
# Curve OLS
# ---------------------------------------------------------------------------
def run_curve_ols(
    merged_df: pd.DataFrame, crv1: str, crv2: str
) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run OLS for one curve spread. Y is in bps."""
    Y = (merged_df[crv1] - merged_df[crv2]) * 100.0
    X = merged_df.iloc[:, PREDICTOR_INDICES]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return Y, model


def build_summary_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Build the summary table requested by the user."""
    rows = []
    for crv1, crv2, name in CURVE_PAIRS:
        Y, model = run_curve_ols(merged_df, crv1, crv2)
        resid_std = float(np.std(model.resid))
        mkt_val = float(Y.iloc[-1])
        mdl_val = float(model.fittedvalues.iloc[-1])
        diff = mkt_val - mdl_val
        ratio = diff / resid_std if resid_std != 0 else np.nan

        rows.append(
            {
                "curve": name,
                "mkt_val": mkt_val,
                "mdl_val": mdl_val,
                "diff": diff,
                "resid_std": resid_std,
                "ratio": ratio,
                "r2": float(model.rsquared),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.set_index("curve")
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _single_curve_axis(
    ax, merged_df: pd.DataFrame, crv1: str, crv2: str, name: str
) -> None:
    """Plot fitted vs actual for one curve spread on a given axis."""
    Y, model = run_curve_ols(merged_df, crv1, crv2)
    ax.plot(Y.index, Y, color="red", alpha=0.6, label="Actual")
    ax.plot(Y.index, model.fittedvalues, color="blue", alpha=0.6, label="Fitted")
    ax.set_title(name, fontsize=10, weight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    resid_std = float(np.std(model.resid))
    mkt_val = float(Y.iloc[-1])
    mdl_val = float(model.fittedvalues.iloc[-1])
    diff = mkt_val - mdl_val
    ratio = diff / resid_std if resid_std != 0 else np.nan

    annot_text = (
        f"mkt={mkt_val:.1f}\n"
        f"mdl={mdl_val:.1f}\n"
        f"diff={diff:.1f}\n"
        f"ratio={ratio:.2f}"
    )
    ax.text(
        0.98,
        0.03,
        annot_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def plot_all_curves(merged_df: pd.DataFrame) -> plt.Figure:
    """2x3 grid of fitted-vs-actual plots for all curve spreads."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    for ax, (crv1, crv2, name) in zip(axes, CURVE_PAIRS):
        _single_curve_axis(ax, merged_df, crv1, crv2, name)

    fig.suptitle(
        "UST Curve Spreads: Actual vs Model Fitted",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def _idx_date(idx_value) -> date:
    return idx_value.date() if hasattr(idx_value, "date") else idx_value


def generate_pdf_report(
    merged_df: pd.DataFrame, pdf_path: Optional[Path] = None
) -> Path:
    """Generate the UST curve-spread PDF report."""
    import matplotlib.pyplot as plt

    if pdf_path is None:
        pdf_path = OUTPUT_DIR / "UST_Curve_Report.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"UST_Curve_Report_{ts}.pdf"

    summary = build_summary_table(merged_df)
    subtitle = (
        f"Latest: {_idx_date(merged_df.index[-1])}  |  "
        f"Obs: {len(merged_df)}  |  "
        f"Predictors: SOFR 1y1y, 5y5y infl zero, FedHoldRatio, dummy"
    )

    with PdfPages(pdf_path) as pdf:
        # 1. Summary table
        fig = _df_to_figure(
            summary,
            "UST Curve Spread Fair-Value Summary",
            subtitle=subtitle,
            gradient_columns=["diff", "ratio"],
            fontsize=9,
            row_height=0.07,
            pagesize=(11.0, 4.5),
            index_col_width=0.08,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2. Fitted-vs-actual grid
        fig = plot_all_curves(merged_df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_path


def main() -> None:
    merged_df = build_merged_data()
    summary = build_summary_table(merged_df)

    print("\n" + "=" * 70)
    print("UST Curve Spread Fair-Value Summary")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    pdf_path = generate_pdf_report(merged_df)
    print(f"PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
