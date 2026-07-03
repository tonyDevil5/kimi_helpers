r"""
MBS RV Analysis Report
======================
Standardized extraction of the post-fName logic from MBS_OAS_RV.ipynb.

This script loads:
    D:\python\notebook\data\mbs\mbs_rv.xlsx

and produces a nicely formatted PDF report with:
    - coupon swap / fly z-scores
    - G2/FN swap z-scores
    - historical coupon swap / fly vs CT10 (time series + scatter)
    - yield basis z-scores
    - performance summaries
    - OLS fair-value regression
    - current-coupon quadratic curve fit

Run:
    python mbs_rv_report.py

Output:
    ./mbs_rv_output/MBS_RV_Report.pdf

Outputs are written to ./mbs_rv_output/ by default.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
FNAME = r"D:\python\notebook\data\mbs\mbs_rv.xlsx"
OUTPUT_DIR = Path(__file__).parent / "mbs_rv_output"

COUPONS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

# Reference columns used across analyses
REF_RATE_CURVE = {"short": "CT2", "long": "CT10"}
REF_VOL = "1y10y"
REF_BASIS = "CT10"

# Default lookbacks and score reference dates
DEFAULT_SCORE_INDICES = [-1, -2, -5, -10, -20, -30, -40]
DEFAULT_SWAP_LOOKBACK = 40
DEFAULT_ROLL_LOOKBACK = 90
DEFAULT_BASIS_LOOKBACK = 60
DEFAULT_PERF_WINDOWS = [-1, -5, -10, -35, -60, -90, -120, -200]

# Rolling performance z-score section
ROLL_SUM_WINDOW = 5          # days for rolling performance sum
ROLL_SUM_Z_WINDOW = 20       # days for z-score lookback on the rolling sum

# Current-coupon basis analysis
CC_TICKERS = {
    "Conventional": ".FNCC105 G Index",
    "Ginnie": ".G2CC105 G Index",
}
CC_PREDICTORS = ["10y", "2s10s", "1y10y_vol"]
CC_LAG_PREDICTORS = False  # use t-1 predictors for t response
CC_ROLLING_WINDOW = 120    # days for rolling current-coupon basis OLS

# Per-coupon yield basis OLS (same predictors as CC basis, applied to FN/G2 coupons)
COUPON_BASIS_AGENCIES = ["FN", "G2"]
COUPON_BASIS_PREDICTORS = CC_PREDICTORS
COUPON_BASIS_LAG_PREDICTORS = CC_LAG_PREDICTORS

# Historical coupon-swap / fly plot section (added after the cpn/fly tables)
HIST_PLOT_MODE = "fly"          # "cpn" or "fly"
HIST_CPN_PAIR = ("G2 5.5", "FN 5.5")   # cpn1, cpn2 (without the " Price" suffix)
HIST_FLY_TRIPLE = ("G2 3", "G2 3.5", "G2 4")  # front, middle, back (without suffix)
HIST_REF_COLS = ["CT10", "1y10y"]      # one column = 1 ref; two columns = 2x2 page


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mbs_rv(fpath: str | Path = FNAME) -> pd.DataFrame:
    """Load and clean the MBS RV Excel file."""
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"MBS RV file not found: {fpath}")

    dat = pd.read_excel(fpath, index_col=0)
    dat.index = pd.to_datetime(dat.index)
    dat = dat.dropna()
    print(f"Loaded {fpath.name}: {dat.shape[0]} rows x {dat.shape[1]} cols, "
          f"{dat.index[0].date()} to {dat.index[-1].date()}")
    return dat


# ---------------------------------------------------------------------------
# TSY OAS z-score section (Data tab)
# ---------------------------------------------------------------------------
TSY_OAS_LOOKBACK = 252  # ~1 year trading days


def load_tsy_oas_data(fpath: str | Path = FNAME) -> pd.DataFrame:
    """Load FN/G2 TSY OAS columns from the 'Data' tab."""
    fpath = Path(fpath)
    df = pd.read_excel(fpath, sheet_name="Data")
    # First column is the date regardless of header
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    oas_cols = [c for c in df.columns if c.endswith(" TSY OAS")]
    oas_df = df[oas_cols].copy()
    oas_df = oas_df.dropna()
    print(f"Loaded TSY OAS data: {oas_df.shape[0]} rows x {oas_df.shape[1]} cols, "
          f"{oas_df.index[0].date()} to {oas_df.index[-1].date()}")
    return oas_df


def _parse_oas_col_name(col: str) -> Tuple[str, float]:
    """Parse 'FN 2.5 TSY OAS' -> ('FN', 2.5) or 'G2 6.5 TSY OAS' -> ('G2', 6.5)."""
    parts = col.split()
    agency = parts[0]
    coupon = float(parts[1])
    return agency, coupon


def build_tsy_oas_summary(oas_df: pd.DataFrame, lookback: int = TSY_OAS_LOOKBACK) -> pd.DataFrame:
    """
    For each FN/G2 coupon, compute latest OAS and z-score vs trailing 1y history.
    """
    rows = []
    for col in sorted(oas_df.columns):
        agency, coupon = _parse_oas_col_name(col)
        series = oas_df[col]
        if len(series) < lookback:
            continue
        hist = series.iloc[-lookback:]
        current = float(series.iloc[-1])
        hist_mean = float(hist.mean())
        hist_std = float(hist.std(ddof=1))
        zscore = (current - hist_mean) / hist_std if hist_std != 0 else np.nan
        rows.append({
            "agency": agency,
            "coupon": coupon,
            "current_oas": current,
            "hist_mean": hist_mean,
            "hist_std": hist_std,
            "z_score": zscore,
            "obs": len(hist),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(["agency", "coupon"]).set_index(["agency", "coupon"])
    return summary


# ---------------------------------------------------------------------------
# Current-coupon basis OLS
# ---------------------------------------------------------------------------
def load_ust_yield_vol_data(fpath: str | Path = FNAME) -> pd.DataFrame:
    """Load UST yield / vol predictors from the 'Data1' tab."""
    fpath = Path(fpath)
    df = pd.read_excel(fpath, sheet_name="Data1")
    # First column is the date regardless of its header
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # 2s10s slope = 10y - 2y; 1y10y vol from the ATM column
    predictors = pd.DataFrame(
        {
            "10y": df["10y"],
            "2s10s": df["10y"] - df["2y"],
            "1y10y_vol": df["1y10y_atm"],
        },
        index=df.index,
    )
    return predictors.dropna()


def load_current_coupon_oas(start_date: str, end_date: str) -> pd.DataFrame:
    """Download current-coupon OAS from Bloomberg."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    tickers = list(CC_TICKERS.values())
    print(f"Downloading {len(tickers)} current-coupon tickers from Bloomberg...")
    df = blp.bdh(
        tickers=tickers,
        flds=["px_last"],
        start_date=start_date,
        end_date=end_date,
    )
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    # Bloomberg returns OAS in percent; convert to bps for reporting
    df = df * 100.0
    df = df.rename(
        columns={
            CC_TICKERS["Conventional"]: "FNCC",
            CC_TICKERS["Ginnie"]: "G2CC",
        }
    )
    print(f"Loaded current-coupon OAS: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def build_current_coupon_basis_df(fpath: str | Path = FNAME) -> pd.DataFrame:
    """Merge UST predictors with current-coupon OAS."""
    predictors = load_ust_yield_vol_data(fpath)
    start_date = predictors.index[0].strftime("%Y-%m-%d")
    end_date = predictors.index[-1].strftime("%Y-%m-%d")

    cc = load_current_coupon_oas(start_date, end_date)
    merged = predictors.join(cc, how="inner").dropna()
    print(f"Current-coupon basis dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
    return merged


def run_current_coupon_ols(
    df_basis: pd.DataFrame, response_col: str
) -> Tuple[pd.Series, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run full-sample OLS for one current-coupon series.

    Predictors are lagged by one day if CC_LAG_PREDICTORS is True, matching the
    HK-hours convention that the MBS current coupon does not tick on the local date.
    """
    X = df_basis[CC_PREDICTORS].copy()
    if CC_LAG_PREDICTORS:
        X = X.shift(1)
    Y = df_basis[response_col]

    df = pd.concat([Y, X], axis=1).dropna()
    Y = df[response_col]
    X = sm.add_constant(df[CC_PREDICTORS])
    model = sm.OLS(Y, X).fit()
    return Y, X, model


def build_current_coupon_summary(df_basis: pd.DataFrame) -> pd.DataFrame:
    """Build summary table for Conventional and Ginnie current-coupon OAS."""
    rows = []
    for col, name in [("FNCC", "Conventional"), ("G2CC", "Ginnie")]:
        if col not in df_basis.columns:
            continue
        Y, X, model = run_current_coupon_ols(df_basis, col)
        resid_std = float(np.std(model.resid))
        mkt = float(Y.iloc[-1])
        mdl = float(model.fittedvalues.iloc[-1])
        diff = mkt - mdl
        rows.append(
            {
                "cc_type": name,
                "mkt_oas": mkt,
                "mdl_oas": mdl,
                "diff": diff,
                "resid_std": resid_std,
                "resid_z": diff / resid_std if resid_std != 0 else np.nan,
                "r2": float(model.rsquared),
                "obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows).set_index("cc_type")


def plot_current_coupon_fitted(df_basis: pd.DataFrame) -> plt.Figure:
    """Side-by-side fitted-vs-actual plots for Conventional and Ginnie CC OAS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    for ax, col, name in zip(
        axes, ["FNCC", "G2CC"], ["Conventional (.FNCC105)", "Ginnie (.G2CC105)"]
    ):
        if col not in df_basis.columns:
            continue
        Y, X, model = run_current_coupon_ols(df_basis, col)
        ax.plot(Y.index, Y, color="red", alpha=0.7, label="Actual OAS")
        ax.plot(
            Y.index, model.fittedvalues, color="blue", alpha=0.7, label="Fitted OAS"
        )
        ax.set_title(name, fontsize=11, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("OAS (bps)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        resid_std = float(np.std(model.resid))
        mkt = float(Y.iloc[-1])
        mdl = float(model.fittedvalues.iloc[-1])
        diff = mkt - mdl
        ratio = diff / resid_std if resid_std != 0 else np.nan
        annot_text = (
            f"mkt={mkt:.1f}\n"
            f"mdl={mdl:.1f}\n"
            f"diff={diff:.1f}\n"
            f"z={ratio:.2f}"
        )
        ax.text(
            0.98,
            0.03,
            annot_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle(
        "Current-Coupon Basis: Actual vs Model Fitted OAS",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig


def build_current_coupon_params_table(df_basis: pd.DataFrame) -> pd.DataFrame:
    """Return regression coefficients / std err / t-stat / p-value for each CC type."""
    rows = []
    for col, name in [("FNCC", "Conventional"), ("G2CC", "Ginnie")]:
        if col not in df_basis.columns:
            continue
        Y, X, model = run_current_coupon_ols(df_basis, col)
        for param in model.params.index:
            rows.append(
                {
                    "cc_type": name,
                    "feature": param,
                    "coef": float(model.params[param]),
                    "std_err": float(model.bse[param]),
                    "t_stat": float(model.tvalues[param]),
                    "p_value": float(model.pvalues[param]),
                }
            )
    return pd.DataFrame(rows)


def run_rolling_current_coupon_basis(
    df_basis: pd.DataFrame, window: int = 250
) -> pd.DataFrame:
    """Rolling-window OLS for current-coupon basis; returns mkt/mdl/residual time series."""
    results = []
    for col, name in [("FNCC", "Conventional"), ("G2CC", "Ginnie")]:
        if col not in df_basis.columns:
            continue
        Y = df_basis[col]
        X = df_basis[CC_PREDICTORS].copy()
        if CC_LAG_PREDICTORS:
            X = X.shift(1)
        data = pd.concat([Y, X], axis=1).dropna()

        for end_idx in range(window, len(data) + 1):
            win = data.iloc[end_idx - window : end_idx]
            y_win = win[col]
            x_win = sm.add_constant(win[CC_PREDICTORS])
            m = sm.OLS(y_win, x_win).fit()

            actual = float(y_win.iloc[-1])
            fitted = float(m.fittedvalues.iloc[-1])
            resid_std = float(np.std(m.resid))
            resid = actual - fitted
            results.append(
                {
                    "date": win.index[-1],
                    "cc_type": name,
                    "mkt_val": actual,
                    "mdl_val": fitted,
                    "resid": resid,
                    "resid_std": resid_std,
                    "resid_z": resid / resid_std if resid_std != 0 else np.nan,
                }
            )

    return pd.DataFrame(results).set_index("date").sort_index()


def plot_rolling_current_coupon_basis(
    rolling_df: pd.DataFrame, window: int = 250
) -> plt.Figure:
    """Plot rolling actual vs fitted OAS for Conventional and Ginnie."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, name in zip(axes, ["Conventional", "Ginnie"]):
        sub = rolling_df[rolling_df["cc_type"] == name]
        ax.plot(sub.index, sub["mkt_val"], color="red", alpha=0.7, label="Actual OAS")
        ax.plot(
            sub.index, sub["mdl_val"], color="blue", alpha=0.7, label="Fitted OAS"
        )
        ax.set_title(f"{name}  ({window}-day rolling window)", fontsize=11, weight="bold")
        ax.set_ylabel("OAS (bps)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        latest = sub.iloc[-1]
        annot_text = (
            f"mkt={latest['mkt_val']:.1f}\n"
            f"mdl={latest['mdl_val']:.1f}\n"
            f"diff={latest['resid']:.1f}\n"
            f"z={latest['resid_z']:.2f}"
        )
        ax.text(
            0.98,
            0.03,
            annot_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle(
        f"Rolling Current-Coupon Basis: Actual vs Model Fitted OAS ({window}-day window)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_conv_ginnie_cc_spread(
    df_basis: pd.DataFrame, window: int = 90
) -> plt.Figure:
    """Plot Conventional minus Ginnie current-coupon OAS spread with rolling z-score."""
    spread = df_basis["FNCC"] - df_basis["G2CC"]
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: spread
    ax1 = axes[0]
    ax1.plot(spread.index, spread, color="navy", alpha=0.8, label="Conv - Ginnie OAS")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("OAS spread (bps)")
    ax1.set_title(
        f"Conv vs Ginnie Current-Coupon OAS Spread ({window}-day z-score)",
        fontsize=12,
        weight="bold",
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom: z-score
    ax2 = axes[1]
    ax2.plot(zscore.index, zscore, color="purple", alpha=0.8, label="z-score")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(1, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(-1, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Current annotation
    latest_diff = float(spread.iloc[-1])
    latest_z = float(zscore.iloc[-1])
    annot_text = f"current diff={latest_diff:.1f}\ncurrent z={latest_z:.2f}"
    ax1.text(
        0.98,
        0.03,
        annot_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
def build_coupon_basis_df(
    dat: pd.DataFrame,
    predictors: pd.DataFrame,
    agencies: List[str] = COUPON_BASIS_AGENCIES,
    ref_basis: str = REF_BASIS,
) -> pd.DataFrame:
    """Merge UST predictors with per-coupon yield basis vs CT10 for each agency."""
    basis_frames = [predictors]
    for agency in agencies:
        for cpn in COUPONS:
            yield_col = _col(agency, cpn, "Yield")
            if yield_col not in dat.columns or ref_basis not in dat.columns:
                continue
            basis = (dat[yield_col] - dat[ref_basis]) * 100.0
            basis.name = f"{agency} {cpn:g}"
            basis_frames.append(basis)

    df = pd.concat(basis_frames, axis=1).dropna()
    print(f"Coupon basis dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def run_coupon_basis_ols(
    df_basis: pd.DataFrame, response_col: str
) -> Tuple[pd.Series, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run full-sample OLS for one coupon yield-basis series."""
    X = df_basis[COUPON_BASIS_PREDICTORS].copy()
    if COUPON_BASIS_LAG_PREDICTORS:
        X = X.shift(1)
    Y = df_basis[response_col]

    df = pd.concat([Y, X], axis=1).dropna()
    Y = df[response_col]
    X = sm.add_constant(df[COUPON_BASIS_PREDICTORS])
    model = sm.OLS(Y, X).fit()
    return Y, X, model


def build_coupon_basis_summary(df_basis: pd.DataFrame) -> pd.DataFrame:
    """Summary table for every agency-coupon yield-basis OLS."""
    rows = []
    response_cols = [c for c in df_basis.columns if c not in COUPON_BASIS_PREDICTORS]
    for col in response_cols:
        Y, X, model = run_coupon_basis_ols(df_basis, col)
        resid_std = float(np.std(model.resid))
        mkt = float(Y.iloc[-1])
        mdl = float(model.fittedvalues.iloc[-1])
        diff = mkt - mdl
        rows.append(
            {
                "structure": col,
                "mkt_val": mkt,
                "mdl_val": mdl,
                "diff": diff,
                "resid_std": resid_std,
                "resid_z": diff / resid_std if resid_std != 0 else np.nan,
                "r2": float(model.rsquared),
                "obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows).set_index("structure")


def build_coupon_basis_params_table(df_basis: pd.DataFrame) -> pd.DataFrame:
    """Regression parameters for every agency-coupon yield-basis OLS."""
    rows = []
    response_cols = [c for c in df_basis.columns if c not in COUPON_BASIS_PREDICTORS]
    for col in response_cols:
        Y, X, model = run_coupon_basis_ols(df_basis, col)
        for param in model.params.index:
            rows.append(
                {
                    "structure": col,
                    "feature": param,
                    "coef": float(model.params[param]),
                    "std_err": float(model.bse[param]),
                    "t_stat": float(model.tvalues[param]),
                    "p_value": float(model.pvalues[param]),
                }
            )
    return pd.DataFrame(rows)


def build_coupon_basis_coef_matrix(df_basis: pd.DataFrame) -> pd.DataFrame:
    """
    Wide coefficient matrix: rows = coupons, columns = agency-predictor pairs.

    Excludes the constant. Useful for comparing how a given predictor's
    sensitivity varies across coupons and agencies.
    """
    response_cols = [c for c in df_basis.columns if c not in COUPON_BASIS_PREDICTORS]
    records = []
    for col in response_cols:
        parts = col.split()
        if len(parts) != 2:
            continue
        agency, cpn_str = parts
        Y, X, model = run_coupon_basis_ols(df_basis, col)
        row = {"coupon": float(cpn_str), "agency": agency}
        for feat in COUPON_BASIS_PREDICTORS:
            row[feat] = float(model.params.get(feat, np.nan))
        records.append(row)

    df = pd.DataFrame(records).sort_values(["agency", "coupon"])
    if df.empty:
        return pd.DataFrame()

    matrix = df.pivot(index="coupon", columns="agency", values=COUPON_BASIS_PREDICTORS)
    # Flatten columns to "FN 10y", "G2 10y", "FN 2s10s", ...
    matrix.columns = [f"{agency} {feat}" for feat, agency in matrix.columns]
    matrix = matrix.reindex([c for c in COUPONS if c in matrix.index])
    return matrix


def plot_coupon_basis_fitted(df_basis: pd.DataFrame) -> List[plt.Figure]:
    """Actual-vs-fitted plots per agency (2x4 grid of coupons)."""
    response_cols = [c for c in df_basis.columns if c not in COUPON_BASIS_PREDICTORS]
    agency_groups: Dict[str, List[str]] = {}
    for col in response_cols:
        agency = col.split()[0]
        agency_groups.setdefault(agency, []).append(col)

    figures = []
    for agency, cols in agency_groups.items():
        # sort by coupon numeric value
        cols_sorted = sorted(cols, key=lambda x: float(x.split()[1]))
        n_cols = len(cols_sorted)
        nrows = int(np.ceil(n_cols / 4))
        ncols = min(n_cols, 4)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), sharex=False)
        if n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, col in zip(axes, cols_sorted):
            Y, X, model = run_coupon_basis_ols(df_basis, col)
            ax.plot(Y.index, Y, color="red", alpha=0.7, label="Actual", linewidth=1.0)
            ax.plot(Y.index, model.fittedvalues, color="blue", alpha=0.7, label="Fitted", linewidth=1.0)
            ax.set_title(col, fontsize=9, weight="bold")
            ax.set_ylabel("Basis (bps)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")

            resid_std = float(np.std(model.resid))
            diff = float(Y.iloc[-1] - model.fittedvalues.iloc[-1])
            ax.text(
                0.98, 0.03,
                f"mkt={Y.iloc[-1]:.1f}\nmdl={model.fittedvalues.iloc[-1]:.1f}\nz={diff / resid_std if resid_std else np.nan:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # hide unused subplots
        for ax in axes[n_cols:]:
            ax.axis("off")

        fig.suptitle(
            f"Per-Coupon Yield Basis OLS: {agency} MBS",
            fontsize=13,
            weight="bold",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        figures.append(fig)

    return figures


def _col(agency: str, cpn: float, field: str) -> str:
    """Build a column name like 'FN 5.5 Price'."""
    return f"{agency} {cpn:g} {field}"


def price_cols(agency: str) -> List[str]:
    return [_col(agency, c, "Price") for c in COUPONS]


def drop_cols(agency: str) -> List[str]:
    return [_col(agency, c, "Drop") for c in COUPONS]


def yield_cols(agency: str) -> List[str]:
    return [_col(agency, c, "Yield") for c in COUPONS]


def perf_cols(agency: str) -> List[str]:
    return [_col(agency, c, "Perf") for c in COUPONS]


def get_block_columns(dat: pd.DataFrame, agency: str, field: str) -> List[str]:
    """Return columns for a given agency and field that actually exist in the data."""
    candidates = [_col(agency, c, field) for c in COUPONS]
    return [c for c in candidates if c in dat.columns]


# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------
def get_series_zscore(ts: pd.Series, horizon: int, score_idx: int = -1) -> float:
    """
    Z-score of the observation at score_idx over the trailing `horizon` window.

    Parameters
    ----------
    ts : pd.Series
        Input time series.
    horizon : int
        Number of trailing observations to use for mean/std.
    score_idx : int, default -1
        Index of the observation to score (Python-style, relative to series end).

    Returns
    -------
    float
        Z-score of the selected observation.
    """
    recent = ts.iloc[-horizon:]
    z = stats.zscore(recent, nan_policy="omit")
    # z is a numpy array; preserve index
    z_series = pd.Series(z, index=recent.index)
    return float(z_series.iloc[score_idx])


# ---------------------------------------------------------------------------
# Coupon swap / fly / G2-FN swap scoring
# ---------------------------------------------------------------------------
def _score_row(
    name: str,
    series: pd.Series,
    lookback: int,
    score_idx: int,
    date: pd.Timestamp,
) -> Dict:
    return {
        "date": date,
        "name": name,
        "vals": float(series.iloc[score_idx]),
        "scores": float(get_series_zscore(series, lookback, score_idx)),
    }


def calc_cpn_swap_scores(
    dat: pd.DataFrame,
    agency: str,
    lookback: int = DEFAULT_SWAP_LOOKBACK,
    score_idx: int = -1,
) -> pd.DataFrame:
    """Calculate adjacent coupon swap z-scores for one agency."""
    cols = get_block_columns(dat, agency, "Price")
    rows = []
    date = dat.index[score_idx]
    for i in range(len(cols) - 1):
        name = f"{cols[i+1]}_{cols[i]}"
        diff = (dat[cols[i+1]] - dat[cols[i]]) * 32
        rows.append(_score_row(name, diff, lookback, score_idx, date))
    return pd.DataFrame(rows)


def calc_cpn_fly_scores(
    dat: pd.DataFrame,
    agency: str,
    lookback: int = DEFAULT_SWAP_LOOKBACK,
    score_idx: int = -1,
) -> pd.DataFrame:
    """Calculate butterfly (2*mid - front - back) z-scores for one agency."""
    cols = get_block_columns(dat, agency, "Price")
    rows = []
    date = dat.index[score_idx]
    for i in range(len(cols) - 2):
        front, middle, back = cols[i], cols[i + 1], cols[i + 2]
        name = f"{front}_{middle}_{back}_fly"
        fly = (2 * dat[middle] - dat[front] - dat[back]) * 32
        rows.append(_score_row(name, fly, lookback, score_idx, date))
    return pd.DataFrame(rows)


def calc_g2fn_swap_scores(
    dat: pd.DataFrame,
    lookback: int = DEFAULT_SWAP_LOOKBACK,
    score_idx: int = -1,
) -> pd.DataFrame:
    """Calculate G2 vs FN same-coupon swap z-scores."""
    g2_cols = get_block_columns(dat, "G2", "Price")
    fn_cols = get_block_columns(dat, "FN", "Price")
    if len(g2_cols) != len(fn_cols):
        raise ValueError("G2 and FN price columns must line up 1-to-1.")

    rows = []
    date = dat.index[score_idx]
    for g2_col, fn_col in zip(g2_cols, fn_cols):
        name = f"{g2_col}_{fn_col}"
        diff = (dat[g2_col] - dat[fn_col]) * 32
        rows.append(_score_row(name, diff, lookback, score_idx, date))
    return pd.DataFrame(rows)


def calc_all_cpn_scores(
    dat: pd.DataFrame,
    lookback: int = DEFAULT_SWAP_LOOKBACK,
    score_indices: List[int] = None,
) -> pd.DataFrame:
    """
    Combine FN swaps, G2 swaps, G2/FN swaps, FN flys and G2 flys
    across multiple reference dates.
    """
    score_indices = score_indices or DEFAULT_SCORE_INDICES
    frames = []
    for score_idx in score_indices:
        frames.extend([
            calc_cpn_swap_scores(dat, "FN", lookback, score_idx),
            calc_cpn_swap_scores(dat, "G2", lookback, score_idx),
            calc_g2fn_swap_scores(dat, lookback, score_idx),
            calc_cpn_fly_scores(dat, "FN", lookback, score_idx),
            calc_cpn_fly_scores(dat, "G2", lookback, score_idx),
        ])
    return pd.concat(frames, ignore_index=True)


def build_cpn_fly_summary_table(
    dat: pd.DataFrame,
    lookback: int = DEFAULT_SWAP_LOOKBACK,
    score_indices: List[int] = None,
) -> pd.DataFrame:
    """
    Reproduce the notebook-style wide table from cell 24:
    one row per swap/fly name, columns like t0_vals, t0_scores, t1_vals, ...
    """
    score_indices = score_indices or DEFAULT_SCORE_INDICES
    scores_list = []

    for i, score_idx in enumerate(score_indices):
        df = calc_all_cpn_scores(dat, lookback=lookback, score_indices=[score_idx])
        df = df.set_index("name")[["vals", "scores"]]
        df.columns = pd.MultiIndex.from_product([[f"t{i}"], df.columns])
        scores_list.append(df)

    tmp = pd.concat(scores_list, axis=1)
    # Flatten multi-index columns to t0_vals, t0_scores, t1_vals, t1_scores, ...
    tmp.columns = [f"{second}_{first}" for first, second in tmp.columns]
    tmp["carry"] = [_calc_latest_carry_for_swap_fly(dat, name) for name in tmp.index]
    return tmp


def _parse_swap_fly_name(name: str) -> Tuple[str, ...]:
    """
    Parse a swap/fly name into (agency, coupon) tuples.

    Examples
    --------
    'FN 5.5 Price_FN 5 Price'              -> (('FN', '5.5'), ('FN', '5'))
    'G2 5 Price_FN 5 Price'                -> (('G2', '5'), ('FN', '5'))
    'FN 5 Price_FN 4.5 Price_FN 4 Price_fly' -> (('FN','5'),('FN','4.5'),('FN','4'))
    """
    parts = name.replace("_fly", "").split("_")
    result = []
    for part in parts:
        # part is like 'FN 5.5 Price' -> tokens ['FN', '5.5', 'Price']
        tokens = part.strip().split()
        if len(tokens) >= 2:
            result.append((tokens[0], tokens[1]))
    return tuple(result)


def _calc_latest_carry_for_swap_fly(dat: pd.DataFrame, name: str) -> float:
    """Return the latest carry (drop units) for a swap or fly structure."""
    legs = _parse_swap_fly_name(name)
    if len(legs) == 2:
        (ag1, cp1), (ag2, cp2) = legs
        drop1 = f"{ag1} {cp1} Drop"
        drop2 = f"{ag2} {cp2} Drop"
        if drop1 not in dat.columns or drop2 not in dat.columns:
            return np.nan
        return float(dat[drop1].iloc[-1] - dat[drop2].iloc[-1])
    if len(legs) == 3:
        drops = [f"{ag} {cp} Drop" for ag, cp in legs]
        if any(d not in dat.columns for d in drops):
            return np.nan
        return float(dat[drops[1]].iloc[-1] * 2 - dat[drops[0]].iloc[-1] - dat[drops[2]].iloc[-1])
    return np.nan


def run_ols_for_swap_fly(
    dat: pd.DataFrame,
    name: str,
    dat_len: int = 250,
) -> Dict[str, float]:
    """
    Run the notebook-style OLS fair-value regression for one swap or fly.

    Returns
    -------
    dict with mkt_val, mdl_val, resid_std, r2
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels is required for OLS regression.") from exc

    legs = _parse_swap_fly_name(name)
    if len(legs) not in (2, 3):
        return {"mkt_val": np.nan, "mdl_val": np.nan, "resid_std": np.nan, "r2": np.nan}

    # Build Ys and carry
    if len(legs) == 2:
        # Swap: cpn1 - cpn2
        (ag1, cp1), (ag2, cp2) = legs
        price1 = f"{ag1} {cp1} Price"
        price2 = f"{ag2} {cp2} Price"
        drop1 = f"{ag1} {cp1} Drop"
        drop2 = f"{ag2} {cp2} Drop"

        Ys = (dat[price1].tail(dat_len) - dat[price2].tail(dat_len)) * 32
        carry = dat[drop1].tail(dat_len) - dat[drop2].tail(dat_len)
    else:
        # Fly: 2*middle - front - back
        (ag1, cp1), (ag2, cp2), (ag3, cp3) = legs
        price1 = f"{ag1} {cp1} Price"
        price2 = f"{ag2} {cp2} Price"
        price3 = f"{ag3} {cp3} Price"
        drop1 = f"{ag1} {cp1} Drop"
        drop2 = f"{ag2} {cp2} Drop"
        drop3 = f"{ag3} {cp3} Drop"

        Ys = (dat[price2].tail(dat_len) * 2 - dat[price1].tail(dat_len) - dat[price3].tail(dat_len)) * 32
        carry = (dat[drop2].tail(dat_len) * 2 - dat[drop1].tail(dat_len) - dat[drop3].tail(dat_len))

    # Independent variables (use first leg's price as 'price')
    price_col = f"{legs[0][0]} {legs[0][1]} Price"
    required = [price_col, REF_RATE_CURVE["short"], REF_RATE_CURVE["long"], REF_VOL]
    for c in required:
        if c not in dat.columns:
            return {"mkt_val": np.nan, "mdl_val": np.nan, "resid_std": np.nan, "r2": np.nan}

    price = dat[price_col].tail(dat_len)
    slope = (dat[REF_RATE_CURVE["long"]].tail(dat_len) - dat[REF_RATE_CURVE["short"]].tail(dat_len)) * 100
    vol = dat[REF_VOL].tail(dat_len)

    X = pd.DataFrame({"price": price, "slope": slope, "carry": carry, "vol": vol})
    X = sm.add_constant(X)

    # Drop rows with NaN in X or Y
    combined = pd.concat([Ys, X], axis=1).dropna()
    if len(combined) < 20:
        return {"mkt_val": np.nan, "mdl_val": np.nan, "resid_std": np.nan, "r2": np.nan}

    Y_clean = combined.iloc[:, 0]
    X_clean = combined.iloc[:, 1:]

    try:
        model = sm.OLS(Y_clean, X_clean).fit()
    except Exception:
        return {"mkt_val": np.nan, "mdl_val": np.nan, "resid_std": np.nan, "r2": np.nan}

    return {
        "mkt_val": float(Y_clean.iloc[-1]),
        "mdl_val": float(model.fittedvalues.iloc[-1]),
        "resid_std": float(np.std(model.resid)),
        "r2": float(model.rsquared),
        "coef_const": float(model.params.get("const", np.nan)),
        "coef_price": float(model.params.get("price", np.nan)),
        "coef_slope": float(model.params.get("slope", np.nan)),
        "coef_carry": float(model.params.get("carry", np.nan)),
        "coef_vol": float(model.params.get("vol", np.nan)),
    }


def add_ols_metrics_to_summary(
    dat: pd.DataFrame,
    summary_df: pd.DataFrame,
    dat_len: int = 250,
) -> pd.DataFrame:
    """Add OLS fair-value metrics for each swap/fly row."""
    results = []
    for name in summary_df.index:
        metrics = run_ols_for_swap_fly(dat, name, dat_len=dat_len)
        results.append(metrics)

    metrics_df = pd.DataFrame(results, index=summary_df.index)
    metrics_df["resid_z"] = (metrics_df["mkt_val"] - metrics_df["mdl_val"]) / metrics_df["resid_std"]
    return pd.concat([summary_df, metrics_df], axis=1)


def _build_swap_fly_series(dat: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Build the full historical price series (in 32nds) for a swap or fly name."""
    legs = _parse_swap_fly_name(name)
    if len(legs) not in (2, 3):
        return None

    if len(legs) == 2:
        (ag1, cp1), (ag2, cp2) = legs
        price1 = f"{ag1} {cp1} Price"
        price2 = f"{ag2} {cp2} Price"
        if price1 not in dat.columns or price2 not in dat.columns:
            return None
        series = (dat[price1] - dat[price2]) * 32
    else:
        (ag1, cp1), (ag2, cp2), (ag3, cp3) = legs
        price1 = f"{ag1} {cp1} Price"
        price2 = f"{ag2} {cp2} Price"
        price3 = f"{ag3} {cp3} Price"
        if price1 not in dat.columns or price2 not in dat.columns or price3 not in dat.columns:
            return None
        series = (dat[price2] * 2 - dat[price1] - dat[price3]) * 32

    return series.dropna()


def add_range_metrics_to_summary(
    dat: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical range metrics for each swap/fly row:
    hist_min, hist_max, range_pos (= current - min).
    """
    results = []
    for name in summary_df.index:
        series = _build_swap_fly_series(dat, name)
        if series is None or series.empty:
            results.append({"hist_min": np.nan, "hist_max": np.nan, "range_pos": np.nan})
            continue

        current = float(series.iloc[-1])
        hist_min = float(series.min())
        hist_max = float(series.max())
        min_curr = current - hist_min
        curr_max = hist_max - current
        range_ratio = curr_max / min_curr if min_curr != 0 else np.nan
        results.append({
            "hist_min": hist_min,
            "hist_max": hist_max,
            "range_ratio": range_ratio,
        })

    range_df = pd.DataFrame(results, index=summary_df.index)
    return pd.concat([summary_df, range_df], axis=1)


# ---------------------------------------------------------------------------
# Roll (drop) scoring
# ---------------------------------------------------------------------------
def calc_roll_scores(
    dat: pd.DataFrame,
    num_drops: int = 16,
    lookback: int = DEFAULT_ROLL_LOOKBACK,
    score_indices: List[int] = None,
) -> pd.DataFrame:
    """Calculate z-scores for roll (drop) columns."""
    score_indices = score_indices or DEFAULT_SCORE_INDICES
    drop_columns = get_block_columns(dat, "FN", "Drop") + get_block_columns(dat, "G2", "Drop")
    drop_columns = drop_columns[:num_drops]

    rows = []
    for score_idx in score_indices:
        date = dat.index[score_idx]
        for col in drop_columns:
            rows.append(_score_row(col, dat[col], lookback, score_idx, date))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Yield basis scoring
# ---------------------------------------------------------------------------
def calc_yield_basis(
    dat: pd.DataFrame,
    ref_idx: str = REF_BASIS,
    lookback: int = DEFAULT_BASIS_LOOKBACK,
    score_indices: List[int] = None,
) -> pd.DataFrame:
    """Calculate bond yield minus CT10 basis z-scores."""
    score_indices = score_indices or DEFAULT_SCORE_INDICES
    combined = get_block_columns(dat, "FN", "Yield") + get_block_columns(dat, "G2", "Yield")

    rows = []
    for score_idx in score_indices:
        date = dat.index[score_idx]
        for col in combined:
            basis = (dat[col] - dat[ref_idx]) * 100
            rows.append(_score_row(col + "_basis", basis, lookback, score_idx, date))
    return pd.DataFrame(rows)


def build_yield_basis_summary_table(
    dat: pd.DataFrame,
    lookback: int = DEFAULT_BASIS_LOOKBACK,
    score_indices: List[int] = None,
) -> pd.DataFrame:
    """
    Wide-format yield-basis summary table (notebook cell 40 style).
    Columns: t0_vals, t0_scores, t1_vals, t1_scores, ...
    """
    score_indices = score_indices or [-1, -5, -10, -20, -30, -40, -50]
    frames = []

    for i, score_idx in enumerate(score_indices):
        df = calc_yield_basis(dat, lookback=lookback, score_indices=[score_idx])
        df = df.set_index("name")[["vals", "scores"]]
        df.columns = pd.MultiIndex.from_product([[f"t{i}"], df.columns])
        frames.append(df)

    tmp = pd.concat(frames, axis=1)
    tmp.columns = [f"{second}_{first}" for first, second in tmp.columns]
    return tmp


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def calc_perf_summary(
    dat: pd.DataFrame,
    lookback_windows: List[int] = None,
) -> pd.DataFrame:
    """Sum of returns over multiple trailing windows for all perf columns."""
    lookback_windows = lookback_windows or DEFAULT_PERF_WINDOWS
    all_perf = get_block_columns(dat, "FN", "Perf") + get_block_columns(dat, "G2", "Perf")

    combined = pd.DataFrame(index=all_perf)
    for lb in lookback_windows:
        # iloc[lb:] works for negative lb (trailing window)
        combined[f"perf{lb}"] = [dat[col].iloc[lb:].sum() for col in all_perf]
    return combined


def build_perf_summary_table(
    dat: pd.DataFrame,
    lookback_windows: List[int] = None,
) -> pd.DataFrame:
    """
    Build a formatted performance-over-periods table: one row per FN/G2 coupon,
    columns for each trailing window in days.
    """
    lookback_windows = lookback_windows or DEFAULT_PERF_WINDOWS
    df = calc_perf_summary(dat, lookback_windows)
    df = df.rename(columns={f"perf{lb}": f"{abs(lb)}d" for lb in lookback_windows})
    df.index.name = "name"
    df = df.reset_index()
    df["agency"] = df["name"].apply(
        lambda x: "FN" if str(x).startswith("FN") else ("G2" if str(x).startswith("G2") else "Other")
    )
    return df.set_index(["agency", "name"])


def build_rolling_perf_zscore_table(
    dat: pd.DataFrame,
    roll_sum_window: int = ROLL_SUM_WINDOW,
    roll_z_window: int = ROLL_SUM_Z_WINDOW,
) -> pd.DataFrame:
    """
    Rolling-sum performance z-score table.

    For each FN/G2 coupon Perf column:
      - compute rolling sum over roll_sum_window
      - compute z-score of that rolling sum over roll_z_window
    Returns the latest values for each series.
    """
    total_perf = get_block_columns(dat, "FN", "Perf") + get_block_columns(dat, "G2", "Perf")
    rows = []
    for name in total_perf:
        col = dat[name]
        rolling_sum = col.rolling(window=roll_sum_window).sum()
        rolling_mean = rolling_sum.rolling(window=roll_z_window).mean()
        rolling_std = rolling_sum.rolling(window=roll_z_window).std()
        z_score = (rolling_sum - rolling_mean) / rolling_std.replace(0, np.nan)
        rows.append(
            {
                "name": name,
                "latest_perf": col.iloc[-1],
                f"roll_sum_{roll_sum_window}d": rolling_sum.iloc[-1],
                f"z_score_{roll_z_window}d": z_score.iloc[-1],
            }
        )

    df = pd.DataFrame(rows)
    df["agency"] = df["name"].apply(
        lambda x: "FN" if str(x).startswith("FN") else ("G2" if str(x).startswith("G2") else "Other")
    )
    return df.set_index(["agency", "name"])


# ---------------------------------------------------------------------------
# OLS fair-value regression
# ---------------------------------------------------------------------------
def run_ols_regression(
    dat: pd.DataFrame,
    cpn1: str = "G2 5.5",
    cpn2: str = "FN 5.5",
    dat_len: int = 250,
) -> Tuple[object, Dict]:
    """
    OLS of a coupon swap vs price, curve slope, carry and vol.

    Parameters
    ----------
    dat : pd.DataFrame
        Cleaned MBS RV data.
    cpn1, cpn2 : str
        Coupon names WITHOUT the ' Price' suffix.
    dat_len : int
        Number of trailing observations to use.

    Returns
    -------
    model : statsmodels OLS results
    summary : dict with key metrics
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels is required for OLS regression.") from exc

    col1 = f"{cpn1} Price"
    col2 = f"{cpn2} Price"
    drop1 = f"{cpn1} Drop"
    drop2 = f"{cpn2} Drop"

    for c in [col1, col2, drop1, drop2, REF_RATE_CURVE["short"], REF_RATE_CURVE["long"], REF_VOL]:
        if c not in dat.columns:
            raise KeyError(f"Required column missing: {c}")

    Ys = (dat[col1].tail(dat_len) - dat[col2].tail(dat_len)) * 32
    carry = dat[drop1].tail(dat_len) - dat[drop2].tail(dat_len)
    price = dat[col1].tail(dat_len)
    slope = (dat[REF_RATE_CURVE["long"]].tail(dat_len) - dat[REF_RATE_CURVE["short"]].tail(dat_len)) * 100
    vol = dat[REF_VOL].tail(dat_len)

    X = pd.DataFrame({"price": price, "slope": slope, "carry": carry, "vol": vol})
    X = sm.add_constant(X)
    model = sm.OLS(Ys, X).fit()

    fitted = model.fittedvalues
    resid = model.resid
    resid_std = float(np.std(resid))
    latest_actual = float(Ys.iloc[-1])
    latest_fitted = float(fitted.iloc[-1])
    latest_residual = latest_actual - latest_fitted

    summary = {
        "cpn1": cpn1,
        "cpn2": cpn2,
        "latest_actual": latest_actual,
        "latest_fitted": latest_fitted,
        "latest_residual": latest_residual,
        "residual_std": resid_std,
        "residual_z": latest_residual / resid_std if resid_std > 0 else np.nan,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "params": model.params.to_dict(),
    }
    return model, summary


# ---------------------------------------------------------------------------
# Current-coupon quadratic curve fit
# ---------------------------------------------------------------------------
def fit_current_coupon_curve(
    dat: pd.DataFrame,
    agency: str = "FN",
    par_price: float = 100.0,
    degree: int = 2,
) -> Dict:
    """
    Fit price as a polynomial of (coupon - current coupon) for the latest row.

    Returns
    -------
    dict with fitted coefficients, predicted prices, residuals and current coupon.
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise ImportError("scikit-learn is required for the current-coupon curve fit.") from exc

    price_columns = get_block_columns(dat, agency, "Price")
    if len(price_columns) != len(COUPONS):
        raise ValueError(f"Expected {len(COUPONS)} price columns for {agency}, found {len(price_columns)}")

    # Current coupon by interpolating coupon list vs prices to par_price
    cc_list = []
    for _, row in dat.iterrows():
        prices = [row[col] for col in price_columns]
        cc = float(np.interp(par_price, prices, COUPONS))
        cc_list.append(cc)
    dat = dat.copy()
    dat["CC"] = cc_list

    # Stack all rows: x = (cpn - CC)*100, y = price
    Xs, Ys = [], []
    row_data = []
    for idx, row in dat.iterrows():
        cc = row["CC"]
        row_xs, row_ys = [], []
        for cpn, px_col in zip(COUPONS, price_columns):
            x = (cpn - cc) * 100
            y = row[px_col]
            Xs.append(x)
            Ys.append(y)
            row_xs.append(x)
            row_ys.append(y)
        row_data.append((row_xs, row_ys, idx, cc))

    X = np.array(Xs).reshape(-1, 1)
    Y = np.array(Ys)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_poly, Y)

    # Predictions for most recent row
    recent_xs, recent_ys, recent_idx, recent_cc = row_data[-1]
    recent_x_array = np.array(recent_xs).reshape(-1, 1)
    recent_x_poly = poly.transform(recent_x_array)
    recent_pred = lr.predict(recent_x_poly)
    recent_residuals = np.array(recent_ys) - recent_pred

    mae = float(np.mean(np.abs(recent_residuals)))
    rmse = float(np.sqrt(np.mean(recent_residuals ** 2)))

    return {
        "agency": agency,
        "latest_date": recent_idx,
        "current_coupon": float(recent_cc),
        "intercept": float(lr.intercept_),
        "coefs": lr.coef_.tolist(),
        "mae": mae,
        "rmse": rmse,
        "max_abs_resid": float(np.max(np.abs(recent_residuals))),
        "latest_residuals": pd.DataFrame({
            "coupon": COUPONS,
            "x": recent_xs,
            "actual": recent_ys,
            "predicted": recent_pred,
            "residual": recent_residuals,
        }),
    }


# ---------------------------------------------------------------------------
# Optional plotting helpers
# ---------------------------------------------------------------------------
def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_ols_fitted(
    dat: pd.DataFrame,
    model_summary: Dict,
    cpn1: str,
    cpn2: str,
    dat_len: int = 250,
) -> Optional[plt.Figure]:
    """Create an actual vs fitted OLS plot and return the figure."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping OLS plot.")
        return None

    col1 = f"{cpn1} Price"
    col2 = f"{cpn2} Price"
    drop1 = f"{cpn1} Drop"
    drop2 = f"{cpn2} Drop"

    Ys = (dat[col1].tail(dat_len) - dat[col2].tail(dat_len)) * 32
    carry = dat[drop1].tail(dat_len) - dat[drop2].tail(dat_len)
    price = dat[col1].tail(dat_len)
    slope = (dat[REF_RATE_CURVE["long"]].tail(dat_len) - dat[REF_RATE_CURVE["short"]].tail(dat_len)) * 100
    vol = dat[REF_VOL].tail(dat_len)

    X = pd.DataFrame({"price": price, "slope": slope, "carry": carry, "vol": vol})
    import statsmodels.api as sm
    X = sm.add_constant(X)
    fitted = sm.OLS(Ys, X).fit().fittedvalues

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(Ys.index, Ys, label="Actual", color="red", alpha=0.7)
    ax.plot(fitted.index, fitted, label="Fitted", color="blue", alpha=0.7)
    ax.set_title(f"OLS Fair Value: {cpn1} vs {cpn2}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Swap (32nds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDF report generation
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
    gradient_vranges: Optional[Dict[str, Tuple[float, float]]] = None,
    diverging_columns: Optional[List[str]] = None,
    compact: bool = False,
    include_index: bool = True,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib figure (one page)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    # Reset index so it appears as a column (unless caller wants a clean data-only table)
    if include_index:
        df = df.reset_index()
    # Round floats for display
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    # Leave top margin for title + subtitle; fill the rest with the table
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.83])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.965, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.88, subtitle, fontsize=8, ha="left", transform=fig.transFigure, color="#555555")

    table_data = [display_df.columns.tolist()] + display_df.values.tolist()
    # First column for row labels (names), remaining columns share rest.
    # In compact mode with many columns, shrink the name column so data columns get more space.
    if n_cols > 1:
        if compact and n_cols >= 10:
            name_width = 0.20
        elif not include_index:
            name_width = 0.0
        else:
            name_width = 0.5
        if include_index:
            col_widths = [name_width] + [(1.0 - name_width) / (n_cols - 1)] * (n_cols - 1)
        else:
            col_widths = [1.0 / n_cols] * n_cols
    else:
        col_widths = [1.0]
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )

    # Size rows: compact fixed height, or fill the page vertically
    total_cells = n_rows + 1
    if compact:
        cell_h = row_height
        auto_font = fontsize
    else:
        cell_h = max(row_height, 0.88 / total_cells)
        auto_font = min(fontsize, int(cell_h * 280))
    table.auto_set_font_size(False)
    table.set_fontsize(auto_font)

    # Determine gradient columns by name in the *original* df (before reset_index)
    gradient_cols = []
    if gradient_columns:
        for gc in gradient_columns:
            if gc in df.columns:
                gradient_cols.append(gc)

    # Header style
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight="bold", color="white")
        cell.set_height(cell_h)

    # Pre-compute color functions for each gradient column.
    # Use caller-supplied global ranges if provided so all pages share the same scale.
    gradient_vranges = gradient_vranges or {}
    diverging_columns = diverging_columns or []
    color_funcs = {}
    for col_name in gradient_cols:
        if col_name in df.columns:
            vr = gradient_vranges.get(col_name)
            vmin, vmax = (vr[0], vr[1]) if vr else (None, None)
            color_funcs[col_name] = _column_color_func(
                df[col_name],
                vmin=vmin,
                vmax=vmax,
                force_diverging=(col_name in diverging_columns),
            )

    # Row styles + optional heatmap on selected columns
    for i in range(1, n_rows + 1):
        for j, col_name in enumerate(display_df.columns):
            cell = table[(i, j)]
            cell.set_height(cell_h)

            # Zebra stripe
            base_color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
            cell.set_facecolor(base_color)

            # Heatmap background on selected columns
            if col_name in color_funcs:
                val = df[col_name].iloc[i - 1]
                if pd.notna(val):
                    color = color_funcs[col_name](val)
                    cell.set_facecolor(color)

    return fig


def _score_to_color(
    score: float,
    vmin: float = -2.5,
    vmax: float = 2.5,
) -> str:
    """Map a z-score / residual to a diverging blue-white-red color."""
    import matplotlib.colors as mcolors

    norm = (score - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))

    # Use matplotlib's coolwarm: 0 = blue, 0.5 = white, 1 = red
    cmap = plt.get_cmap("coolwarm")
    return mcolors.to_hex(cmap(norm)[:3])


def _sequential_color(
    value: float,
    vmin: float,
    vmax: float,
) -> str:
    """Map a non-negative distance metric to a white-to-red sequential color."""
    import matplotlib.colors as mcolors

    if vmax == vmin:
        return "#ffffff"

    norm = (value - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))

    cmap = mcolors.LinearSegmentedColormap.from_list("sequential", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26"])
    return mcolors.to_hex(cmap(norm)[:3])


def _column_color_func(
    series: pd.Series,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    force_diverging: bool = False,
):
    """Pick the right coloring function and range for a numeric column."""
    s = series.dropna()
    if s.empty:
        return lambda x: "#ffffff"

    if vmin is None:
        vmin = float(s.min())
    if vmax is None:
        vmax = float(s.max())

    # For z-score-like columns (or when forced), always use diverging blue-white-red
    # centered at 0 so negative and positive values are colored consistently.
    if force_diverging or (vmin < 0 < vmax):
        limit = max(abs(vmin), abs(vmax), 0.01)
        return lambda x: _score_to_color(float(x), -limit, limit)

    # Otherwise use sequential white-to-red
    return lambda x: _sequential_color(float(x), vmin, vmax)


def _add_summary_page(
    pdf: PdfPages,
    dat: pd.DataFrame,
    ols_summary: Dict,
    cc_fit: Dict,
) -> None:
    """Add a nicely formatted summary page."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")

    title = "MBS RV Analysis Report"
    subtitle = f"Latest date: {dat.index[-1].date()}  |  Observations: {len(dat)}"

    ax.text(0.5, 0.92, title, fontsize=24, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.86, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    # OLS box
    ols_text = (
        "OLS Fair-Value Regression\n"
        f"Pair:          {ols_summary['cpn1']} vs {ols_summary['cpn2']}\n"
        f"Latest actual: {ols_summary['latest_actual']:.2f} 32nds\n"
        f"Latest fitted: {ols_summary['latest_fitted']:.2f} 32nds\n"
        f"Residual:      {ols_summary['latest_residual']:.2f} 32nds\n"
        f"Residual std:  {ols_summary['residual_std']:.4f}\n"
        f"Residual z:    {ols_summary['residual_z']:.2f}\n"
        f"R-squared:     {ols_summary['r_squared']:.4f}"
    )
    ax.text(0.15, 0.70, ols_text, fontsize=11, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2ff", edgecolor="#40466e"))

    # Current coupon box
    cc_text = (
        "Current-Coupon Curve Fit (FN)\n"
        f"Latest date:    {cc_fit['latest_date'].date()}\n"
        f"Current coupon: {cc_fit['current_coupon']:.3f}%\n"
        f"MAE:            {cc_fit['mae']:.4f}\n"
        f"RMSE:           {cc_fit['rmse']:.4f}\n"
        f"Max abs resid:  {cc_fit['max_abs_resid']:.4f}"
    )
    ax.text(0.55, 0.70, cc_text, fontsize=11, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8ee", edgecolor="#8b5a00"))

    # Output list
    ax.text(0.5, 0.35, "Report contents:", fontsize=12, weight="bold",
            ha="center", transform=ax.transAxes)
    contents = (
        "1. OLS regression parameters\n"
        "2. Coupon swap / fly / G2-FN scores\n"
        "3. Roll (drop) scores\n"
        "4. Yield basis scores\n"
        "5. Performance summary\n"
        "6. Current-coupon curve residuals\n"
        "7. OLS fitted vs actual plot\n"
        "8. Latest z-score snapshot charts"
    )
    ax.text(0.5, 0.22, contents, fontsize=10, family="monospace",
            ha="center", verticalalignment="top", transform=ax.transAxes)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_reference_dates_table(dat: pd.DataFrame) -> pd.DataFrame:
    """Build a table of reference dates for all lookback windows."""
    rows = []
    basis_score_indices = [-1, -5, -10, -20, -30, -40, -50]

    for i, idx in enumerate(DEFAULT_SCORE_INDICES):
        d = dat.index[idx]
        rows.append(
            {
                "category": "Cpn/Fly/G2FN score",
                "label": f"t{i}",
                "date_ymd": d.strftime("%Y%m%d"),
                "calendar_date": str(d.date()),
            }
        )

    for i, idx in enumerate(basis_score_indices):
        d = dat.index[idx]
        rows.append(
            {
                "category": "Yield basis score",
                "label": f"t{i}",
                "date_ymd": d.strftime("%Y%m%d"),
                "calendar_date": str(d.date()),
            }
        )

    for lb in DEFAULT_PERF_WINDOWS:
        d = dat.index[lb]
        rows.append(
            {
                "category": "Perf window start",
                "label": f"{lb}d",
                "date_ymd": d.strftime("%Y%m%d"),
                "calendar_date": str(d.date()),
            }
        )

    return pd.DataFrame(rows)


def _add_contents_page(
    pdf: PdfPages,
    dat: pd.DataFrame,
    num_cpn_fly_structures: int = 0,
) -> None:
    """Add a simple contents page as the first page of the report."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")

    title = "MBS RV Analysis Report"
    subtitle = f"Latest date: {dat.index[-1].date()}  |  Observations: {len(dat)}"

    ax.text(0.5, 0.90, title, fontsize=24, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.84, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    contents = (
        "Report contents\n\n"
        "1. Coupon swap / fly / G2-FN score tables\n"
        "2. Yield basis score tables\n"
        "3. Performance over periods\n"
        "4. Rolling performance z-scores\n"
        "5. Current-coupon basis OLS summary\n"
        "6. Current-coupon basis OLS parameters\n"
        "7. Current-coupon basis fitted vs actual\n"
        "8. Rolling current-coupon basis (120-day window)\n"
        "9. Conventional vs Ginnie current-coupon OAS spread\n"
        "10. Per-coupon yield basis OLS summary\n"
        "11. Per-coupon yield basis OLS parameters\n"
        "12. Per-coupon yield basis fitted vs actual\n"
        f"13. Appendix: historical charts for {num_cpn_fly_structures} cpn/fly structures"
    )
    ax.text(
        0.5, 0.52, contents,
        fontsize=12, family="monospace",
        ha="center", va="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#40466e"),
    )

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    gradient_columns: Optional[List[str]] = None,
    gradient_vranges: Optional[Dict[str, Tuple[float, float]]] = None,
    diverging_columns: Optional[List[str]] = None,
    fontsize: int = 8,
    row_height: float = 0.35,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    compact: bool = False,
    include_index: bool = True,
) -> None:
    """Add a DataFrame as a single PDF page."""
    fig = _df_to_figure(
        df,
        title,
        subtitle=subtitle,
        gradient_columns=gradient_columns,
        gradient_vranges=gradient_vranges,
        diverging_columns=diverging_columns,
        fontsize=fontsize,
        row_height=row_height,
        pagesize=pagesize,
        compact=compact,
        include_index=include_index,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_zscore_bar_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    top_n: int = 30,
) -> None:
    """Add a horizontal bar chart of the latest z-scores."""
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    latest = latest.sort_values("scores", key=abs, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    colors = ["#d62728" if s > 0 else "#1f77b4" for s in latest["scores"]]
    ax.barh(range(len(latest)), latest["scores"], color=colors, edgecolor="black")
    ax.set_yticks(range(len(latest)))
    ax.set_yticklabels(latest["name"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score", fontsize=10)
    ax.set_title(f"{title}\n(latest: {latest_date.date()})", fontsize=12, weight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_hist_swap_fly_series(
    dat: pd.DataFrame,
    mode: str = HIST_PLOT_MODE,
    pair: Tuple[str, str] = HIST_CPN_PAIR,
    triple: Tuple[str, str, str] = HIST_FLY_TRIPLE,
) -> Tuple[pd.Series, str]:
    """Build the historical swap/fly series (in 32nds) and a descriptive label."""
    if mode == "cpn":
        cpn1, cpn2 = pair
        col1 = f"{cpn1} Price"
        col2 = f"{cpn2} Price"
        missing = [c for c in (col1, col2) if c not in dat.columns]
        if missing:
            raise KeyError(f"Missing price columns for historical coupon plot: {missing}")
        series = (dat[col1] - dat[col2]) * 32
        label = f"{cpn1} vs {cpn2}"
    elif mode == "fly":
        front, middle, back = triple
        cols = [f"{c} Price" for c in (front, middle, back)]
        missing = [c for c in cols if c not in dat.columns]
        if missing:
            raise KeyError(f"Missing price columns for historical fly plot: {missing}")
        series = (dat[cols[1]] * 2 - dat[cols[0]] - dat[cols[2]]) * 32
        label = f"{front}/{middle}/{back} fly"
    else:
        raise ValueError(f"HIST_PLOT_MODE must be 'cpn' or 'fly', got {mode!r}")
    return series.dropna(), label


def _build_hist_carry_series(
    dat: pd.DataFrame,
    mode: str = HIST_PLOT_MODE,
    pair: Tuple[str, str] = HIST_CPN_PAIR,
    triple: Tuple[str, str, str] = HIST_FLY_TRIPLE,
) -> Optional[pd.Series]:
    """Build the historical carry series for the configured swap or fly."""
    if mode == "cpn":
        cpn1, cpn2 = pair
        col1 = f"{cpn1} Drop"
        col2 = f"{cpn2} Drop"
        if col1 not in dat.columns or col2 not in dat.columns:
            return None
        carry = dat[col1] - dat[col2]
    elif mode == "fly":
        front, middle, back = triple
        cols = [f"{c} Drop" for c in (front, middle, back)]
        if any(c not in dat.columns for c in cols):
            return None
        carry = dat[cols[1]] * 2 - dat[cols[0]] - dat[cols[2]]
    else:
        return None
    return carry.dropna()


def plot_historical_cpn_fly(
    dat: pd.DataFrame,
    mode: str = HIST_PLOT_MODE,
    pair: Tuple[str, str] = HIST_CPN_PAIR,
    triple: Tuple[str, str, str] = HIST_FLY_TRIPLE,
    ref_cols: List[str] = HIST_REF_COLS,
) -> plt.Figure:
    """
    Plot a historical coupon swap or fly (in 32nds) vs one or more references.

    Returns a figure with 2 rows x N columns:
        - top row: time series of the swap/fly with each reference on a twin axis
        - bottom row: scatter of swap/fly vs each reference, latest point highlighted
    """
    if isinstance(ref_cols, str):
        ref_cols = [ref_cols]

    missing = [c for c in ref_cols if c not in dat.columns]
    if missing:
        raise KeyError(f"Reference column(s) not found: {missing}")

    series_full, label = _build_hist_swap_fly_series(dat, mode=mode, pair=pair, triple=triple)
    latest_val = float(series_full.iloc[-1])
    hist_min = float(series_full.min())
    hist_max = float(series_full.max())

    carry_full = _build_hist_carry_series(dat, mode=mode, pair=pair, triple=triple)
    carry_drops_text = ""
    if carry_full is not None and not carry_full.empty:
        latest_carry = float(carry_full.iloc[-1])
        if mode == "cpn":
            cpn1, cpn2 = pair
            d1 = float(dat[f"{cpn1} Drop"].iloc[-1])
            d2 = float(dat[f"{cpn2} Drop"].iloc[-1])
            carry_drops_text = (
                f"Carry: {latest_carry:.2f}  |  "
                f"Drops: {cpn1}={d1:.2f}, {cpn2}={d2:.2f}"
            )
        elif mode == "fly":
            front, middle, back = triple
            d_front = float(dat[f"{front} Drop"].iloc[-1])
            d_mid = float(dat[f"{middle} Drop"].iloc[-1])
            d_back = float(dat[f"{back} Drop"].iloc[-1])
            carry_drops_text = (
                f"Carry: {latest_carry:.2f}  |  "
                f"Drops: {front}={d_front:.2f}, "
                f"{middle}={d_mid:.2f}, {back}={d_back:.2f}"
            )

    stats_text = (
        f"Latest: {latest_val:.2f}\n"
        f"Max: {hist_max:.2f}\n"
        f"Min: {hist_min:.2f}"
    )

    n_refs = len(ref_cols)
    fig, axes = plt.subplots(2, n_refs, figsize=(5.5 * n_refs, 8.0), sharey="all")

    if n_refs == 1:
        axes = axes.reshape(-1, 1)

    for j, ref_col in enumerate(ref_cols):
        ref = dat[ref_col]
        df = pd.concat([series_full, ref], axis=1).dropna()
        if df.empty:
            continue
        series = df.iloc[:, 0]
        ref = df.iloc[:, 1]

        # --- Top: time series ---
        ax1 = axes[0, j]
        color_swap = "navy"
        ax1.plot(series.index, series, color=color_swap, linewidth=1.2, label=label)
        if j == 0:
            ax1.set_ylabel(f"{label} (32nds)", color=color_swap)
        ax1.tick_params(axis="y", labelcolor=color_swap)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(ref.index, ref, color="red", linewidth=1.0, label=ref_col)
        ax2.set_ylabel(ref_col, color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        title_text = f"{label} vs {ref_col}"
        if carry_drops_text:
            title_text += f"\n{carry_drops_text}"
        ax1.set_title(title_text, fontsize=10, weight="bold", ha="center")
        for tick in ax1.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2,
            loc="lower left", fontsize=7, framealpha=0.9,
        )

        ax1.text(
            0.98, 0.95, stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

        # --- Bottom: scatter ---
        ax = axes[1, j]
        ax.scatter(
            ref.iloc[:-1], series.iloc[:-1],
            c="gray", alpha=0.3, s=25, label="Historical data",
        )
        ax.scatter(
            ref.iloc[-1], series.iloc[-1],
            c="red", s=120, marker="*",
            edgecolors="black", linewidth=1.5,
            zorder=10, label="Latest point",
        )
        ax.set_xlabel(ref_col, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"{label} (32nds)", fontsize=9)
        ax.set_title(f"{label} vs {ref_col}", fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"Historical {label}", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _group_cpn_fly_rows(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split the wide cpn/fly summary table into logical groups for compact display.
    Preserves the original row order within each group.
    """

    def _category(name: str) -> str:
        name = str(name)
        is_fly = name.endswith("_fly")
        has_fn = "FN " in name
        has_g2 = "G2 " in name

        if is_fly:
            if name.startswith("FN"):
                return "FN Flies"
            if name.startswith("G2"):
                return "G2 Flies"
            return "Flies"

        if has_fn and has_g2:
            return "G2 / FN Swaps"
        if has_fn:
            return "FN Coupon Swaps"
        if has_g2:
            return "G2 Coupon Swaps"
        return "Other"

    df = df.copy()
    df["_category"] = df.index.map(_category)

    order = ["FN Coupon Swaps", "G2 Coupon Swaps", "G2 / FN Swaps", "FN Flies", "G2 Flies"]
    groups: Dict[str, pd.DataFrame] = {}
    for cat in order:
        sub = df[df["_category"] == cat].drop(columns="_category")
        if not sub.empty:
            groups[cat] = sub
    return groups


def _group_yield_basis_rows(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split yield-basis rows into FN and G2 groups."""
    fn_rows = df[df.index.str.startswith("FN")]
    g2_rows = df[df.index.str.startswith("G2")]
    groups: Dict[str, pd.DataFrame] = {}
    if not fn_rows.empty:
        groups["FN Yield Basis vs CT10"] = fn_rows
    if not g2_rows.empty:
        groups["G2 Yield Basis vs CT10"] = g2_rows
    return groups


def generate_pdf_report(
    dat: pd.DataFrame,
    df_basis: Optional[pd.DataFrame] = None,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate a single nicely formatted PDF report.

    Parameters
    ----------
    dat : pd.DataFrame
        Main MBS RV data loaded from the Excel file.
    df_basis : pd.DataFrame, optional
        Pre-built current-coupon basis DataFrame. If not provided it will be
        built inside this function (requires Bloomberg access).
    pdf_path : Path, optional
        Output PDF path.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"MBS_RV_Report_{report_date}.pdf"
        # If the dated file is locked (e.g., open in a PDF reader), use a timestamped name
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"MBS_RV_Report_{report_date}_{ts}.pdf"

    # Build the notebook-style wide cpn/fly summary table and add OLS + range metrics
    cpn_fly_summary = build_cpn_fly_summary_table(dat)
    cpn_fly_summary = add_ols_metrics_to_summary(dat, cpn_fly_summary)
    cpn_fly_summary = add_range_metrics_to_summary(dat, cpn_fly_summary)

    # Reorder columns: time-slice scores first, then model stats, then mkt + historical range
    time_cols = [f"{prefix}_t{i}" for i in range(len(DEFAULT_SCORE_INDICES)) for prefix in ("vals", "scores")]
    model_cols = ["mdl_val", "resid_std", "r2", "resid_z"]
    coef_cols = ["coef_price", "coef_slope", "coef_carry", "coef_vol"]
    market_cols = ["mkt_val", "carry", "hist_min", "hist_max", "range_ratio"]
    cpn_fly_summary = cpn_fly_summary[time_cols + model_cols + market_cols + coef_cols]

    score_cols = ["scores_t0", "resid_z", "range_ratio"]
    diverging_cols = ["scores_t0", "resid_z"]

    # Global value ranges for heatmap colors so every page uses the same scale
    gradient_vranges = {}
    for col in score_cols:
        if col in cpn_fly_summary.columns:
            vmin = float(cpn_fly_summary[col].min())
            vmax = float(cpn_fly_summary[col].max())
            if col in diverging_cols:
                # Symmetric diverging scale centered at 0
                limit = max(abs(vmin), abs(vmax), 0.01)
                gradient_vranges[col] = (-limit, limit)
            else:
                gradient_vranges[col] = (vmin, vmax)

    # Group rows by category so each page is tight and readable
    groups = _group_cpn_fly_rows(cpn_fly_summary)

    # Date mapping for t0...t6 (cpn/fly uses DEFAULT_SCORE_INDICES)
    date_labels = [f"t{i} = {dat.index[idx].date()}" for i, idx in enumerate(DEFAULT_SCORE_INDICES)]
    subtitle = "  |  ".join(date_labels)

    # Yield basis summary (notebook uses its own score indices)
    basis_score_indices = [-1, -5, -10, -20, -30, -40, -50]
    basis_summary = build_yield_basis_summary_table(dat, score_indices=basis_score_indices)
    basis_groups = _group_yield_basis_rows(basis_summary)
    basis_date_labels = [f"t{i} = {dat.index[idx].date()}" for i, idx in enumerate(basis_score_indices)]
    basis_subtitle = "  |  ".join(basis_date_labels)
    basis_score_cols = ["scores_t0"]

    with PdfPages(pdf_path) as pdf:
        # Contents page
        _add_contents_page(pdf, dat, num_cpn_fly_structures=len(cpn_fly_summary))

        # Cpn/fly pages
        for group_name, group_df in groups.items():
            _add_table_page(
                pdf,
                group_df,
                f"{group_name}  (latest: {dat.index[-1].date()})",
                subtitle=subtitle,
                gradient_columns=score_cols,
                gradient_vranges=gradient_vranges,
                diverging_columns=diverging_cols,
                fontsize=5.5,
                row_height=0.05,
                pagesize=(17.0, 8.5),
                compact=True,
            )

        # Yield basis pages
        for group_name, group_df in basis_groups.items():
            _add_table_page(
                pdf,
                group_df,
                f"{group_name}  (latest: {dat.index[-1].date()})",
                subtitle=basis_subtitle,
                gradient_columns=basis_score_cols,
                fontsize=6,
                row_height=0.06,
                pagesize=(14.0, 8.5),
                compact=True,
            )

        # Performance over periods
        perf_summary = build_perf_summary_table(dat)
        perf_cols = [f"{abs(lb)}d" for lb in DEFAULT_PERF_WINDOWS]
        _add_table_page(
            pdf,
            perf_summary,
            "Performance Over Periods",
            subtitle=f"Cumulative performance over trailing windows  |  Latest: {dat.index[-1].date()}",
            gradient_columns=perf_cols,
            fontsize=8,
            row_height=0.06,
            pagesize=(12.0, 8.5),
            compact=True,
        )

        # Rolling performance z-scores
        rolling_perf_z = build_rolling_perf_zscore_table(dat)
        rolling_perf_cols = ["latest_perf", f"roll_sum_{ROLL_SUM_WINDOW}d", f"z_score_{ROLL_SUM_Z_WINDOW}d"]
        _add_table_page(
            pdf,
            rolling_perf_z,
            f"Rolling Performance Z-Scores ({ROLL_SUM_WINDOW}d sum / {ROLL_SUM_Z_WINDOW}d z-score)",
            subtitle=f"Latest: {dat.index[-1].date()}",
            gradient_columns=rolling_perf_cols,
            fontsize=8,
            row_height=0.06,
            pagesize=(12.0, 8.5),
            compact=True,
        )

        # Current-coupon basis OLS (Conventional + Ginnie)
        try:
            if df_basis is None:
                df_basis = build_current_coupon_basis_df(FNAME)
            cc_summary = build_current_coupon_summary(df_basis)
            cc_lag_text = " (lagged 1d)" if CC_LAG_PREDICTORS else ""
            cc_subtitle = (
                f"Latest: {df_basis.index[-1].date()}  |  "
                f"Obs: {len(df_basis)}  |  "
                f"Predictors: 10y, 2s10s, 1y10y_vol{cc_lag_text}"
            )
            _add_table_page(
                pdf,
                cc_summary,
                "Current-Coupon Basis OLS Summary",
                subtitle=cc_subtitle,
                gradient_columns=["diff", "resid_z"],
                fontsize=9,
                row_height=0.08,
                pagesize=(10.0, 4.0),
                compact=True,
            )

            # Regression parameters table
            cc_params = build_current_coupon_params_table(df_basis)
            _add_table_page(
                pdf,
                cc_params,
                "Current-Coupon Basis OLS Parameters",
                subtitle=cc_subtitle,
                gradient_columns=["coef", "p_value"],
                fontsize=8,
                row_height=0.06,
                pagesize=(10.0, 5.5),
                compact=True,
                include_index=False,
            )

            fig = plot_current_coupon_fitted(df_basis)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Rolling basis OLS market-vs-model plot
            cc_rolling = run_rolling_current_coupon_basis(df_basis, window=CC_ROLLING_WINDOW)
            fig = plot_rolling_current_coupon_basis(cc_rolling, window=CC_ROLLING_WINDOW)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Conventional vs Ginnie CC OAS spread with 90-day z-score
            fig = plot_conv_ginnie_cc_spread(df_basis, window=90)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"Current-coupon basis section skipped: {exc}")

        # FN/G2 TSY OAS current level vs 1y history z-score
        try:
            oas_df = load_tsy_oas_data(FNAME)
            oas_summary = build_tsy_oas_summary(oas_df, lookback=TSY_OAS_LOOKBACK)
            if not oas_summary.empty:
                oas_subtitle = (
                    f"Latest: {oas_df.index[-1].date()}  |  "
                    f"1y lookback: {TSY_OAS_LOOKBACK} trading days  |  "
                    f"z-score = (current - 1y mean) / 1y std"
                )
                _add_table_page(
                    pdf,
                    oas_summary.reset_index(),
                    "FN / G2 TSY OAS vs 1-Year History Z-Score",
                    subtitle=oas_subtitle,
                    gradient_columns=["z_score"],
                    fontsize=8,
                    row_height=0.05,
                    pagesize=(8.5, 5.5),
                    compact=True,
                    include_index=False,
                )
        except Exception as exc:
            print(f"TSY OAS z-score section skipped: {exc}")

        # Per-coupon yield basis OLS (same predictors as CC basis)
        try:
            predictors = load_ust_yield_vol_data(FNAME)
            cb_basis = build_coupon_basis_df(dat, predictors)
            if not cb_basis.empty:
                cb_summary = build_coupon_basis_summary(cb_basis)
                cb_params = build_coupon_basis_params_table(cb_basis)
                cb_lag_text = " (lagged 1d)" if COUPON_BASIS_LAG_PREDICTORS else ""
                cb_subtitle = (
                    f"Latest: {cb_basis.index[-1].date()}  |  "
                    f"Obs: {len(cb_basis)}  |  "
                    f"Predictors: 10y, 2s10s, 1y10y_vol{cb_lag_text}"
                )
                _add_table_page(
                    pdf,
                    cb_summary,
                    "Per-Coupon Yield Basis OLS Summary",
                    subtitle=cb_subtitle,
                    gradient_columns=["diff", "resid_z"],
                    fontsize=8,
                    row_height=0.06,
                    pagesize=(11.0, 6.0),
                    compact=True,
                )
                cb_coef_matrix = build_coupon_basis_coef_matrix(cb_basis)
                if not cb_coef_matrix.empty:
                    for feat in COUPON_BASIS_PREDICTORS:
                        feat_cols = [c for c in cb_coef_matrix.columns if c.endswith(f" {feat}")]
                        if not feat_cols:
                            continue
                        _add_table_page(
                            pdf,
                            cb_coef_matrix[feat_cols],
                            f"Per-Coupon Yield Basis Coefficients — {feat}",
                            subtitle=cb_subtitle,
                            gradient_columns=feat_cols,
                            fontsize=10,
                            row_height=0.09,
                            pagesize=(8.0, 4.5),
                            compact=True,
                        )
                for fig in plot_coupon_basis_fitted(cb_basis):
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            print(f"Per-coupon yield basis section skipped: {exc}")

        # Appendix: historical coupon/fly charts for every cpn/fly structure
        try:
            for name in cpn_fly_summary.index:
                legs = _parse_swap_fly_name(name)
                if len(legs) == 2:
                    mode = "cpn"
                    pair = (f"{legs[0][0]} {legs[0][1]}", f"{legs[1][0]} {legs[1][1]}")
                    triple = HIST_FLY_TRIPLE
                elif len(legs) == 3:
                    mode = "fly"
                    pair = HIST_CPN_PAIR
                    triple = (
                        f"{legs[0][0]} {legs[0][1]}",
                        f"{legs[1][0]} {legs[1][1]}",
                        f"{legs[2][0]} {legs[2][1]}",
                    )
                else:
                    continue

                fig = plot_historical_cpn_fly(
                    dat,
                    mode=mode,
                    pair=pair,
                    triple=triple,
                    ref_cols=HIST_REF_COLS,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            print(f"Historical cpn/fly appendix skipped: {exc}")

    return pdf_path


# ---------------------------------------------------------------------------
# Optional CSV output
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=(isinstance(df.index, pd.RangeIndex) is False))
    return path


def print_report(
    dat: pd.DataFrame,
    ols_summary: Dict,
    cc_fit: Dict,
) -> None:
    """Print a concise console summary."""
    print("\n" + "=" * 70)
    print("MBS RV Analysis Report")
    print("=" * 70)
    print(f"Latest date : {dat.index[-1].date()}")
    print(f"Observations: {len(dat)}")
    print()
    print("OLS Fair-Value Regression")
    print(f"  Pair          : {ols_summary['cpn1']} vs {ols_summary['cpn2']}")
    print(f"  Latest actual : {ols_summary['latest_actual']: .2f} 32nds")
    print(f"  Latest fitted : {ols_summary['latest_fitted']: .2f} 32nds")
    print(f"  Residual      : {ols_summary['latest_residual']: .2f} 32nds")
    print(f"  Residual std  : {ols_summary['residual_std']:.4f}")
    print(f"  Residual z    : {ols_summary['residual_z']: .2f}")
    print(f"  R-squared     : {ols_summary['r_squared']:.4f}")
    print()
    print("Current-Coupon Curve Fit (FN)")
    print(f"  Latest date   : {cc_fit['latest_date'].date()}")
    print(f"  Current coupon: {cc_fit['current_coupon']:.3f}%")
    print(f"  MAE           : {cc_fit['mae']:.4f}")
    print(f"  RMSE          : {cc_fit['rmse']:.4f}")
    print()
    print("=" * 70)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and generate the cpn/fly summary PDF
    dat = load_mbs_rv(FNAME)

    # Build current-coupon basis dataset once and reuse it
    try:
        df_basis = build_current_coupon_basis_df(FNAME)
        cc_summary = build_current_coupon_summary(df_basis)
        cc_params = build_current_coupon_params_table(df_basis)
        cc_rolling = run_rolling_current_coupon_basis(df_basis, window=CC_ROLLING_WINDOW)
    except Exception as exc:
        df_basis = None
        cc_summary = None
        cc_params = None
        cc_rolling = None
        print(f"Current-coupon basis section skipped: {exc}")

    pdf_path = generate_pdf_report(dat, df_basis=df_basis)

    print("\n" + "=" * 70)
    print("MBS RV Analysis Report")
    print("=" * 70)
    print(f"Latest date : {dat.index[-1].date()}")
    print(f"Observations: {len(dat)}")
    if cc_summary is not None:
        print("\nCurrent-Coupon Basis OLS Summary")
        print(cc_summary.to_string())
        print("\nCurrent-Coupon Basis OLS Parameters")
        print(cc_params.to_string(index=False))
    if cc_rolling is not None:
        print(f"\nLatest Rolling Current-Coupon Basis ({CC_ROLLING_WINDOW}d window)")
        print(cc_rolling.tail(2).to_string())
    if df_basis is not None:
        spread = df_basis["FNCC"] - df_basis["G2CC"]
        z90 = (spread - spread.rolling(90).mean()) / spread.rolling(90).std()
        print("\nConv vs Ginnie CC OAS Spread")
        print(f"  Current diff: {spread.iloc[-1]:.2f} bps")
        print(f"  90-day z-score: {z90.iloc[-1]:.2f}")
    print(f"\nPDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
