r"""
UST 10Y Fair Value OLS Regression Report
==========================================
Based on D:\python\notebook\ustFairBbg.ipynb.

1. Pulls market data and builds a fair-value model for 10Y UST yields.
2. Runs a full-sample OLS regression and reports parameters / VIF.
3. Uses the user's USTDurationStrategy library to produce rolling market-vs-model values.
4. Outputs a PDF with tables and plots.

Run:
    python ust_fair_value_report.py

Output:
    ./ust_fair_value_output/UST_FairValue_Report.pdf
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add user's helper library path
sys.path.append(r"D:\python\pycharm\pythonProject")
from UstDurationModel import USTDurationStrategy
from strategy import Context

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "ust_fair_value_output"

START_DATE = "2021-05-21"

# Main regression tickers
BOND_TICKER = "GT10 Govt"
PREDICTOR_TICKERS = [
    "USSO1Z BGN Curncy",      # 1w OIS
    "S0490FS 1Y1Y BLC Curncy",  # SOFR 1y1y
    "USSWIT5 BGN Curncy",      # 5y infl zero
    "USSWIT10 BGN Curncy",     # 10y infl zero
    "USSNA55 FMCA Curncy",     # 5y5y forward or similar
]
ALL_TICKERS = [BOND_TICKER] + PREDICTOR_TICKERS

# UST supply / Fed holdings tickers
UST_SUPPLY_TICKERS = ["DEBPBOND Index", "DEBPNOTE Index", "DEBPBILL Index"]
FED_HOLDING_TICKER = "FARBNTNM Index"

# Regression setup
PREDICTOR_LIST = [1, 2, 5, 6, 7]  # indices in datFinal
RESPONSE_IDX = 0
ROLLING_WINDOW = 250
STRAT_PARAMS = (1.25, 0.3, 10, 5)  # openThreshold, closeThreshold, maxHoldingDays, stopLoss

# ---------------------------------------------------------------------------
# Curve-spread analysis setup
# ---------------------------------------------------------------------------
TSY_TICKERS = ["GT2 Govt", "GT5 Govt", "GT7 Govt", "GT20 Govt", "GT30 Govt"]
DUMMY_DATE = "2025-04-09"
# Predictor columns used for curve/fly regressions (column names are robust to index shifts)
CURVE_PREDICTORS = [
    "S0490FS 1Y1Y BLC Curncy",  # SOFR 1y1y
    "5y5y Infl Zero",
    "FedHoldRatio",
    "dummy",
]
CURVE_PAIRS: List[Tuple[str, str, str]] = [
    # (long-end, short-end, display_name)
    ("GT5 Govt", "GT2 Govt", "2s5s"),
    ("GT10 Govt", "GT2 Govt", "2s10s"),
    ("GT10 Govt", "GT5 Govt", "5s10s"),
    ("GT30 Govt", "GT5 Govt", "5s30s"),
    ("GT30 Govt", "GT7 Govt", "7s30s"),
    ("GT30 Govt", "GT10 Govt", "10s30s"),
]

# Butterfly triples: (short, belly, long, display_name)
FLY_TRIPLES: List[Tuple[str, str, str, str]] = [
    ("GT2 Govt", "GT5 Govt", "GT10 Govt", "2s5s10s"),
    ("GT5 Govt", "GT10 Govt", "GT30 Govt", "5s10s30s"),
    ("GT10 Govt", "GT20 Govt", "GT30 Govt", "10s20s30s"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bloomberg_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Pull all required tickers from Bloomberg."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(ALL_TICKERS)} main tickers from Bloomberg...")
    df = blp.bdh(tickers=ALL_TICKERS, flds=["px_last"], start_date=start_date, end_date=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.ffill()

    # Compute 5y5y inflation zero
    df["5y5y Infl Zero"] = (
        (((1 + df.iloc[:, 4] / 100) ** 10) / ((1 + df.iloc[:, 3] / 100) ** 5)) ** 0.2 - 1
    ) * 100

    print(f"Loaded main data: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_fed_holdings_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Pull UST outstanding and Fed holdings to compute FedHoldRatio."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today().strftime("%Y-%m-%d")

    print("Downloading UST supply / Fed holdings from Bloomberg...")
    df_total = blp.bdh(tickers=UST_SUPPLY_TICKERS, flds=["px_last"], start_date=start_date, end_date=end_date)
    df_total.index = pd.to_datetime(df_total.index)
    if isinstance(df_total.columns, pd.MultiIndex):
        df_total.columns = df_total.columns.get_level_values(0)

    df_fed = blp.bdh(tickers=[FED_HOLDING_TICKER], flds=["px_last"], start_date=start_date, end_date=end_date)
    df_fed.index = pd.to_datetime(df_fed.index)
    if isinstance(df_fed.columns, pd.MultiIndex):
        df_fed.columns = df_fed.columns.get_level_values(0)

    # Total UST outstanding
    ust_total = pd.DataFrame({"UstTot": df_total.sum(axis=1)})

    # Merge on year-month and forward-fill
    fed = df_fed.copy()
    fed["ym"] = fed.index.to_period("M")
    ust = ust_total.copy()
    ust["ym"] = ust.index.to_period("M")

    merged = fed.reset_index().merge(ust, on="ym", how="left").ffill()
    merged = merged.drop(columns="ym")
    merged["FedHoldRatio"] = merged[FED_HOLDING_TICKER] / merged["UstTot"]
    merged = merged.dropna()

    result = merged[["index", "FedHoldRatio"]].set_index("index")
    result.index = pd.to_datetime(result.index)
    print(f"Loaded Fed holdings ratio: {len(result)} rows")
    return result


def _clean_series_outliers(s: pd.Series) -> pd.Series:
    """Replace isolated single-day bad ticks with interpolated values.

    A point is treated as a bad tick when it jumps away from both the
    previous and next valid observations while those neighbours remain close
    to each other. This preserves genuine level shifts and the original NaN
    boundaries.
    """
    original_nan = s.isna()
    valid = s.dropna()
    daily_changes = valid.diff().abs().dropna()
    if daily_changes.empty:
        return s

    # Adaptive threshold: at least 3 points, otherwise ~3x typical daily move
    threshold = max(4 * daily_changes.median(), 3.0)

    prev = s.shift(1)
    next_ = s.shift(-1)
    gap_prev = (s - prev).abs()
    gap_next = (s - next_).abs()
    gap_neighbors = (prev - next_).abs()

    spike = (
        gap_prev.gt(threshold)
        & gap_next.gt(threshold)
        & gap_neighbors.lt(threshold)
        & ~original_nan
    )

    cleaned = s.where(~spike)
    n_replaced = int(cleaned.isna().sum() - s.isna().sum())
    if n_replaced > 0:
        print(
            f"  Cleaned {n_replaced} single-day spike(s) in {s.name} "
            f"(threshold={threshold:.2f})"
        )
    cleaned = cleaned.interpolate(method="linear")
    cleaned = cleaned.where(~original_nan, np.nan)
    return cleaned


def build_dat_final(df: pd.DataFrame, fed_holdings: pd.DataFrame) -> pd.DataFrame:
    """Merge main data with Fed holdings ratio and clean obvious bad ticks."""
    # The USSNA55 proxy occasionally prints bad ticks (e.g. 2024-06-10: 43.0 vs ~98).
    # Clean it before it distorts the regression and fitted line.
    proxy_col = "USSNA55 FMCA Curncy"
    if proxy_col in df.columns:
        df = df.copy()
        df[proxy_col] = _clean_series_outliers(df[proxy_col])

    dat_final = df.join(fed_holdings, how="left")
    dat_final["FedHoldRatio"] = dat_final["FedHoldRatio"].bfill().ffill()
    dat_final = dat_final.dropna()
    return dat_final


# ---------------------------------------------------------------------------
# Curve-spread analysis (GT2/GT5/GT7/GT10/GT30)
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


def build_merged_data(dat_final: pd.DataFrame, start_date: str = START_DATE) -> pd.DataFrame:
    """Combine fair-value predictors with Treasury yields."""
    tsy = load_treasury_yields(start_date=start_date)
    merged = dat_final.join(tsy, how="left")
    merged = merged.ffill().dropna()
    print(f"Merged dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
    return merged


def run_curve_ols(
    merged_df: pd.DataFrame, crv1: str, crv2: str
) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run OLS for one curve spread. Y is in bps (long-end minus short-end)."""
    Y = (merged_df[crv1] - merged_df[crv2]) * 100.0
    X = merged_df[CURVE_PREDICTORS]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return Y, model


def build_curve_summary_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table for all curve spreads."""
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
    summary = pd.DataFrame(rows).set_index("curve")
    return summary


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
# Fly (butterfly) analysis
# ---------------------------------------------------------------------------
def run_fly_ols(
    merged_df: pd.DataFrame, short: str, belly: str, long: str
) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run OLS for one butterfly. Y is in bps: 2*belly - short - long.

    Adds two fly-specific predictors:
      - wing_slope: long - short (bps)
      - belly_yield: belly yield (percent)
    """
    Y = (2 * merged_df[belly] - merged_df[short] - merged_df[long]) * 100.0
    X = merged_df[CURVE_PREDICTORS].copy()
    X["wing_slope"] = (merged_df[long] - merged_df[short]) * 100.0
    X["belly_yield"] = merged_df[belly]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return Y, model


def build_fly_summary_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table for all butterflies."""
    rows = []
    for short, belly, long, name in FLY_TRIPLES:
        Y, model = run_fly_ols(merged_df, short, belly, long)
        resid_std = float(np.std(model.resid))
        mkt_val = float(Y.iloc[-1])
        mdl_val = float(model.fittedvalues.iloc[-1])
        diff = mkt_val - mdl_val
        ratio = diff / resid_std if resid_std != 0 else np.nan
        rows.append(
            {
                "fly": name,
                "mkt_val": mkt_val,
                "mdl_val": mdl_val,
                "diff": diff,
                "resid_std": resid_std,
                "ratio": ratio,
                "r2": float(model.rsquared),
            }
        )
    summary = pd.DataFrame(rows).set_index("fly")
    return summary


def _single_fly_axis(
    ax, merged_df: pd.DataFrame, short: str, belly: str, long: str, name: str
) -> None:
    """Plot fitted vs actual for one butterfly on a given axis."""
    Y, model = run_fly_ols(merged_df, short, belly, long)
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


def plot_all_flies(merged_df: pd.DataFrame) -> plt.Figure:
    """1x3 grid of fitted-vs-actual plots for all butterflies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False)
    axes = axes.flatten()
    for ax, (short, belly, long, name) in zip(axes, FLY_TRIPLES):
        _single_fly_axis(ax, merged_df, short, belly, long, name)
    fig.suptitle(
        "UST Butterfly Spreads: Actual vs Model Fitted",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------
def run_full_sample_regression(dat_final: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Run full-sample OLS and compute VIF."""
    X = dat_final.iloc[:, PREDICTOR_LIST]
    y = dat_final.iloc[:, RESPONSE_IDX]
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    vif = pd.DataFrame({
        "feature": X_const.columns,
        "vif": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
    })

    return model, vif


def build_regression_params_table(model) -> pd.DataFrame:
    """Build a clean regression parameters table."""
    params = pd.DataFrame({
        "parameter": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    return params


def build_model_summary_dict(model, vif: pd.DataFrame) -> Dict:
    """Collect key regression metrics for the PDF summary."""
    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "resid_std": float(np.std(model.resid)),
        "latest_actual": float(model.model.endog[-1]),
        "latest_fitted": float(model.fittedvalues.iloc[-1]),
        "latest_residual": float(model.resid.iloc[-1]),
        "latest_resid_z": float(model.resid.iloc[-1] / np.std(model.resid)),
        "obs": int(model.nobs),
    }


def build_regression_stats_table(model) -> pd.DataFrame:
    """Build a compact regression-statistics table."""
    stats = pd.DataFrame(
        {
            "statistic": [
                "R-squared",
                "Adj R-squared",
                "F-statistic",
                "F p-value",
                "Residual Std",
                "Observations",
                "Latest Actual",
                "Latest Fitted",
                "Latest Residual",
                "Latest Residual Z",
            ],
            "value": [
                float(model.rsquared),
                float(model.rsquared_adj),
                float(model.fvalue),
                float(model.f_pvalue),
                float(np.std(model.resid)),
                int(model.nobs),
                float(model.model.endog[-1]),
                float(model.fittedvalues.iloc[-1]),
                float(model.resid.iloc[-1]),
                float(model.resid.iloc[-1] / np.std(model.resid)),
            ],
        }
    ).set_index("statistic")
    return stats


# ---------------------------------------------------------------------------
# Rolling market-vs-model values using user's library
# ---------------------------------------------------------------------------
def run_rolling_fair_value(dat_final: pd.DataFrame) -> pd.DataFrame:
    """
    Use USTDurationStrategy to compute rolling market and model values.
    Returns a DataFrame with dates, mktVal, mdlVal, residual, ratio.
    """
    ust_strat = USTDurationStrategy(*STRAT_PARAMS)
    regression_params = [ROLLING_WINDOW, RESPONSE_IDX, PREDICTOR_LIST]
    strat_cont = Context(ust_strat)

    # Use a start date that is inside the data range so we get a useful rolling history
    start_date = date(2024, 5, 1)
    strat_cont.perform_operation([dat_final, regression_params], start_date)

    # Extract signals for all available dates from the strategy
    results = []
    for d in strat_cont._strategy._dates:
        try:
            dt, most_recent_res, m_res_std, ratio, undl_val, pred_val, _ = strat_cont.getSignalOnDate(d)
            results.append({
                "date": pd.to_datetime(dt),
                "mkt_val": undl_val,
                "mdl_val": pred_val,
                "resid": most_recent_res,
                "resid_std": m_res_std,
                "resid_z": ratio,
            })
        except Exception:
            continue

    rolling_df = pd.DataFrame(results).set_index("date").sort_index()
    print(f"Rolling fair value: {len(rolling_df)} observations")
    return rolling_df


def run_recent_rolling_fair_value(
    dat_final: pd.DataFrame,
    start_date: date = date(2025, 2, 11),
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Mimic the UST_Yields.ipynb snippet: query getSignalOnDate over a fixed
    date range and return a DataFrame with market/model values and residuals.
    Residuals are reported in bps (multiplied by 100).
    """
    if end_date is None:
        end_date = date.today()

    ust_strat = USTDurationStrategy(*STRAT_PARAMS)
    regression_params = [ROLLING_WINDOW, RESPONSE_IDX, PREDICTOR_LIST]
    strat_cont = Context(ust_strat)

    # Fit the rolling strategy; start a bit before the desired range so the
    # first requested date already has a fitted window.
    fit_start = min(date(2024, 5, 1), start_date)
    strat_cont.perform_operation([dat_final, regression_params], fit_start)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    results = []
    for d in date_range:
        try:
            dt, most_recent_res, m_res_std, ratio, undl_val, pred_val, _ = (
                strat_cont.getSignalOnDate(d.date())
            )
            results.append(
                {
                    "date": pd.to_datetime(dt),
                    "mkt_val": undl_val,
                    "mdl_val": pred_val,
                    "resid_bps": most_recent_res * 100,
                    "resid_std_bps": m_res_std * 100,
                    "resid_z": ratio,
                }
            )
        except Exception:
            continue

    recent_df = pd.DataFrame(results).set_index("date").sort_index()
    print(f"Recent rolling fair value ({start_date} to {end_date}): {len(recent_df)} observations")
    return recent_df


# ---------------------------------------------------------------------------
# PDF report helpers
# ---------------------------------------------------------------------------
def _idx_date(idx_value) -> date:
    return idx_value.date() if hasattr(idx_value, "date") else idx_value


def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    fontsize: int = 9,
    header_color: str = "#40466e",
    row_height: float = 0.08,
    gradient_columns: Optional[List[str]] = None,
    compact: bool = True,
    index_col_width: float = 0.28,
    include_index: bool = True,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib figure (one page)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    if include_index:
        df = df.reset_index()
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.78])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.965, ha="left", x=0.03)
    if subtitle:
        ax.text(0.03, 0.88, subtitle, fontsize=8, ha="left", transform=fig.transFigure, color="#555555")

    table_data = [display_df.columns.tolist()] + display_df.values.tolist()
    if n_cols > 1 and include_index:
        data_width = 1.0 - index_col_width
        col_widths = [index_col_width] + [data_width / (n_cols - 1)] * (n_cols - 1)
    else:
        col_widths = [1.0 / n_cols] * n_cols

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )

    cell_h = row_height
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

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
        limit = max(abs(vmin), abs(vmax), 0.001)

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
    fontsize: int = 9,
    row_height: float = 0.08,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    compact: bool = True,
    index_col_width: float = 0.28,
    include_index: bool = True,
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
        index_col_width=index_col_width,
        include_index=include_index,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_fitted_vs_actual(model, dat_final: pd.DataFrame, summary: Optional[Dict] = None) -> plt.Figure:
    """Full-sample actual vs fitted plot with current-value annotation."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dat_final.index, dat_final.iloc[:, RESPONSE_IDX], label="Actual 10Y Yield", color="red", alpha=0.7)
    ax.plot(dat_final.index, model.fittedvalues, label="Model Fitted", color="blue", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Full-Sample Actual vs Model Fitted 10Y UST Yield")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if summary is None:
        latest_actual = float(model.model.endog[-1])
        latest_fitted = float(model.fittedvalues.iloc[-1])
        latest_resid = latest_actual - latest_fitted
        resid_std = float(np.std(model.resid))
    else:
        latest_actual = summary["latest_actual"]
        latest_fitted = summary["latest_fitted"]
        latest_resid = summary["latest_residual"]
        resid_std = summary["resid_std"]

    ratio = latest_resid / resid_std if resid_std != 0 else np.nan
    annot_text = (
        f"Current mkt: {latest_actual:.3f}\n"
        f"Current mdl: {latest_fitted:.3f}\n"
        f"Residual: {latest_resid:.3f}\n"
        f"Mkt - mdl / std: {ratio:.2f}"
    )
    ax.text(
        0.02, 0.95, annot_text,
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    fig.tight_layout()
    return fig


def plot_rolling_mkt_vs_mdl(rolling_df: pd.DataFrame) -> plt.Figure:
    """Rolling market vs model value plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_df.index, rolling_df["mkt_val"], label="Market 10Y Yield", color="red", alpha=0.7)
    ax.plot(rolling_df.index, rolling_df["mdl_val"], label="Model Fair Value", color="blue", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Rolling Market vs Model 10Y UST Yield")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Latest annotation
    latest = rolling_df.iloc[-1]
    ax.text(
        0.02, 0.95,
        f"Latest: mkt={latest['mkt_val']:.3f}, mdl={latest['mdl_val']:.3f}, z={latest['resid_z']:.2f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    return fig


def plot_rolling_residual(rolling_df: pd.DataFrame) -> plt.Figure:
    """Plot rolling residual z-score."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_df.index, rolling_df["resid_z"], color="purple", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.25, color="red", linestyle="--", linewidth=0.8, label="open threshold")
    ax.axhline(-1.25, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual / In-Sample Std")
    ax.set_title(f"Rolling Fair-Value Residual Z-Score ({ROLLING_WINDOW}-day rolling OLS)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_recent_rolling_mkt_vs_mdl(recent_df: pd.DataFrame) -> plt.Figure:
    """Market vs model plot for the recent rolling window (mimics UST_Yields.ipynb)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_df.index, recent_df["mkt_val"], label="Market 10Y Yield", color="red", alpha=0.7)
    ax.plot(recent_df.index, recent_df["mdl_val"], label="Model Fair Value", color="blue", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.set_title(
        f"Recent Rolling Market vs Model 10Y UST Yield "
        f"(current mkt={recent_df['mkt_val'].iloc[-1]:.3f}, "
        f"mdl={recent_df['mdl_val'].iloc[-1]:.3f}, "
        f"resid ratio={recent_df['resid_z'].iloc[-1]:.3f}) "
        f"— {ROLLING_WINDOW}-day rolling window"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_recent_rolling_residual(recent_df: pd.DataFrame) -> plt.Figure:
    """Recent rolling residual in bps."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(recent_df.index, recent_df["resid_bps"], color="purple", alpha=0.8, label="residual (bps)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (bps)")
    ax.set_title(f"Recent Rolling Fair-Value Residual ({ROLLING_WINDOW}-day rolling window)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    dat_final: pd.DataFrame,
    model,
    rolling_df: pd.DataFrame,
    recent_df: Optional[pd.DataFrame] = None,
    merged_df: Optional[pd.DataFrame] = None,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate the UST fair-value PDF report (including curve spreads and recent rolling window)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"UST_FairValue_Report_{report_date}.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"UST_FairValue_Report_{report_date}_{ts}.pdf"

    summary = build_model_summary_dict(model, pd.DataFrame())
    params_table = build_regression_params_table(model)

    # Use the parameter names as the index so the table has a single feature column
    params_table = params_table.set_index("parameter").rename_axis("feature")

    subtitle = (
        f"Latest: {_idx_date(dat_final.index[-1])}  |  "
        f"Obs: {summary['obs']}  |  "
        f"R²: {summary['r_squared']:.3f}  |  "
        f"Resid Std: {summary['resid_std']:.3f}"
    )

    with PdfPages(pdf_path) as pdf:
        # 1. Regression parameters table (compact)
        _add_table_page(
            pdf,
            params_table,
            "Full-Sample OLS Regression Parameters",
            subtitle=subtitle,
            gradient_columns=["coef", "p_value"],
            fontsize=8,
            row_height=0.06,
            pagesize=(8.5, 4.5),
            index_col_width=0.22,
        )

        # 1b. Regression statistics table
        stats_table = build_regression_stats_table(model)
        _add_table_page(
            pdf,
            stats_table,
            "Regression Statistics",
            subtitle=subtitle,
            gradient_columns=["value"],
            fontsize=8,
            row_height=0.06,
            pagesize=(6.0, 4.5),
            index_col_width=0.55,
        )

        # 2. Full-sample fitted vs actual plot
        fig = plot_fitted_vs_actual(model, dat_final, summary=summary)
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Rolling mkt vs mdl plot
        fig = plot_rolling_mkt_vs_mdl(rolling_df)
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Rolling residual plot
        fig = plot_rolling_residual(rolling_df)
        pdf.savefig(fig)
        plt.close(fig)

        # 4b. Recent rolling market vs model plot (2025-02-11 onwards)
        if recent_df is not None:
            fig = plot_recent_rolling_mkt_vs_mdl(recent_df)
            pdf.savefig(fig)
            plt.close(fig)

            fig = plot_recent_rolling_residual(recent_df)
            pdf.savefig(fig)
            plt.close(fig)

        # 5. Curve-spread summary table (if Treasury yield data provided)
        if merged_df is not None:
            curve_summary = build_curve_summary_table(merged_df)
            curve_subtitle = (
                f"Latest: {_idx_date(merged_df.index[-1])}  |  "
                f"Obs: {len(merged_df)}  |  "
                f"Predictors: SOFR 1y1y, 5y5y infl zero, FedHoldRatio, dummy"
            )
            _add_table_page(
                pdf,
                curve_summary,
                "UST Curve Spread Fair-Value Summary",
                subtitle=curve_subtitle,
                gradient_columns=["diff", "ratio"],
                fontsize=9,
                row_height=0.07,
                pagesize=(11.0, 4.5),
                index_col_width=0.08,
            )

            # 6. Curve-spread fitted-vs-actual plots
            fig = plot_all_curves(merged_df)
            pdf.savefig(fig)
            plt.close(fig)

            # 7. Fly (butterfly) summary table
            fly_summary = build_fly_summary_table(merged_df)
            fly_subtitle = (
                f"Latest: {_idx_date(merged_df.index[-1])}  |  "
                f"Obs: {len(merged_df)}  |  "
                f"Fly = 2 x belly - short - long (bps)"
            )
            _add_table_page(
                pdf,
                fly_summary,
                "UST Butterfly Spread Fair-Value Summary",
                subtitle=fly_subtitle,
                gradient_columns=["diff", "ratio"],
                fontsize=9,
                row_height=0.08,
                pagesize=(9.0, 4.0),
                index_col_width=0.12,
            )

            # 8. Fly fitted-vs-actual plots
            fig = plot_all_flies(merged_df)
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def _add_summary_page(pdf: PdfPages, dat_final: pd.DataFrame, summary: Dict, params_table: pd.DataFrame) -> None:
    """Cover page with key regression stats."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")

    title = "UST 10Y Fair Value OLS Regression Report"
    subtitle = f"Latest: {_idx_date(dat_final.index[-1])}  |  Observations: {summary['obs']}"

    ax.text(0.5, 0.92, title, fontsize=22, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.86, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)

    stats_text = (
        f"R-squared:        {summary['r_squared']:.4f}\n"
        f"Adj R-squared:    {summary['adj_r_squared']:.4f}\n"
        f"F-statistic:      {summary['f_stat']:.2f}\n"
        f"F p-value:        {summary['f_pvalue']:.4e}\n"
        f"Residual Std:     {summary['resid_std']:.4f}\n"
        f"Latest Actual:    {summary['latest_actual']:.3f}\n"
        f"Latest Fitted:    {summary['latest_fitted']:.3f}\n"
        f"Latest Residual:  {summary['latest_residual']:.3f} ({summary['latest_resid_z']:.2f} std)"
    )
    ax.text(0.15, 0.72, stats_text, fontsize=11, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2ff", edgecolor="#40466e"))

    contents = (
        "Report contents:\n"
        "1. OLS regression parameters\n"
        "2. Variance inflation factors (VIF)\n"
        "3. Full-sample actual vs fitted plot\n"
        "4. Rolling market vs model value plot\n"
        "5. Rolling residual z-score plot"
    )
    ax.text(0.55, 0.72, contents, fontsize=10, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8ee", edgecolor="#8b5a00"))

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_bloomberg_data()
    fed_holdings = load_fed_holdings_data()
    dat_final = build_dat_final(df, fed_holdings)

    print("Running full-sample OLS regression...")
    model, _ = run_full_sample_regression(dat_final)

    print("Running rolling fair-value model via USTDurationStrategy...")
    rolling_df = run_rolling_fair_value(dat_final)

    print("Running recent rolling fair-value window (2025-02-11 onwards)...")
    recent_df = run_recent_rolling_fair_value(dat_final)

    print("Building curve-spread / fly dataset...")
    merged_df = build_merged_data(dat_final)
    curve_summary = build_curve_summary_table(merged_df)
    fly_summary = build_fly_summary_table(merged_df)

    stats_table = build_regression_stats_table(model)

    print("Generating PDF report...")
    pdf_path = generate_pdf_report(
        dat_final, model, rolling_df, recent_df=recent_df, merged_df=merged_df
    )

    print("\n" + "=" * 70)
    print("UST 10Y Fair Value OLS Regression Report")
    print("=" * 70)
    print(f"Latest date : {_idx_date(dat_final.index[-1])}")
    print(f"Observations: {len(dat_final)}")
    print(f"R-squared   : {model.rsquared:.4f}")
    print("\nRegression Statistics")
    print(stats_table.to_string())
    print("\nUST Curve Spread Fair-Value Summary")
    print(curve_summary.to_string())
    print("\nUST Butterfly Spread Fair-Value Summary")
    print(fly_summary.to_string())
    print(f"\nPDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
