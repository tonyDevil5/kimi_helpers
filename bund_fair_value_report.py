r"""
Bund 10Y Fair Value OLS Regression Report
==========================================
Mirrors the structure of ust_fair_value_report.py (interactive HTML only).

Model: Bund 10Y yield (GTDEM10Y Govt, px_last, %) regressed on
    1. ESTR 1y1y forward swap rate (Citi, mbs_rv_citi.xlsx sheet ESTR)
    2. ESTR 3y1y forward swap rate (same sheet)
    3. ECB balance sheet as % of EUR GDP: EBBSTOTA Index (weekly, EUR bn ->
       EUR mn x1000, ffilled daily) / rolling 4-quarter sum of EUGDEMU Index
       (quarterly EUR GDP, EUR mn, ffilled daily), expressed in percent
    4. BTP-OAT 10Y spread (GTITL10Y - GTFRF10Y, bp)

Report sections:
    1. Contents & key stats overview
    2. Full-sample OLS parameters (with latest predictor values)
    3. Regression statistics
    4. Full-sample observed vs fitted vs residual
    5. Rolling 120-observation fair value (observed/fitted/residual)
    6. Rolling 250-observation residual z-score
    7. Rolling 120-observation regression betas

Run:
    python bund_fair_value_report.py

Output:
    ./bund_fair_value_output/Bund_FairValue_Report_<latest data date>.html
"""

from __future__ import annotations

import html
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# Cross-repo import of the shared backtester (same pattern as the
# BackTestingTool strategy examples, which import from this repo).
sys.path.append(r"D:\python\BackTestingTool")
try:
    from simple_backtester import BacktestConfig, SimpleBacktester
    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CITI_FNAME = r"D:\python\notebook\data\mbs\mbs_rv_citi.xlsx"
ESTR_SHEET = "ESTR"
OUTPUT_DIR = Path(__file__).parent / "bund_fair_value_output"
REPORT_TITLE = "Bund 10Y Fair Value OLS Regression Report"

START_DATE = "2015-01-01"

BUND_TICKER = "GTDEM10Y Govt"
BTP_TICKER = "GTITL10Y Govt"
OAT_TICKER = "GTFRF10Y Govt"
ECB_BS_TICKER = "EBBSTOTA Index"   # ECB total assets, weekly, EUR bn
EUR_GDP_TICKER = "EUGDEMU Index"   # EUR GDP, quarterly, EUR mn
BBG_TICKERS = [BUND_TICKER, BTP_TICKER, OAT_TICKER, ECB_BS_TICKER, EUR_GDP_TICKER]

ESTR_FWD_COLS = {
    "ESTR 1y1y": "RATES.OIS.EUR_EUROSTR.FWD.1Y.1Y - CLOSE",
    "ESTR 3y1y": "RATES.OIS.EUR_EUROSTR.FWD.3Y.1Y - CLOSE",
}

RESPONSE_COL = "Bund 10Y"
PREDICTOR_COLS = ["ESTR 1y1y", "ESTR 3y1y", "ECB BS (% GDP)", "BTP-OAT (bp)"]

ROLLING_WINDOW_120 = 120
ROLLING_WINDOW_250 = 250

# ---------------------------------------------------------------------------
# Backtest configuration (mirrors BackTestingTool fair_value_120_backtest.py)
# ---------------------------------------------------------------------------
# Score = rolling-120 residual z (actual - fitted, in yield terms). The
# reference UST analog enters LONG when z >= +2 (yield above model = cheap)
# and SHORT when z <= -2, exiting to flat when |z| <= 0.5. Signals observed
# at close t execute at close t+1 (score shifted by one observation).
BT_LONG_ENTRY = 2.0
BT_SHORT_ENTRY = -2.0
BT_LONG_EXIT = 0.5
BT_SHORT_EXIT = -0.5
BT_DV01_EUR = 1000.0  # EUR per bp per position unit

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_bdh(df, tickers=None) -> pd.DataFrame:
    """Normalize xbbg bdh output to a wide pandas DataFrame.

    xbbg >= 1.x returns a narwhals DataFrame in long format
    (ticker / date / field / value); older versions return a pandas
    DataFrame with MultiIndex columns.
    """
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(
            index="date", columns="ticker", values="value", aggfunc="last"
        )
        if tickers is not None:
            df = df.reindex(columns=[t for t in tickers if t in df.columns])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_bund_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Pull Bund/BTP/OAT yields and ECB BS / EUR GDP from Bloomberg.

    Builds the ECB balance-sheet-to-GDP ratio in percent:
    EBBSTOTA (EUR bn) x1000 -> EUR mn, ffilled daily; EUGDEMU quarterly
    rolling(4).sum() -> trailing annual GDP in EUR mn, ffilled daily;
    ratio = bs / gdp * 100.
    """
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(BBG_TICKERS)} tickers from Bloomberg...")
    df = blp.bdh(tickers=BBG_TICKERS, flds=["px_last"], start_date=start_date, end_date=end_date)
    df = _normalize_bdh(df, tickers=BBG_TICKERS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    out = pd.DataFrame(index=df.index)
    out[RESPONSE_COL] = df[BUND_TICKER]
    out["BTP-OAT (bp)"] = (df[BTP_TICKER] - df[OAT_TICKER]) * 100.0

    # ECB balance sheet as % of trailing-annual EUR GDP
    bs_mn = df[ECB_BS_TICKER].dropna() * 1000.0  # EUR bn -> EUR mn
    gdp_q = df[EUR_GDP_TICKER].dropna()
    gdp_annual = gdp_q.rolling(4, min_periods=4).sum().dropna()  # EUR mn
    daily_index = df.index
    bs_daily = bs_mn.reindex(daily_index).ffill()
    gdp_daily = gdp_annual.reindex(daily_index).ffill()
    out["ECB BS (% GDP)"] = bs_daily / gdp_daily * 100.0

    latest_ratio = out["ECB BS (% GDP)"].dropna().iloc[-1]
    print(
        f"ECB BS/GDP sanity: EBBSTOTA {bs_mn.iloc[-1] / 1000:,.1f} bn / "
        f"rolling 4q GDP {gdp_annual.iloc[-1] / 1e6:,.2f} tn "
        f"= {latest_ratio:.1f}%"
    )
    print(f"Loaded Bund data: {out.shape[0]} rows x {out.shape[1]} cols, "
          f"{out.index[0].date()} to {out.index[-1].date()}")
    return out


def load_estr_forwards(fpath: str = CITI_FNAME) -> pd.DataFrame:
    """Load ESTR 1y1y / 3y1y forward swap rates from the Citi workbook."""
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    try:
        df = pd.read_excel(fpath, sheet_name=ESTR_SHEET, header=1)
    except ValueError as exc:
        raise ValueError(f"Sheet '{ESTR_SHEET}' not found in {fpath}.") from exc

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    out = pd.DataFrame(index=df.index)
    for name, col in ESTR_FWD_COLS.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in sheet '{ESTR_SHEET}'.")
        out[name] = pd.to_numeric(df[col], errors="coerce")
    out = out.dropna(how="all")
    print(f"Loaded ESTR forwards: {out.shape[0]} rows, "
          f"{out.index[0].date()} to {out.index[-1].date()}")
    return out


def build_dat_final(bund: pd.DataFrame, estr: pd.DataFrame) -> pd.DataFrame:
    """Inner-join all predictors + response, ffill lower-frequency series, dropna."""
    merged = bund.join(estr, how="inner")
    merged = merged.ffill().dropna()
    merged = merged[[RESPONSE_COL] + PREDICTOR_COLS]
    print(f"Aligned sample: {merged.shape[0]} obs, "
          f"{merged.index[0].date()} to {merged.index[-1].date()}")
    return merged


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------
def run_full_sample_regression(dat_final: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Full-sample OLS of Bund 10Y on the four predictors (with intercept)."""
    X = sm.add_constant(dat_final[PREDICTOR_COLS])
    y = dat_final[RESPONSE_COL]
    return sm.OLS(y, X).fit()


def build_regression_params_table(model, dat_final: pd.DataFrame) -> pd.DataFrame:
    """Clean regression parameters table with a latest_value column."""
    params = pd.DataFrame({
        "parameter": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    }).set_index("parameter").rename_axis("feature")
    latest_row = dat_final.iloc[-1]
    params.insert(1, "latest_value", [
        round(float(latest_row[f]), 4) if f in dat_final.columns else np.nan
        for f in params.index
    ])
    return params


def build_model_summary_dict(model) -> Dict:
    """Key regression metrics for the report."""
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
    """Compact regression-statistics table."""
    stats = pd.DataFrame(
        {
            "statistic": [
                "R-squared", "Adj R-squared", "F-statistic", "F p-value",
                "Residual Std", "Observations", "Latest Actual", "Latest Fitted",
                "Latest Residual", "Latest Residual Z",
            ],
            "value": [
                float(model.rsquared), float(model.rsquared_adj),
                float(model.fvalue), float(model.f_pvalue),
                float(np.std(model.resid)), int(model.nobs),
                float(model.model.endog[-1]), float(model.fittedvalues.iloc[-1]),
                float(model.resid.iloc[-1]),
                float(model.resid.iloc[-1] / np.std(model.resid)),
            ],
        }
    ).set_index("statistic")
    return stats


def run_rolling_fair_value(dat_final: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling OLS; each point is an in-window estimate for that date.

    Yields and residuals are stored in percent; the renderers convert
    residuals to basis points for display.
    """
    if len(dat_final) < window:
        raise ValueError(f"Need at least {window} observations; got {len(dat_final)}")

    rows = []
    y_all = dat_final[RESPONSE_COL]
    X_all = sm.add_constant(dat_final[PREDICTOR_COLS], has_constant="add")
    for end_idx in range(window, len(dat_final) + 1):
        y_win = y_all.iloc[end_idx - window: end_idx]
        X_win = X_all.iloc[end_idx - window: end_idx]
        model = sm.OLS(y_win, X_win).fit()
        actual = float(y_win.iloc[-1])
        fitted = float(model.fittedvalues.iloc[-1])
        residual = actual - fitted
        resid_std = float(np.std(model.resid))
        rows.append({
            "date": y_win.index[-1],
            "mkt_val": actual,
            "mdl_val": fitted,
            "resid": residual,
            "resid_bps": residual * 100.0,
            "resid_std": resid_std,
            "resid_z": residual / resid_std if resid_std != 0 else np.nan,
            "r2": float(model.rsquared),
        })
    return pd.DataFrame(rows).set_index("date").sort_index()


def run_rolling_betas(dat_final: pd.DataFrame, window: int = ROLLING_WINDOW_120) -> pd.DataFrame:
    """Rolling OLS betas over time (window observations, np.linalg.lstsq)."""
    y = dat_final[RESPONSE_COL].values
    X = np.column_stack([np.ones(len(dat_final)), dat_final[PREDICTOR_COLS].values])
    names = ["const"] + PREDICTOR_COLS
    rows = []
    for end_idx in range(window, len(dat_final) + 1):
        beta, _, _, _ = np.linalg.lstsq(
            X[end_idx - window: end_idx], y[end_idx - window: end_idx], rcond=None
        )
        rows.append(dict(zip(names, beta)) | {"date": dat_final.index[end_idx - 1]})
    return pd.DataFrame(rows).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# 120-observation fair-value strategy backtest
# ---------------------------------------------------------------------------
def _period_metrics(daily: pd.DataFrame) -> Dict[str, float]:
    """PnL/Sharpe/drawdown metrics for a (sub-)period of backtest daily rows."""
    pnl = daily["pnl_eur"]
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    vol = pnl.std(ddof=1)
    years = len(daily) / 252.0
    total = float(pnl.sum())
    ann = total / years if years > 0 else np.nan
    max_dd = float(drawdown.min())
    return {
        "obs": len(daily),
        "pnl_eur": total,
        "annualized_eur": ann,
        "sharpe": float(np.sqrt(252) * pnl.mean() / vol) if vol > 0 else np.nan,
        "max_drawdown_eur": max_dd,
        "calmar": ann / abs(max_dd) if max_dd < 0 else np.nan,
        "time_in_market_pct": float(100 * (daily["held_position"] != 0).mean()),
    }


def run_backtest(rolling_df: pd.DataFrame) -> Dict[str, object]:
    """120-obs fair-value mean-reversion backtest on the rolling residual z.

    Entry long when z >= +2 (yield above model = cheap), short when z <= -2,
    exit to flat when |z| <= 0.5. Signal at close t executes at close t+1
    (execution_score = resid_z.shift(1)); position +1 = long the Bund, daily
    PnL = -held_position x dyield(%) x 100 x DV01, DV01 = EUR 1,000/bp.
    """
    if not BACKTESTER_AVAILABLE:
        raise ImportError("simple_backtester not importable from D:\\python\\BackTestingTool")

    d = rolling_df.sort_index().copy()
    # Yesterday's signal can only trade at today's close (no lookahead)
    d["execution_score"] = d["resid_z"].shift(1)
    d["yield_decimal"] = d["mkt_val"] / 100.0  # percent -> decimal

    bt = SimpleBacktester(d, "yield_decimal", BacktestConfig(
        instrument_type="rates", score_col="execution_score",
        long_entry=BT_LONG_ENTRY, short_entry=BT_SHORT_ENTRY,
        long_exit=BT_LONG_EXIT, short_exit=BT_SHORT_EXIT,
        freeze_after_stop_loss=0,
    ))
    result = bt.run()
    d["position"] = result.position
    d["held_position"] = result.position.shift(1).fillna(0)
    d["daily_pnl_bp"] = result.daily_pnl
    d["pnl_eur"] = result.daily_pnl * BT_DV01_EUR
    d["cumulative_pnl_eur"] = d["pnl_eur"].cumsum()

    # Duration-matched buy-and-hold Bund benchmark (always +1, same DV01)
    d["bh_pnl_eur"] = -d["mkt_val"].diff().fillna(0) * 100.0 * BT_DV01_EUR
    d["bh_cumulative_pnl_eur"] = d["bh_pnl_eur"].cumsum()

    trades = bt.trades_df()
    trades["pnl_eur"] = trades["pnl"] * BT_DV01_EUR
    trades["entry_pct"] = trades["entry_price"] * 100.0
    trades["exit_pct"] = trades["exit_price"] * 100.0
    trades["status"] = np.where(trades.exit_reason == "end_of_data", "Marked at sample end", "Closed")

    # Reconciliation checks (same discipline as the UST analog)
    if len(trades) and not np.isclose(trades["pnl_eur"].sum(), d["pnl_eur"].sum(), atol=1e-6):
        raise AssertionError("Trade P&L does not reconcile to daily P&L")
    expected = -d["held_position"] * d["mkt_val"].diff().fillna(0) * 100.0 * BT_DV01_EUR
    if not np.allclose(expected, d["pnl_eur"], atol=1e-6):
        raise AssertionError("Position timing / bp scaling mismatch")

    closed = trades[trades.exit_reason != "end_of_data"]
    summary = _period_metrics(d)
    summary.update({
        "start": str(d.index[0].date()),
        "end": str(d.index[-1].date()),
        "trades": int(len(trades)),
        "closed_trades": int(len(closed)),
        "win_rate_pct": float(100 * (closed.pnl > 0).mean()) if len(closed) else np.nan,
        "avg_holding_obs": float(closed.holding_days.mean()) if len(closed) else np.nan,
        "latest_z": float(d["resid_z"].iloc[-1]),
        "current_position": int(d["position"].iloc[-1]),
    })
    ytd = d.loc[d.index >= pd.Timestamp(f"{d.index[-1].year}-01-01")]
    ytd_summary = _period_metrics(ytd) if not ytd.empty else {}
    return {"daily": d, "trades": trades, "summary": summary, "ytd": ytd_summary}


def _fig_backtest(daily: pd.DataFrame) -> go.Figure:
    """Cumulative strategy vs buy-and-hold PnL (EUR) with position subplot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06,
        subplot_titles=["Cumulative P&L (EUR, DV01 = €1,000/bp)", "Position (+1 long / -1 short)"],
    )
    fig.add_trace(
        go.Scatter(x=_xs(daily.index), y=_ys(daily["cumulative_pnl_eur"]), mode="lines",
                   name="FV strategy", line=dict(color="#2563eb", width=1.4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(daily.index), y=_ys(daily["bh_cumulative_pnl_eur"]), mode="lines",
                   name="Buy & hold Bund (duration-matched)",
                   line=dict(color="#94a3b8", width=1.1, dash="dash")),
        row=1, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=_xs(daily.index), y=_ys(daily["position"]), mode="lines",
                   name="Position", line=dict(color="#059669", width=1.2, shape="hv"),
                   fill="tozeroy", fillcolor="rgba(5, 150, 105, 0.12)"),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="EUR", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    _apply_layout(
        fig, "120-Observation Fair-Value Strategy Backtest", height=680,
        subtitle=(
            f"Long when z ≥ +{BT_LONG_ENTRY:g} (yield above model), short when z ≤ "
            f"{BT_SHORT_ENTRY:g}, exit |z| ≤ {BT_LONG_EXIT:g}  |  T+1 execution  |  "
            "gross of carry, roll and transaction costs"
        ),
    )
    return fig


def _backtest_summary_html(summary: Dict[str, float], ytd: Dict[str, float]) -> str:
    """Metric/value table for the backtest (full sample + current-year YTD)."""
    pos_label = {1: "Long", -1: "Short", 0: "Flat"}[int(summary["current_position"])]

    def row(metric: str, full: str, ytd_val: str = "—") -> str:
        return f"<tr><td>{html.escape(metric)}</td><td>{html.escape(full)}</td><td>{html.escape(ytd_val)}</td></tr>"

    rows = [
        row("Sample", f"{summary['start']} to {summary['end']} ({summary['obs']:,} obs)",
            f"{ytd.get('obs', 0):,} obs" if ytd else "—"),
        row("Total PnL (EUR)", f"€{summary['pnl_eur']:,.0f}",
            f"€{ytd['pnl_eur']:,.0f}" if ytd else "—"),
        row("Annualized PnL (EUR)", f"€{summary['annualized_eur']:,.0f}",
            f"€{ytd['annualized_eur']:,.0f}" if ytd else "—"),
        row("Sharpe (daily, ann.)", f"{summary['sharpe']:.2f}",
            f"{ytd['sharpe']:.2f}" if ytd else "—"),
        row("Max drawdown (EUR)", f"€{summary['max_drawdown_eur']:,.0f}",
            f"€{ytd['max_drawdown_eur']:,.0f}" if ytd else "—"),
        row("Calmar", f"{summary['calmar']:.2f}" if pd.notna(summary["calmar"]) else "n/a",
            f"{ytd['calmar']:.2f}" if ytd and pd.notna(ytd.get("calmar")) else "—"),
        row("Win rate (closed trades)", f"{summary['win_rate_pct']:.1f}%"),
        row("Trades (entries / closed)", f"{summary['trades']} / {summary['closed_trades']}"),
        row("Avg holding (obs)", f"{summary['avg_holding_obs']:.1f}"),
        row("Time in market", f"{summary['time_in_market_pct']:.1f}%"),
        row("Latest residual z (120d)", f"{summary['latest_z']:+.2f}"),
        row("Current position", pos_label),
    ]
    return (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        "<th>Metric</th><th>Full sample</th><th>YTD</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: go.Figure, title: str, height: int = 520, subtitle: Optional[str] = None) -> go.Figure:
    title_text = title.replace("\n", "<br>")
    if subtitle:
        title_text += f'<br><span style="font-size:12px;color:#64748b">{subtitle}</span>'
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(font=dict(size=11)),
    )
    return fig


def _fig_fitted_vs_actual(dat_final: pd.DataFrame, model, summary: Dict) -> go.Figure:
    """Full-sample actual vs fitted with residual panel (mirrors UST report)."""
    y = dat_final[RESPONSE_COL]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=_xs(y.index), y=_ys(y), mode="lines", name="Actual Bund 10Y",
                   line=dict(color="firebrick", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(y.index), y=_ys(model.fittedvalues), mode="lines",
                   name="Full-sample fitted", line=dict(color="royalblue", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(y.index), y=_ys(model.resid), mode="lines", fill="tozeroy",
                   name="Residual = actual - fitted", line=dict(color="slateblue", width=1),
                   fillcolor="rgba(106, 90, 205, 0.15)"),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (%)", row=2, col=1)
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.02, y=0.03,
        text=(
            f"Latest actual: {summary['latest_actual']:.3f}%<br>"
            f"Latest fitted: {summary['latest_fitted']:.3f}%<br>"
            f"Residual: {summary['latest_residual']:+.3f} ({summary['latest_residual'] * 100:+.1f} bp)"
            f"  |  z = {summary['latest_resid_z']:.2f}"
        ),
        showarrow=False, xanchor="left", yanchor="bottom", align="left",
        font=dict(size=10), bgcolor="white", bordercolor="#cbd5e1",
        borderwidth=1, borderpad=4, opacity=0.95,
    )
    _apply_layout(
        fig, "Full-Sample Actual vs Fitted Bund 10Y Yield", height=720,
        subtitle=(
            f"Sample: {y.index[0].date()} to {y.index[-1].date()}  |  "
            f"N = {summary['obs']:,}  |  R² = {summary['r_squared']:.3f}  |  "
            f"Predictors: {', '.join(PREDICTOR_COLS)}"
        ),
    )
    return fig


def _fig_rolling_fv(rolling_df: pd.DataFrame, window: int) -> go.Figure:
    latest = rolling_df.iloc[-1]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=_xs(rolling_df.index), y=_ys(rolling_df["mkt_val"]), mode="lines",
                   name="Actual Bund 10Y", line=dict(color="firebrick", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(rolling_df.index), y=_ys(rolling_df["mdl_val"]), mode="lines",
                   name=f"{window}-day rolling fitted", line=dict(color="royalblue", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(rolling_df.index), y=_ys(rolling_df["resid"]), mode="lines", fill="tozeroy",
                   name="Residual = actual - fitted", line=dict(color="slateblue", width=1),
                   fillcolor="rgba(106, 90, 205, 0.15)"),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (%)", row=2, col=1)
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.02, y=0.03,
        text=(
            f"Latest actual: {latest['mkt_val']:.3f}%<br>"
            f"Latest fitted: {latest['mdl_val']:.3f}%<br>"
            f"Residual: {latest['resid']:+.3f} ({latest['resid_bps']:+.1f} bp)"
            f"  |  z = {latest['resid_z']:.2f}"
        ),
        showarrow=False, xanchor="left", yanchor="bottom", align="left",
        font=dict(size=10), bgcolor="white", bordercolor="#cbd5e1",
        borderwidth=1, borderpad=4, opacity=0.95,
    )
    _apply_layout(
        fig, f"Rolling {window}-Observation Fair Value: Actual, Fitted and Residual",
        height=720,
        subtitle=(
            f"Rolling estimates: {rolling_df.index[0].date()} to {rolling_df.index[-1].date()}  |  "
            f"Window = {window} observations  |  Predictors: {', '.join(PREDICTOR_COLS)}"
        ),
    )
    return fig


def _fig_rolling_resid_z(rolling_df: pd.DataFrame, window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=_xs(rolling_df.index), y=_ys(rolling_df["resid_z"]), mode="lines",
                   name=f"{window}d residual z", line=dict(color="purple", width=1.2))
    )
    x0, x1 = _xs(rolling_df.index[:1])[0], _xs(rolling_df.index[-1:])[0]
    for level, color, dash in ((0, "#111827", "solid"), (2, "#b91c1c", "dash"), (-2, "#b91c1c", "dash")):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                      line=dict(color=color, width=0.9, dash=dash))
    fig.update_yaxes(title_text="Residual / in-sample std")
    _apply_layout(
        fig, f"Rolling {window}-Observation Fair-Value Residual Z-Score", height=440,
        subtitle=f"Latest z: {rolling_df['resid_z'].iloc[-1]:+.2f}",
    )
    return fig


def _fig_rolling_betas(betas: pd.DataFrame, window: int) -> go.Figure:
    fig = go.Figure()
    colors = ["#64748b", "#2563eb", "#059669", "#d97706", "#dc2626"]
    for i, col in enumerate(betas.columns):
        fig.add_trace(
            go.Scatter(x=_xs(betas.index), y=_ys(betas[col]), mode="lines", name=col,
                       line=dict(color=colors[i % len(colors)], width=1.2))
        )
    fig.add_hline(0, line=dict(color="#111827", width=0.7))
    _apply_layout(
        fig, f"Rolling {window}-Observation Regression Betas", height=520,
        subtitle="Coefficients re-estimated on each trailing window (in-sample).",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
def _df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as an HTML table (toolbar wired up in the page JS)."""
    df = df.reset_index()
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")

    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    for col in display.columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")
    for i in range(len(display)):
        parts.append("<tr>" + "".join(
            f"<td>{html.escape(str(display[col].iloc[i]))}</td>" for col in display.columns
        ) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


_CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--panel:#fff;--bg:#eef2f6;--accent:#174a6e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
.header{position:sticky;top:0;z-index:100;background:#102b3c;color:#e7f0f5;padding:10px 18px;
box-shadow:0 2px 8px rgba(15,23,42,.25)}
.header .tb-title{font-size:16px;font-weight:bold}
.header .tb-sub{font-size:12px;color:#b9cad5;margin-top:2px}
main{padding:20px;max-width:1500px;margin:0 auto}
.section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
box-shadow:0 3px 12px rgba(15,23,42,.06);padding:16px 18px;margin-bottom:20px}
.section h2{font-size:17px;margin:2px 0 12px;color:var(--accent)}
.chart{width:100%;margin-bottom:18px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px;margin-bottom:12px}
.table-tools{display:flex;gap:8px;align-items:center;margin:2px 0 6px;font-size:12px}
.table-tools input[type=search]{padding:5px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:12px;min-width:180px}
.table-tools button{border:1px solid #b6c5cf;background:#fff;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px}
.table-count{color:var(--muted)}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 8px;font-weight:bold;text-align:center;white-space:nowrap}
.sortable-table th{cursor:pointer;user-select:none}
.sortable-table th.sort-asc::after{content:" ▲";font-size:9px}
.sortable-table th.sort-desc::after{content:" ▼";font-size:9px}
.data-table td{border:1px solid #e2e8f0;padding:4px 8px;text-align:center;white-space:nowrap}
.ov{display:flex;gap:16px;flex-wrap:wrap}
.ov-box{border:1px solid #40466e;border-radius:8px;background:#f7f8fc;padding:14px 18px;flex:1;min-width:340px}
.ov-box h3{margin:0 0 8px;font-size:14px}
.ov-box pre{font:12px/1.6 monospace;margin:0;white-space:pre-wrap}
.note{font-size:12px;color:var(--muted)}
@media print{.header{position:static}.section{break-after:page}}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
document.querySelectorAll('.chart').forEach(el=>renderChart(el.id));
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text==='')return {blank:true,num:false,value:''};const n=Number(text.replace(/%$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
function csvCell(value){return '"'+value.replace(/"/g,'""')+'"';}
document.querySelectorAll('.sortable-table').forEach((table,tableIndex)=>{
const wrap=table.closest('.table-wrap');if(!wrap)return;const rows=[...table.tBodies[0].rows];const tools=document.createElement('div');tools.className='table-tools';
const search=document.createElement('input');search.type='search';search.placeholder='Filter this table';search.setAttribute('aria-label','Filter this table');
const reset=document.createElement('button');reset.type='button';reset.textContent='Reset sort';
const csv=document.createElement('button');csv.type='button';csv.textContent='Download CSV';
const count=document.createElement('span');count.className='table-count';tools.append(search,reset,csv,count);wrap.parentNode.insertBefore(tools,wrap);
const refresh=()=>{let shown=0;rows.forEach(row=>{if(row.style.display!=='none')shown++;});count.textContent=`${shown} of ${rows.length} rows`;};refresh();
search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();rows.forEach(row=>{row.style.display=(!q||row.textContent.toLowerCase().includes(q))?'':'none';});refresh();});
table.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{const asc=!th.classList.contains('sort-asc');table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));th.classList.add(asc?'sort-asc':'sort-desc');const sorted=[...rows].sort((a,b)=>{const av=cellValue(a.cells[col]),bv=cellValue(b.cells[col]);if(av.blank!==bv.blank)return av.blank?1:-1;let c;if(av.num&&bv.num)c=av.value-bv.value;else c=String(av.value).localeCompare(String(bv.value));return asc?c:-c;});sorted.forEach(row=>table.tBodies[0].appendChild(row));}));
reset.addEventListener('click',()=>{rows.forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));});
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`bund-fairvalue-table-${String(tableIndex+1).padStart(2,'0')}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
});
"""


class _HtmlBuilder:
    """Collects figures as JSON and emits self-contained HTML blocks."""

    def __init__(self) -> None:
        self._fig_count = 0

    def fig_block(self, fig: go.Figure) -> str:
        self._fig_count += 1
        fid = f"fig-{self._fig_count:03d}"
        fig_json = pio.to_json(fig).replace("</", "<\\/")
        return (
            f'<div class="chart" id="{fid}"></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{fig_json}</script>"
        )


def build_html(payloads: Dict, report_path: Path) -> Path:
    builder = _HtmlBuilder()
    dat_final = payloads["dat_final"]
    summary = payloads["summary"]
    as_of = dat_final.index[-1].date()

    stats_text = (
        f"Latest actual:   {summary['latest_actual']:.3f}%\n"
        f"Latest fitted:   {summary['latest_fitted']:.3f}%\n"
        f"Residual:        {summary['latest_residual']:+.3f} ({summary['latest_resid_z']:+.2f} std)\n"
        f"R-squared:       {summary['r_squared']:.4f}\n"
        f"Adj R-squared:   {summary['adj_r_squared']:.4f}\n"
        f"Residual std:    {summary['resid_std']:.4f}\n"
        f"Observations:    {summary['obs']:,}\n"
        f"Sample:          {dat_final.index[0].date()} to {as_of}"
    )
    contents = (
        "1. Full-sample OLS parameters (with latest predictor values)\n"
        "2. Regression statistics\n"
        "3. Full-sample observed vs fitted vs residual\n"
        f"4. Rolling {ROLLING_WINDOW_120}-observation fair value\n"
        "5. 120-day fair-value strategy backtest (EUR, DV01 = 1,000/bp)\n"
        f"6. Rolling {ROLLING_WINDOW_250}-observation residual z-score\n"
        f"7. Rolling {ROLLING_WINDOW_120}-observation regression betas"
    )
    methodology = (
        "Response: Bund 10Y yield (GTDEM10Y Govt, %). Predictors: ESTR 1y1y / 3y1y "
        "forward swap rates (Citi, mbs_rv_citi.xlsx sheet ESTR, %); ECB balance sheet "
        "as % of trailing-annual EUR GDP (EBBSTOTA Index, EUR bn x1000 -> EUR mn, "
        "ffilled daily, divided by rolling 4-quarter sum of EUGDEMU Index, EUR mn, "
        "ffilled daily, x100); BTP-OAT 10Y spread (GTITL10Y - GTFRF10Y, bp). "
        "All series inner-joined on date, lower-frequency series forward-filled, "
        "then dropna."
    )

    sections = [
        '<section class="section"><h2>Contents &amp; Key Stats</h2>'
        '<div class="ov">'
        f'<div class="ov-box"><h3>Key regression stats</h3><pre>{html.escape(stats_text)}</pre></div>'
        f'<div class="ov-box"><h3>Report contents</h3><pre>{html.escape(contents)}</pre></div>'
        "</div>"
        f'<p class="note" style="margin-top:10px">Methodology: {html.escape(methodology)}</p>'
        "</section>",
        '<section class="section"><h2>Full-Sample OLS Regression Parameters</h2>'
        + _df_to_html_table(payloads["params_table"])
        + "</section>",
        '<section class="section"><h2>Regression Statistics</h2>'
        + _df_to_html_table(payloads["stats_table"])
        + "</section>",
        '<section class="section"><h2>Full-Sample Actual vs Fitted</h2>'
        + builder.fig_block(payloads["fig_full_sample"])
        + "</section>",
        f'<section class="section"><h2>Rolling {ROLLING_WINDOW_120}-Observation Fair Value</h2>'
        + builder.fig_block(payloads["fig_rolling_120"])
        + "</section>",
        '<section class="section"><h2>120-Day Fair-Value Strategy Backtest</h2>'
        + _backtest_summary_html(payloads["backtest"]["summary"], payloads["backtest"]["ytd"])
        + builder.fig_block(payloads["fig_backtest"])
        + '<p class="note">Signal: rolling 120-obs residual z (actual − fitted yield). '
        "Long entry z ≥ +2 (yield above model = cheap), short entry z ≤ −2, exit to flat "
        "when |z| ≤ 0.5. Signal at close t executes at close t+1; the new position first "
        "earns the t+1 → t+2 move. Fixed DV01 = €1,000/bp; position +1 = long the Bund. "
        "Gross approximation: no carry, financing, roll or transaction costs. Buy-and-hold "
        "benchmark is duration-matched (always +1, same DV01). Backtest starts after the "
        "120-observation warmup on the aligned sample (2021-10-27 start).</p>"
        "</section>",
        f'<section class="section"><h2>Rolling {ROLLING_WINDOW_250}-Observation Residual Z-Score</h2>'
        + builder.fig_block(payloads["fig_rolling_250_z"])
        + "</section>",
        f'<section class="section"><h2>Rolling {ROLLING_WINDOW_120}-Observation Betas</h2>'
        + builder.fig_block(payloads["fig_betas"])
        + "</section>",
    ]

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<nav class="header"><div class="tb-title">{html.escape(REPORT_TITLE)}</div>
<div class="tb-sub">as of {as_of}  |  Obs: {summary['obs']:,}  |  R² {summary['r_squared']:.3f}</div></nav>
<main>{''.join(sections)}</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def assemble_report_data(dat_final: pd.DataFrame) -> Dict:
    """Run the regressions and build all figures once."""
    print("Running full-sample OLS regression...")
    model = run_full_sample_regression(dat_final)
    summary = build_model_summary_dict(model)
    params_table = build_regression_params_table(model, dat_final)
    stats_table = build_regression_stats_table(model)

    print(f"Running rolling {ROLLING_WINDOW_120}-observation fair-value OLS...")
    rolling_120 = run_rolling_fair_value(dat_final, window=ROLLING_WINDOW_120)
    print(f"Running rolling {ROLLING_WINDOW_250}-observation fair-value OLS...")
    rolling_250 = run_rolling_fair_value(dat_final, window=ROLLING_WINDOW_250)
    print(f"Computing rolling {ROLLING_WINDOW_120}-observation betas...")
    betas = run_rolling_betas(dat_final, window=ROLLING_WINDOW_120)

    print("Running 120-observation fair-value strategy backtest...")
    backtest = run_backtest(rolling_120)

    print("Building figures...")
    return {
        "dat_final": dat_final,
        "model": model,
        "summary": summary,
        "params_table": params_table,
        "stats_table": stats_table,
        "rolling_120": rolling_120,
        "rolling_250": rolling_250,
        "betas": betas,
        "backtest": backtest,
        "fig_full_sample": _fig_fitted_vs_actual(dat_final, model, summary),
        "fig_rolling_120": _fig_rolling_fv(rolling_120, ROLLING_WINDOW_120),
        "fig_rolling_250_z": _fig_rolling_resid_z(rolling_250, ROLLING_WINDOW_250),
        "fig_betas": _fig_rolling_betas(betas, ROLLING_WINDOW_120),
        "fig_backtest": _fig_backtest(backtest["daily"]),
    }


def print_console_summary(payloads: Dict, report_path: Path) -> None:
    summary = payloads["summary"]
    dat_final = payloads["dat_final"]
    print("\n" + "=" * 70)
    print(REPORT_TITLE)
    print("=" * 70)
    print(f"Aligned sample : {dat_final.index[0].date()} to {dat_final.index[-1].date()} "
          f"({summary['obs']:,} obs)")
    print(f"R-squared      : {summary['r_squared']:.4f}")
    print(f"Latest actual  : {summary['latest_actual']:.3f}%")
    print(f"Latest fitted  : {summary['latest_fitted']:.3f}%")
    print(f"Residual       : {summary['latest_residual']:+.3f} "
          f"({summary['latest_residual'] * 100:+.1f} bp, z {summary['latest_resid_z']:+.2f})")
    print("\nRegression parameters (latest predictor values):")
    print(payloads["params_table"].to_string())
    rolling_120 = payloads["rolling_120"]
    print(f"\nRolling {ROLLING_WINDOW_120}d latest z: {rolling_120['resid_z'].iloc[-1]:+.2f}")
    rolling_250 = payloads["rolling_250"]
    print(f"Rolling {ROLLING_WINDOW_250}d latest z: {rolling_250['resid_z'].iloc[-1]:+.2f}")

    bt = payloads["backtest"]["summary"]
    ytd = payloads["backtest"]["ytd"]
    pos_label = {1: "Long", -1: "Short", 0: "Flat"}[int(bt["current_position"])]
    calmar_txt = f"{bt['calmar']:.2f}" if pd.notna(bt["calmar"]) else "n/a"
    print("\n120-Day Fair-Value Strategy Backtest (DV01 = EUR 1,000/bp)")
    print(f"  Full sample ({bt['start']} to {bt['end']}): "
          f"PnL {bt['pnl_eur']:,.0f} EUR  |  Sharpe {bt['sharpe']:.2f}  |  "
          f"max DD {bt['max_drawdown_eur']:,.0f} EUR  |  Calmar {calmar_txt}")
    print(f"  Trades {bt['trades']} ({bt['closed_trades']} closed)  |  "
          f"win rate {bt['win_rate_pct']:.1f}%  |  "
          f"time in market {bt['time_in_market_pct']:.1f}%")
    if ytd:
        print(f"  YTD {pd.Timestamp(bt['end']).year}: PnL {ytd['pnl_eur']:,.0f} EUR  |  "
              f"Sharpe {ytd['sharpe']:.2f}")
    print(f"  Latest z (120d) {bt['latest_z']:+.2f}  |  current position: {pos_label}")
    print(f"\nHTML report saved to: {report_path}")
    print("=" * 70)


def main() -> Path:
    bund = load_bund_data()
    estr = load_estr_forwards()
    dat_final = build_dat_final(bund, estr)

    payloads = assemble_report_data(dat_final)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = dat_final.index[-1].strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"Bund_FairValue_Report_{report_date}.html"

    print("Writing interactive HTML report...")
    build_html(payloads, report_path)

    print_console_summary(payloads, report_path)
    return report_path


if __name__ == "__main__":
    main()
