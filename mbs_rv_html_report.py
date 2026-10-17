r"""
MBS RV Analysis Report - interactive HTML version
=================================================
HTML counterpart of mbs_rv_report.py. Reuses the analysis layer of
``mbs_rv_report`` (data loading, summary tables, OLS models, momentum and
correlation computations) and the shared ``html_report_utils`` helpers, and
renders the report as a single self-contained HTML file with Plotly charts.

Skipped vs the PDF: the per-structure historical appendix pages, the
perf-method comparison (CSV-only), and dead-code sections.

Run:
    python mbs_rv_html_report.py

Output:
    ./mbs_rv_output/MBS_RV_Report_Plotly_<YYYYMMDD>.html
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mbs_rv_report import (
    CC_FN_10Y_CORR_WINDOWS,
    CC_FN_10Y_CORR_Z_WINDOW,
    CC_LAG_PREDICTORS,
    CC_MA_WINDOWS,
    CC_MOM_WINDOWS,
    CC_PREDICTORS,
    CC_ROLLING_WINDOW,
    CC_TICKERS,
    CC_VOL_ANN_FACTOR,
    CC_VOL_WINDOWS,
    CITI_FNAME,
    COUPON_BASIS_LAG_PREDICTORS,
    COUPON_BASIS_PREDICTORS,
    COUPON_DROP_Z_WINDOW,
    COUPON_FLY_CENTERS,
    COUPON_FORWARD_VOL_COL,
    COUPON_FORWARD_VOL_MONTHS,
    COUPON_SHARPE_MONTH_DAYS,
    COUPON_SHARPE_VOL_WINDOW,
    COUPON_SWAP_PAIRS,
    COUPONS,
    DEFAULT_PERF_WINDOWS,
    DEFAULT_SCORE_INDICES,
    FNAME,
    G2_FN_CORR_WINDOWS,
    G2_FN_PERF_COUPONS,
    G2_FN_RECENT_PLOT_WINDOW,
    OUTPUT_DIR,
    ROLL_SUM_WINDOW,
    ROLL_SUM_Z_WINDOW,
    STRUCTURE_AGENCIES,
    STRUCTURE_MOM_WINDOWS,
    STRUCTURE_RECENT_PLOT_WINDOW,
    TSY_OAS_LOOKBACK,
    WIDE_COUPON_SWAP_PAIRS,
    _adjacent_swap_specs,
    _col,
    _fly_specs,
    _g2_fn_specs,
    _group_cpn_fly_rows,
    _group_yield_basis_rows,
    _wide_swap_specs,
    add_ols_metrics_to_summary,
    add_range_metrics_to_summary,
    build_cc_ma_vol_summary,
    build_cc_momentum,
    build_citi_oas_summary,
    build_citi_perf_summary,
    build_cpn_fly_summary_table,
    build_current_coupon_basis_df,
    build_current_coupon_params_table,
    build_current_coupon_summary,
    build_coupon_basis_coef_matrix,
    build_coupon_basis_df,
    build_coupon_basis_params_table,
    build_coupon_basis_summary,
    build_coupon_fly_price_summary,
    build_coupon_fly_price_timeseries,
    build_coupon_swap_jpm_perf_summary,
    build_coupon_swap_jpm_perf_timeseries,
    build_fn_10y_all_sample_summary,
    build_fn_10y_corr_summary,
    build_fn_10y_rolling_corr,
    build_g2_fn_jpm_perf_summary,
    build_g2_fn_jpm_perf_timeseries,
    build_latest_price_table,
    build_perf_summary_table,
    build_rolling_perf_zscore_table,
    build_structure_momentum,
    build_tsy_oas_summary,
    build_wide_coupon_swap_jpm_perf_summary,
    build_wide_coupon_swap_jpm_perf_timeseries,
    build_yield_basis_summary_table,
    combine_citi_perf_history,
    load_citi_oas_data,
    load_citi_perf_data,
    load_citi_perf_intraday,
    load_mbs_rv,
    load_tsy_oas_data,
    load_ust_yield_vol_data,
    run_coupon_basis_ols,
    run_current_coupon_ols,
    run_rolling_current_coupon_basis,
)
from html_report_utils import df_to_html_table, figs_to_html, page_template

# PDF color conventions
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
CORR_COLORS = {20: "#2563eb", 40: "#d97706", 60: "#059669"}
MOM_COLORS = {20: "#dc2626", 60: "#2563eb"}
MA_COLORS = {50: "orange", 100: "green", 200: "purple"}
VOL_COLORS = {20: "red", 40: "blue", 60: "gray"}
FN_10Y_COLORS = {20: "#dc2626", 40: "#2563eb", 60: "#475569"}
AGENCY_NAMES = {"FN": "Conventional", "G2": "Ginnie"}
DROP_COLORS = {"FN": "#2563eb", "G2": "#dc2626"}
SHARPE_COLORS = {"FN": "#7c3aed", "G2": "#059669"}
FWDVOL_COLORS = {"FN": "#d97706", "G2": "#0891b2"}


def _hline(fig: go.Figure, y: float, dash: str = "dash", color: str = "#64748b",
           row=None, col=None) -> None:
    fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=1, row=row, col=col)


# ---------------------------------------------------------------------------
# Structure-group Plotly figures
# ---------------------------------------------------------------------------
def fig_g2_fn_recent_perf(perf_data: pd.DataFrame) -> go.Figure:
    """Trailing-window cumulative G2-minus-FN JPM performance, two coupon panels."""
    window = G2_FN_RECENT_PLOT_WINDOW
    groups = [G2_FN_PERF_COUPONS[:4], G2_FN_PERF_COUPONS[4:]]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"G2/FN {g[0]:g} to {g[-1]:g}" for g in groups],
    )
    for r, coupons in enumerate(groups, start=1):
        for coupon in coupons:
            col = f"g2_fn_{coupon:g}_perf_32nds"
            cumulative = perf_data[col].dropna().tail(window).cumsum()
            color = TAB10[G2_FN_PERF_COUPONS.index(coupon) % 10]
            fig.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative, mode="lines",
                           name=f"G2/FN {coupon:g}", line=dict(color=color, width=1.5)),
                row=r, col=1,
            )
        _hline(fig, 0, row=r, col=1)
        fig.update_yaxes(title_text="Cumulative relative Perf (32nds)", row=r, col=1)
    fig.update_layout(
        template="plotly_white", height=700,
        title_text=f"G2 Minus FN JPM Performance - Trailing {window} Observations",
    )
    return fig


def fig_g2_fn_ct10_corr(perf_data: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """4x2 grid of G2/FN perf vs CT10-change rolling correlations."""
    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True,
        subplot_titles=[f"G2/FN {c:g}" for c in G2_FN_PERF_COUPONS],
        vertical_spacing=0.06,
    )
    for i, coupon in enumerate(G2_FN_PERF_COUPONS):
        r, c = i // 2 + 1, i % 2 + 1
        row = summary.loc[summary["coupon"] == f"G2/FN {coupon:g}"].iloc[0]
        for w in G2_FN_CORR_WINDOWS:
            fig.add_trace(
                go.Scatter(
                    x=perf_data.index, y=perf_data[f"g2_fn_{coupon:g}_corr_{w}d"],
                    mode="lines", name=f"{w}d: {row[f'corr_{w}d']:+.2f}",
                    line=dict(color=CORR_COLORS[w], width=1.2),
                ),
                row=r, col=c,
            )
        _hline(fig, float(row["corr_all"]), color="#111827", row=r, col=c)
        _hline(fig, 0, dash="solid", color="#94a3b8", row=r, col=c)
        fig.update_yaxes(range=[-1.05, 1.05], row=r, col=c)
    fig.update_layout(
        template="plotly_white", height=800,
        title_text="G2 Minus FN Daily JPM Performance vs Daily CT10 Yield Change",
    )
    return fig


def fig_swap_recent_perf(swap_data: pd.DataFrame, wide: bool = False) -> go.Figure:
    """Trailing-window cumulative adjacent / wide coupon-swap JPM performance."""
    window = STRUCTURE_RECENT_PLOT_WINDOW
    pairs = WIDE_COUPON_SWAP_PAIRS if wide else COUPON_SWAP_PAIRS
    kind = "wide_swap" if wide else "swap"
    title = (
        f"100-bps-Wide Coupon Swap JPM Performance - Trailing {window} Observations"
        if wide else
        f"Coupon Swap JPM Performance - Trailing {window} Observations"
    )
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[
            f"{agency} {'100-bps-wide ' if wide else 'adjacent-'}coupon swaps"
            for agency in STRUCTURE_AGENCIES
        ],
    )
    for r, agency in enumerate(STRUCTURE_AGENCIES, start=1):
        for i, (high, low) in enumerate(pairs):
            key = f"{agency.lower()}_{high:g}_{low:g}_{kind}_perf_32nds"
            cumulative = swap_data[key].dropna().tail(window).cumsum()
            fig.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative, mode="lines",
                           name=f"{agency} {high:g}/{low:g}",
                           line=dict(color=TAB10[i % 10], width=1.4)),
                row=r, col=1,
            )
        _hline(fig, 0, row=r, col=1)
        fig.update_yaxes(title_text="Cumulative swap Perf (32nds)", row=r, col=1)
    fig.update_layout(template="plotly_white", height=700, title_text=title)
    return fig


def fig_fly_recent_price(fly_data: pd.DataFrame) -> go.Figure:
    """Trailing-window coupon-fly price levels rebased to zero at window start."""
    window = STRUCTURE_RECENT_PLOT_WINDOW
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"{agency} coupon flies" for agency in STRUCTURE_AGENCIES],
    )
    for r, agency in enumerate(STRUCTURE_AGENCIES, start=1):
        for i, center in enumerate(COUPON_FLY_CENTERS):
            key = f"{agency.lower()}_{center:g}_fly_level_32nds"
            recent = fly_data[key].dropna().tail(window)
            rebased = recent - recent.iloc[0]
            fig.add_trace(
                go.Scatter(x=rebased.index, y=rebased, mode="lines",
                           name=f"{agency} {center:g} fly",
                           line=dict(color=TAB10[i % 10], width=1.4)),
                row=r, col=1,
            )
        _hline(fig, 0, row=r, col=1)
        fig.update_yaxes(title_text="Fly price change (32nds, rebased)", row=r, col=1)
    fig.update_layout(
        template="plotly_white", height=700,
        title_text=f"Coupon Fly Price Changes - Trailing {window} Observations",
    )
    return fig


def fig_structure_corr_pages(
    structure_data: pd.DataFrame, summary: pd.DataFrame, title: str
) -> List[go.Figure]:
    """One rolling-correlation grid per agency for a structure family."""
    figures = []
    for agency in STRUCTURE_AGENCIES:
        sub = summary[summary["agency"] == agency].reset_index(drop=True)
        nrows = (len(sub) + 1) // 2
        fig = make_subplots(
            rows=nrows, cols=2, shared_xaxes=True,
            subplot_titles=list(sub["structure"]), vertical_spacing=0.09,
        )
        for i, row in sub.iterrows():
            r, c = i // 2 + 1, i % 2 + 1
            key = row["series_key"]
            for w in G2_FN_CORR_WINDOWS:
                fig.add_trace(
                    go.Scatter(
                        x=structure_data.index,
                        y=structure_data[f"{key}_corr_{w}d"],
                        mode="lines", name=f"{w}d {row[f'corr_{w}d']:+.2f}",
                        line=dict(color=CORR_COLORS[w], width=1.0),
                    ),
                    row=r, col=c,
                )
            _hline(fig, float(row["corr_all"]), color="#111827", row=r, col=c)
            _hline(fig, 0, dash="solid", color="#94a3b8", row=r, col=c)
            fig.update_yaxes(range=[-1.05, 1.05], title_text="Correlation", row=r, col=c)
        fig.update_layout(
            template="plotly_white", height=max(500, 260 * nrows),
            title_text=f"{agency} {title} vs Daily CT10 Yield Change",
        )
        figures.append(fig)
    return figures


def fig_structure_momentum(
    history: pd.DataFrame,
    specs: List[Tuple[str, str]],
    title: str,
    panels: Optional[List[Tuple[str, List[Tuple[str, str]]]]] = None,
) -> go.Figure:
    """Volatility-adjusted momentum score histories (60d solid, 20d dashed)."""
    short_w, medium_w = STRUCTURE_MOM_WINDOWS
    if panels is None:
        panels = [
            (agency, [s for s in specs if s[1].startswith(f"{agency} ")])
            for agency in STRUCTURE_AGENCIES
        ]
    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[
            f"{p}  (solid = {medium_w}d score, dashed = {short_w}d)" for p, _ in panels
        ],
    )
    for r, (panel_title, panel_specs) in enumerate(panels, start=1):
        for i, (key, label) in enumerate(panel_specs):
            medium = history.get(f"{key}_mom_{medium_w}d")
            short = history.get(f"{key}_mom_{short_w}d")
            if medium is None:
                continue
            color = TAB10[i % 10]
            short_label = label.replace(f"{panel_title} ", "")
            fig.add_trace(
                go.Scatter(x=medium.index, y=medium, mode="lines",
                           name=f"{short_label} {medium_w}d",
                           line=dict(color=color, width=1.6)),
                row=r, col=1,
            )
            fig.add_trace(
                go.Scatter(x=short.index, y=short, mode="lines",
                           name=f"{short_label} {short_w}d", showlegend=False,
                           line=dict(color=color, width=0.8, dash="dash"),
                           opacity=0.55),
                row=r, col=1,
            )
        _hline(fig, 0, dash="solid", color="#111827", row=r, col=1)
        _hline(fig, 1, color="#94a3b8", row=r, col=1)
        _hline(fig, -1, color="#94a3b8", row=r, col=1)
        fig.update_yaxes(title_text="Momentum score", row=r, col=1)
    fig.update_layout(
        template="plotly_white", height=max(500, 380 * len(panels)), title_text=title,
    )
    return fig


# ---------------------------------------------------------------------------
# Current-coupon basis Plotly figures
# ---------------------------------------------------------------------------
def fig_cc_fitted(df_basis: pd.DataFrame, response_col: str) -> go.Figure:
    """Full-sample actual / fitted OAS and residual for one CC index."""
    name = {"FNCC": "Conventional", "G2CC": "Ginnie"}[response_col]
    Y, _, model = run_current_coupon_ols(df_basis, response_col)
    residual = model.resid
    resid_std = float(np.std(residual))
    latest_resid = float(residual.iloc[-1])
    ratio = latest_resid / resid_std if resid_std != 0 else np.nan
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33],
        vertical_spacing=0.06,
        subplot_titles=[
            f"Sample: {Y.index[0].date()} to {Y.index[-1].date()}  |  "
            f"N = {int(model.nobs):,}  |  R-squared = {model.rsquared:.3f}  |  "
            f"Predictors: 10y, 2s10s, 1y10y_vol"
            f"{' (lagged 1d)' if CC_LAG_PREDICTORS else ''}",
            f"Latest actual {Y.iloc[-1]:.2f}  |  fitted {model.fittedvalues.iloc[-1]:.2f}  |  "
            f"residual {latest_resid:+.2f} bps (z = {ratio:.2f})",
        ],
    )
    fig.add_trace(go.Scatter(x=Y.index, y=Y, mode="lines", name="Actual OAS",
                             line=dict(color="firebrick", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Y.index, y=model.fittedvalues, mode="lines",
                             name="Full-sample fitted OAS",
                             line=dict(color="royalblue", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Y.index, y=residual, mode="lines",
                             name="Residual = actual - fitted",
                             line=dict(color="slateblue", width=1.0),
                             fill="tozeroy", fillcolor="rgba(106,90,205,0.15)"),
                  row=2, col=1)
    _hline(fig, 0, dash="solid", color="black", row=2, col=1)
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=650,
        title_text=f"{name} ({CC_TICKERS[name]}) - Current-Coupon Basis: "
                   "Full-Sample Actual, Fitted and Residual",
    )
    return fig


def fig_cc_rolling(rolling_df: pd.DataFrame, cc_type: str) -> go.Figure:
    """Rolling-window actual / fitted OAS and residual for one CC index."""
    window = CC_ROLLING_WINDOW
    sub = rolling_df[rolling_df["cc_type"] == cc_type].sort_index()
    latest = sub.iloc[-1]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33],
        vertical_spacing=0.06,
        subplot_titles=[
            f"Rolling estimates: {sub.index[0].date()} to {sub.index[-1].date()}  |  "
            f"Window = {window} observations",
            f"Latest actual {latest['mkt_val']:.2f}  |  fitted {latest['mdl_val']:.2f}  |  "
            f"residual {latest['resid']:+.2f} bps (z = {latest['resid_z']:.2f})",
        ],
    )
    fig.add_trace(go.Scatter(x=sub.index, y=sub["mkt_val"], mode="lines",
                             name="Actual OAS",
                             line=dict(color="firebrick", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["mdl_val"], mode="lines",
                             name=f"{window}-day rolling fitted OAS",
                             line=dict(color="royalblue", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["resid"], mode="lines",
                             name="Residual = actual - fitted",
                             line=dict(color="slateblue", width=1.0),
                             fill="tozeroy", fillcolor="rgba(106,90,205,0.15)"),
                  row=2, col=1)
    _hline(fig, 0, dash="solid", color="black", row=2, col=1)
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=650,
        title_text=f"{cc_type} - Current-Coupon Basis: {window}-Day Rolling "
                   "Actual, Fitted and Residual",
    )
    return fig


def fig_conv_ginnie_spread(df_basis: pd.DataFrame, window: int = 90) -> go.Figure:
    """Conventional minus Ginnie CC OAS spread with rolling z-score."""
    spread = df_basis["FNCC"] - df_basis["G2CC"]
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[
            f"Conv - Ginnie OAS (current diff {float(spread.iloc[-1]):.1f} bps)",
            f"{window}-day z-score (current {float(zscore.iloc[-1]):.2f})",
        ],
    )
    fig.add_trace(go.Scatter(x=spread.index, y=spread, mode="lines",
                             name="Conv - Ginnie OAS",
                             line=dict(color="navy", width=1.2)), row=1, col=1)
    _hline(fig, 0, dash="solid", color="black", row=1, col=1)
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode="lines", name="z-score",
                             line=dict(color="purple", width=1.2)), row=2, col=1)
    _hline(fig, 0, dash="solid", color="black", row=2, col=1)
    _hline(fig, 1, color="red", row=2, col=1)
    _hline(fig, -1, color="red", row=2, col=1)
    fig.update_yaxes(title_text="OAS spread (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=650,
        title_text=f"Conv vs Ginnie Current-Coupon OAS Spread ({window}-day z-score)",
    )
    return fig


def fig_cc_ma_vol(df_basis: pd.DataFrame) -> go.Figure:
    """CC OAS with MA overlays (top) and realized vol of daily changes (bottom)."""
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=[
            "Conventional (.FNCC105)", "Ginnie (.G2CC105)",
            "Conventional realized vol (daily)", "Ginnie realized vol (daily)",
        ],
    )
    for c, col in enumerate(("FNCC", "G2CC"), start=1):
        if col not in df_basis.columns:
            continue
        s = df_basis[col]
        fig.add_trace(
            go.Scatter(x=s.index, y=s, mode="lines",
                       name=f"{col} OAS: {float(s.dropna().iloc[-1]):.1f} bps",
                       line=dict(color="navy", width=1.2)),
            row=1, col=c,
        )
        for w in CC_MA_WINDOWS:
            ma = s.rolling(window=w, min_periods=w).mean()
            fig.add_trace(
                go.Scatter(x=s.index, y=ma, mode="lines",
                           name=f"MA{w}: {float(ma.dropna().iloc[-1]):.1f} bps",
                           line=dict(color=MA_COLORS[w], width=1.1)),
                row=1, col=c,
            )
        d = s.diff()
        for w in CC_VOL_WINDOWS:
            vol = d.rolling(window=w, min_periods=w).std() * np.sqrt(CC_VOL_ANN_FACTOR)
            fig.add_trace(
                go.Scatter(x=d.index, y=vol, mode="lines",
                           name=f"{col} {w}d: {float(vol.dropna().iloc[-1]):.2f} bps/day",
                           line=dict(color=VOL_COLORS[w], width=1.1)),
                row=2, col=c,
            )
        fig.update_yaxes(title_text="OAS (bps)", row=1, col=c)
        fig.update_yaxes(title_text="Vol (bps/day)", row=2, col=c)
    fig.update_layout(
        template="plotly_white", height=700,
        title_text="Current-Coupon Basis: Moving Averages and Realized Vol",
    )
    return fig


def fig_cc_momentum(history: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """CC levels with EMA20/60 (top) and vol-adjusted momentum scores (bottom)."""
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=[
            f"Conventional (.FNCC105) - trend: {summary.loc['Conventional', 'trend']} "
            f"({str(summary.loc['Conventional', 'strength']).lower()})",
            f"Ginnie (.G2CC105) - trend: {summary.loc['Ginnie', 'trend']} "
            f"({str(summary.loc['Ginnie', 'strength']).lower()})",
            "Volatility-adjusted rolling trend", "Volatility-adjusted rolling trend",
        ],
    )
    for c, col in enumerate(("FNCC", "G2CC"), start=1):
        level_col = f"{col}_level"
        if level_col not in history.columns:
            continue
        level = history[level_col]
        fig.add_trace(
            go.Scatter(x=level.index, y=level, mode="lines",
                       name=f"{col} OAS: {float(level.dropna().iloc[-1]):.1f} bps",
                       line=dict(color="#111827", width=1.2)),
            row=1, col=c,
        )
        for w in CC_MOM_WINDOWS:
            ema = level.ewm(span=w, adjust=False, min_periods=w).mean()
            fig.add_trace(
                go.Scatter(x=ema.index, y=ema, mode="lines",
                           name=f"{col} EMA{w}: {float(ema.dropna().iloc[-1]):.1f} bps",
                           line=dict(color=MOM_COLORS[w], width=1.1)),
                row=1, col=c,
            )
            score = history[f"{col}_mom_{w}d"]
            fig.add_trace(
                go.Scatter(x=score.index, y=score, mode="lines",
                           name=f"{col} {w} obs: {float(score.dropna().iloc[-1]):+.2f}",
                           line=dict(color=MOM_COLORS[w], width=1.3)),
                row=2, col=c,
            )
        _hline(fig, 0, dash="solid", color="#111827", row=2, col=c)
        _hline(fig, 1, color="#94a3b8", row=2, col=c)
        _hline(fig, -1, color="#94a3b8", row=2, col=c)
        fig.update_yaxes(title_text="OAS (bps)", row=1, col=c)
        fig.update_yaxes(title_text="Momentum score", row=2, col=c)
    fig.update_layout(
        template="plotly_white", height=700,
        title_text="Current-Coupon Basis Momentum Monitor "
                   "(positive = widening, negative = tightening)",
    )
    return fig


def fig_fn_10y_rolling_corr(corr_df: pd.DataFrame, all_sample: dict) -> go.Figure:
    """Full-history, recent, and z-scored FN/10y rolling correlations."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.07,
        subplot_titles=[
            "Full available aligned sample",
            "Most recent 250 observations",
            f"Rolling-correlation z-score over {CC_FN_10Y_CORR_Z_WINDOW} observations",
        ],
    )
    for w in CC_FN_10Y_CORR_WINDOWS:
        column = f"corr_{w}d"
        z_col = f"corr_{w}d_z{CC_FN_10Y_CORR_Z_WINDOW}"
        color = FN_10Y_COLORS[w]
        fig.add_trace(go.Scatter(x=corr_df.index, y=corr_df[column], mode="lines",
                                 name=f"{w} obs", line=dict(color=color, width=1.3)),
                      row=1, col=1)
        recent = corr_df[column].tail(250)
        fig.add_trace(go.Scatter(x=recent.index, y=recent, mode="lines",
                                 name=f"{w} obs", showlegend=False,
                                 line=dict(color=color, width=1.3)), row=2, col=1)
        recent_z = corr_df[z_col].dropna().tail(250)
        fig.add_trace(go.Scatter(x=recent_z.index, y=recent_z, mode="lines",
                                 name=f"{w} obs", showlegend=False,
                                 line=dict(color=color, width=1.3)), row=3, col=1)
    for r in (1, 2):
        _hline(fig, 0, dash="solid", color="#111827", row=r, col=1)
        fig.update_yaxes(range=[-1, 1], title_text="Correlation", row=r, col=1)
    _hline(fig, 0, dash="solid", color="#111827", row=3, col=1)
    _hline(fig, 2, color="#b91c1c", row=3, col=1)
    _hline(fig, -2, color="#b91c1c", row=3, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    latest_text = "  |  ".join(
        f"{w} obs: {corr_df[f'corr_{w}d'].dropna().iloc[-1]:+.2f} "
        f"(z {corr_df[f'corr_{w}d_z{CC_FN_10Y_CORR_Z_WINDOW}'].dropna().iloc[-1]:+.2f})"
        for w in CC_FN_10Y_CORR_WINDOWS
    )
    fig.update_layout(
        template="plotly_white", height=750,
        title_text=(
            "FN Current-Coupon Basis vs UST 10y: Rolling Correlation of Daily Changes<br>"
            f"<sup>{latest_text}  |  All sample: daily changes "
            f"{all_sample['change_corr']:+.2f} ({all_sample['change_obs']} obs), levels "
            f"{all_sample['level_corr']:+.2f} ({all_sample['level_obs']} obs)</sup>"
        ),
    )
    return fig


def fig_coupon_basis_fitted(cb_basis: pd.DataFrame) -> List[go.Figure]:
    """Per-agency 2x4 grids of actual vs fitted per-coupon yield basis."""
    response_cols = [c for c in cb_basis.columns if c not in COUPON_BASIS_PREDICTORS]
    agency_groups: Dict[str, List[str]] = {}
    for col in response_cols:
        agency_groups.setdefault(col.split()[0], []).append(col)

    figures = []
    for agency, cols in agency_groups.items():
        cols_sorted = sorted(cols, key=lambda x: float(x.split()[1]))
        nrows = int(np.ceil(len(cols_sorted) / 4))
        fig = make_subplots(
            rows=nrows, cols=4, subplot_titles=cols_sorted, vertical_spacing=0.12,
        )
        for i, col in enumerate(cols_sorted):
            r, c = i // 4 + 1, i % 4 + 1
            Y, _, model = run_coupon_basis_ols(cb_basis, col)
            resid_std = float(np.std(model.resid))
            diff = float(Y.iloc[-1] - model.fittedvalues.iloc[-1])
            z = diff / resid_std if resid_std else np.nan
            fig.add_trace(
                go.Scatter(x=Y.index, y=Y, mode="lines", name=f"{col} actual",
                           showlegend=False, line=dict(color="red", width=1.0)),
                row=r, col=c,
            )
            fig.add_trace(
                go.Scatter(x=Y.index, y=model.fittedvalues, mode="lines",
                           name=f"{col} fitted (z={z:.2f})", showlegend=False,
                           line=dict(color="blue", width=1.0)),
                row=r, col=c,
            )
            fig.update_yaxes(title_text="Basis (bps)", row=r, col=c)
        fig.update_layout(
            template="plotly_white", height=max(500, 340 * nrows),
            title_text=f"Per-Coupon Yield Basis OLS: {agency} MBS (actual red, fitted blue)",
        )
        figures.append(fig)
    return figures


# ---------------------------------------------------------------------------
# Appendix Plotly figures
# ---------------------------------------------------------------------------
def _appendix_grid(
    series_by_coupon: List[Tuple[float, pd.Series, str]],
    agency: str,
    color: str,
    title: str,
    y_label: str,
) -> go.Figure:
    """4x2 grid of full-history series with a latest-value marker."""
    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[t for _, _, t in series_by_coupon],
    )
    for i, (_, series, _) in enumerate(series_by_coupon):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Scatter(x=series.index, y=series, mode="lines", showlegend=False,
                       line=dict(color=color, width=1.2)),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(x=[series.index[-1]], y=[series.iloc[-1]], mode="markers",
                       showlegend=False, marker=dict(color="#111827", size=6)),
            row=r, col=c,
        )
        _hline(fig, 0, color="#64748b", row=r, col=c)
        fig.update_yaxes(title_text=y_label, row=r, col=c)
    fig.update_layout(template="plotly_white", height=800, title_text=title)
    return fig


def figs_coupon_drops(dat: pd.DataFrame) -> List[go.Figure]:
    """Full-history drop series per agency with Z120 in the subplot titles."""
    figures = []
    for agency in STRUCTURE_AGENCIES:
        entries = []
        for coupon in COUPONS:
            column = _col(agency, coupon, "Drop")
            if column not in dat.columns:
                continue
            series = pd.to_numeric(dat[column], errors="coerce").dropna()
            if series.empty:
                continue
            rolling_mean = series.rolling(
                COUPON_DROP_Z_WINDOW, min_periods=COUPON_DROP_Z_WINDOW
            ).mean()
            rolling_std = series.rolling(
                COUPON_DROP_Z_WINDOW, min_periods=COUPON_DROP_Z_WINDOW
            ).std(ddof=1)
            rolling_z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            latest_z = float(rolling_z.dropna().iloc[-1])
            title = (f"{agency} {coupon:g}  |  Current: {float(series.iloc[-1]):.2f}  |  "
                     f"Z120: {latest_z:+.2f}")
            entries.append((coupon, series, title))
        if not entries:
            continue
        figures.append(_appendix_grid(
            entries, agency, DROP_COLORS[agency],
            f"{AGENCY_NAMES[agency]} Coupon Drops - Full Available History",
            "Drop (32nds)",
        ))
    return figures


def figs_coupon_drop_sharpes(dat: pd.DataFrame) -> List[go.Figure]:
    """Monthly drop / monthly-equivalent trailing JPM Perf volatility."""
    figures = []
    for agency in STRUCTURE_AGENCIES:
        entries = []
        for coupon in COUPONS:
            drop_col = _col(agency, coupon, "Drop")
            perf_col = _col(agency, coupon, "Perf")
            if drop_col not in dat.columns or perf_col not in dat.columns:
                continue
            drop = pd.to_numeric(dat[drop_col], errors="coerce")
            perf = pd.to_numeric(dat[perf_col], errors="coerce")
            vol = perf.rolling(
                COUPON_SHARPE_VOL_WINDOW, min_periods=COUPON_SHARPE_VOL_WINDOW
            ).std(ddof=1)
            monthly_vol = vol * np.sqrt(COUPON_SHARPE_MONTH_DAYS)
            sharpe = (drop / monthly_vol.replace(0, np.nan)).dropna()
            if sharpe.empty:
                continue
            title = f"{agency} {coupon:g}  |  Monthly Sharpe60: {float(sharpe.iloc[-1]):+.2f}"
            entries.append((coupon, sharpe, title))
        if not entries:
            continue
        figures.append(_appendix_grid(
            entries, agency, SHARPE_COLORS[agency],
            f"{AGENCY_NAMES[agency]} Coupon Monthly Drop Sharpe - Full Available History",
            "Drop / monthly-equivalent vol",
        ))
    return figures


def figs_coupon_forward_vol_sharpes(dat: pd.DataFrame) -> List[go.Figure]:
    """Drop / monthly-equivalent 1y10y ATM normal volatility."""
    annualized_vol = pd.to_numeric(dat[COUPON_FORWARD_VOL_COL], errors="coerce")
    monthly_vol = annualized_vol / np.sqrt(COUPON_FORWARD_VOL_MONTHS)
    figures = []
    for agency in STRUCTURE_AGENCIES:
        entries = []
        for coupon in COUPONS:
            drop_col = _col(agency, coupon, "Drop")
            if drop_col not in dat.columns:
                continue
            drop = pd.to_numeric(dat[drop_col], errors="coerce")
            ratio = (drop / monthly_vol.replace(0, np.nan)).dropna()
            if ratio.empty:
                continue
            title = f"{agency} {coupon:g}  |  Forward-vol proxy: {float(ratio.iloc[-1]):+.3f}"
            entries.append((coupon, ratio, title))
        if not entries:
            continue
        figures.append(_appendix_grid(
            entries, agency, FWDVOL_COLORS[agency],
            f"{AGENCY_NAMES[agency]} Coupon 1y10y Forward-Vol Sharpe Proxy",
            "Drop / monthly 1y10y vol",
        ))
    return figures


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate_html_report(dat: pd.DataFrame, df_basis: Optional[pd.DataFrame] = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = date.today().strftime("%Y%m%d")
    html_path = OUTPUT_DIR / f"MBS_RV_Report_Plotly_{report_date}.html"
    if html_path.exists():
        try:
            with open(html_path, "ab"):
                pass
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = OUTPUT_DIR / f"MBS_RV_Report_Plotly_{report_date}_{ts}.html"

    latest = dat.index[-1].date()
    sections: List[str] = []
    score_gradient = [f"scores_t{i}" for i in range(len(DEFAULT_SCORE_INDICES))]
    mom_gradient = ["chg_20d", "chg_60d", "mom_20d", "mom_60d"]
    struct_gradient = [
        "latest", "perf_5d", "perf_20d", "perf_40d",
        "corr_all", "corr_20d", "corr_40d", "corr_60d",
    ]

    # 1. Latest MBS prices
    sections.append(
        "<section><h2>Latest MBS Prices</h2>"
        f'<p class="subtle">Latest: {latest}  |  Prices shown as decimal and handle-32nds</p>'
        + df_to_html_table(build_latest_price_table(dat), fmt="{:.4f}")
        + "</section>"
    )

    # 2. Coupon swap / fly / G2-FN score tables
    cpn_fly_summary = build_cpn_fly_summary_table(dat)
    cpn_fly_summary = add_ols_metrics_to_summary(dat, cpn_fly_summary)
    cpn_fly_summary = add_range_metrics_to_summary(dat, cpn_fly_summary)
    time_cols = [
        f"{prefix}_t{i}" for i in range(len(DEFAULT_SCORE_INDICES))
        for prefix in ("vals", "scores")
    ]
    cpn_fly_summary = cpn_fly_summary[
        time_cols
        + ["mdl_val", "resid_std", "r2", "resid_z"]
        + ["mkt_val", "carry", "hist_min", "hist_max", "range_ratio"]
        + ["coef_price", "coef_slope", "coef_carry", "coef_vol"]
    ]
    date_labels = "  |  ".join(
        f"t{i} = {dat.index[idx].date()}" for i, idx in enumerate(DEFAULT_SCORE_INDICES)
    )
    parts = []
    for group_name, group_df in _group_cpn_fly_rows(cpn_fly_summary).items():
        parts.append(f"<h3>{group_name}</h3>")
        parts.append(df_to_html_table(
            group_df,
            gradient_cols=score_gradient + ["resid_z", "range_ratio"],
            fmt="{:.2f}",
        ))
    sections.append(
        "<section><h2>Coupon Swap / Fly / G2-FN Scores</h2>"
        f'<p class="subtle">{date_labels}  |  z-score lookback: 40 observations</p>'
        + "".join(parts)
        + "</section>"
    )

    # 3. Yield basis score tables
    basis_score_indices = [-1, -5, -10, -20, -30, -40, -50]
    basis_summary = build_yield_basis_summary_table(dat, score_indices=basis_score_indices)
    basis_labels = "  |  ".join(
        f"t{i} = {dat.index[idx].date()}" for i, idx in enumerate(basis_score_indices)
    )
    basis_score_cols = [f"scores_t{i}" for i in range(len(basis_score_indices))]
    parts = []
    for group_name, group_df in _group_yield_basis_rows(basis_summary).items():
        parts.append(f"<h3>{group_name}</h3>")
        parts.append(df_to_html_table(group_df, gradient_cols=basis_score_cols, fmt="{:.2f}"))
    sections.append(
        "<section><h2>Yield Basis Scores (vs CT10)</h2>"
        f'<p class="subtle">{basis_labels}  |  z-score lookback: 60 observations</p>'
        + "".join(parts)
        + "</section>"
    )

    # 4. Performance over periods
    perf_summary = build_perf_summary_table(dat)
    perf_cols = [f"{abs(lb)}d" for lb in DEFAULT_PERF_WINDOWS]
    sections.append(
        "<section><h2>Performance Over Periods</h2>"
        f'<p class="subtle">Cumulative JPM DataQuery performance (32nds) over trailing '
        f"windows  |  Latest: {latest}</p>"
        + df_to_html_table(perf_summary, gradient_cols=perf_cols, fmt="{:.2f}")
        + "</section>"
    )

    # 5. Citi MBS performance vs 7Y UST
    try:
        citi_perf_calc = combine_citi_perf_history(
            load_citi_perf_data(CITI_FNAME), load_citi_perf_intraday(CITI_FNAME)
        )
        citi_perf_summary = build_citi_perf_summary(citi_perf_calc)
        if not citi_perf_summary.empty:
            eff_date = citi_perf_calc.index[-1].date()
            sections.append(
                "<section><h2>Citi MBS Performance vs 7Y UST</h2>"
                f'<p class="subtle">Latest: {eff_date}  |  '
                "Performance vs 7Y UST in ticks (x32)</p>"
                + df_to_html_table(
                    citi_perf_summary,
                    gradient_cols=["latest_day", "last_5d", "last_30d", "mtd", "ytd"],
                    badge_cols=("signal",),
                    fmt="{:.2f}",
                )
                + "</section>"
            )
    except Exception as exc:
        print(f"Citi performance section skipped: {exc}")

    # 6. Rolling performance z-scores
    rolling_perf_z = build_rolling_perf_zscore_table(dat)
    sections.append(
        f"<section><h2>Rolling Performance Z-Scores "
        f"({ROLL_SUM_WINDOW}d sum / {ROLL_SUM_Z_WINDOW}d z-score)</h2>"
        f'<p class="subtle">Performance = JPM DataQuery daily performance (32nds)  |  '
        f"Latest: {latest}</p>"
        + df_to_html_table(
            rolling_perf_z,
            gradient_cols=["latest_perf", f"roll_sum_{ROLL_SUM_WINDOW}d",
                           f"z_score_{ROLL_SUM_Z_WINDOW}d"],
            fmt="{:.2f}",
        )
        + "</section>"
    )

    # 7. G2-minus-FN relative JPM performance
    g2_fn_perf_data = build_g2_fn_jpm_perf_timeseries(dat)
    g2_fn_perf_summary = build_g2_fn_jpm_perf_summary(g2_fn_perf_data)
    sections.append(
        "<section><h2>G2 Minus FN JPM Performance: 3.0 to 6.5 Coupons</h2>"
        '<p class="subtle">Daily relative Perf = G2 JPM Perf - FN JPM Perf (32nds); '
        "5d/20d/40d are cumulative sums  |  Correlation uses daily CT10 yield changes</p>"
        + df_to_html_table(g2_fn_perf_summary, gradient_cols=struct_gradient, fmt="{:.2f}")
        + figs_to_html([
            fig_g2_fn_recent_perf(g2_fn_perf_data),
            fig_g2_fn_ct10_corr(g2_fn_perf_data, g2_fn_perf_summary),
        ])
        + "</section>"
    )

    # 8. G2/FN relative performance momentum monitor
    g2_fn_specs = _g2_fn_specs()
    g2_fn_mom_history, g2_fn_mom = build_structure_momentum(
        g2_fn_perf_data, g2_fn_specs, cumulative=True
    )
    sections.append(
        "<section><h2>G2/FN Relative Performance Momentum Monitor</h2>"
        '<p class="subtle">Cumulative G2-minus-FN Perf trend; score = rolling OLS slope x '
        "sqrt(window) / daily realized vol  |  Upward = G2 outperforming FN; "
        "Downward = FN outperforming G2</p>"
        + df_to_html_table(g2_fn_mom.drop(columns=["latest"]),
                           gradient_cols=mom_gradient, fmt="{:.2f}")
        + figs_to_html([fig_structure_momentum(
            g2_fn_mom_history,
            g2_fn_specs,
            "G2/FN Relative Performance Momentum Monitor",
            panels=[
                ("G2/FN 3.0 to 4.5", g2_fn_specs[:4]),
                ("G2/FN 5.0 to 6.5", g2_fn_specs[4:]),
            ],
        )])
        + "</section>"
    )

    # 9. Adjacent-coupon swaps using JPM daily performance
    coupon_swap_data = build_coupon_swap_jpm_perf_timeseries(dat)
    coupon_swap_summary = build_coupon_swap_jpm_perf_summary(coupon_swap_data)
    sections.append(
        "<section><h2>Adjacent-Coupon Swap JPM Performance</h2>"
        '<p class="subtle">Daily swap Perf = higher-coupon JPM Perf - lower-coupon '
        "JPM Perf (32nds); 5d/20d/40d are cumulative sums  |  "
        "Correlation uses daily CT10 yield changes</p>"
        + df_to_html_table(coupon_swap_summary.drop(columns=["series_key"]),
                           gradient_cols=struct_gradient, fmt="{:.2f}")
        + figs_to_html(
            [fig_swap_recent_perf(coupon_swap_data)]
            + fig_structure_corr_pages(
                coupon_swap_data, coupon_swap_summary, "Adjacent-Coupon Swap JPM Perf"
            )
        )
        + "</section>"
    )

    # 10. Adjacent-coupon swap momentum monitor
    adj_swap_specs = _adjacent_swap_specs()
    adj_swap_mom_history, adj_swap_mom = build_structure_momentum(
        coupon_swap_data, adj_swap_specs, cumulative=True
    )
    sections.append(
        "<section><h2>Adjacent-Coupon Swap Momentum Monitor</h2>"
        '<p class="subtle">Cumulative swap Perf trend  |  Upward = higher coupon '
        "outperforming; Downward = lower coupon outperforming</p>"
        + df_to_html_table(adj_swap_mom.drop(columns=["latest"]),
                           gradient_cols=mom_gradient, fmt="{:.2f}")
        + figs_to_html([fig_structure_momentum(
            adj_swap_mom_history, adj_swap_specs, "Adjacent-Coupon Swap Momentum Monitor"
        )])
        + "</section>"
    )

    # 11. 100-bps-wide coupon swaps
    wide_swap_data = build_wide_coupon_swap_jpm_perf_timeseries(dat)
    wide_swap_summary = build_wide_coupon_swap_jpm_perf_summary(wide_swap_data)
    sections.append(
        "<section><h2>100-bps-Wide Coupon Swap JPM Performance</h2>"
        '<p class="subtle">Pairs span 100 bps; 5d/20d/40d are cumulative sums  |  '
        "Correlation uses daily CT10 yield changes</p>"
        + df_to_html_table(wide_swap_summary.drop(columns=["series_key"]),
                           gradient_cols=struct_gradient, fmt="{:.2f}")
        + figs_to_html(
            [fig_swap_recent_perf(wide_swap_data, wide=True)]
            + fig_structure_corr_pages(
                wide_swap_data, wide_swap_summary, "100-bps-Wide Coupon Swap JPM Perf"
            )
        )
        + "</section>"
    )

    # 12. 100-bps-wide coupon swap momentum monitor
    wide_swap_specs = _wide_swap_specs()
    wide_swap_mom_history, wide_swap_mom = build_structure_momentum(
        wide_swap_data, wide_swap_specs, cumulative=True
    )
    sections.append(
        "<section><h2>100-bps-Wide Coupon Swap Momentum Monitor</h2>"
        '<p class="subtle">Cumulative 100-bps-wide swap Perf trend  |  '
        "Upward = higher coupon outperforming; Downward = lower coupon outperforming</p>"
        + df_to_html_table(wide_swap_mom.drop(columns=["latest"]),
                           gradient_cols=mom_gradient, fmt="{:.2f}")
        + figs_to_html([fig_structure_momentum(
            wide_swap_mom_history, wide_swap_specs,
            "100-bps-Wide Coupon Swap Momentum Monitor",
        )])
        + "</section>"
    )

    # 13. Three-price coupon flies
    coupon_fly_data = build_coupon_fly_price_timeseries(dat)
    coupon_fly_summary = build_coupon_fly_price_summary(coupon_fly_data)
    fly_gradient = [
        "level_latest", "chg_5d", "chg_20d", "chg_40d",
        "corr_all", "corr_20d", "corr_40d", "corr_60d",
    ]
    sections.append(
        "<section><h2>Three-Price Coupon Fly Performance</h2>"
        '<p class="subtle">Fly level = 2 x center price - lower price - upper price; '
        "values and changes in 32nds; correlation uses daily fly-price changes vs "
        "daily CT10 yield changes</p>"
        + df_to_html_table(coupon_fly_summary.drop(columns=["series_key"]),
                           gradient_cols=fly_gradient, fmt="{:.2f}")
        + figs_to_html(
            [fig_fly_recent_price(coupon_fly_data)]
            + fig_structure_corr_pages(
                coupon_fly_data, coupon_fly_summary, "Three-Price Coupon Fly Changes"
            )
        )
        + "</section>"
    )

    # 14. Three-price coupon fly momentum monitor
    fly_specs = _fly_specs()
    fly_mom_history, fly_mom = build_structure_momentum(coupon_fly_data, fly_specs)
    sections.append(
        "<section><h2>Three-Price Coupon Fly Momentum Monitor</h2>"
        '<p class="subtle">Fly level = 2 x center - lower - upper (32nds)  |  '
        "Upward = fly level rising; Downward = fly level falling</p>"
        + df_to_html_table(fly_mom.rename(columns={"latest": "level_latest"}),
                           gradient_cols=["level_latest"] + mom_gradient, fmt="{:.2f}")
        + figs_to_html([fig_structure_momentum(
            fly_mom_history, fly_specs, "Three-Price Coupon Fly Momentum Monitor"
        )])
        + "</section>"
    )

    # 15-21. Current-coupon basis sections (need Bloomberg)
    if df_basis is not None:
        cc_subtitle = (
            f"Full sample: {df_basis.index[0].date()} to {df_basis.index[-1].date()}  |  "
            f"Obs: {len(df_basis)}  |  Predictors: 10y, 2s10s, 1y10y_vol"
            f"{' (lagged 1d)' if CC_LAG_PREDICTORS else ''}"
        )
        cc_summary = build_current_coupon_summary(df_basis)
        cc_params = build_current_coupon_params_table(df_basis)
        sections.append(
            "<section><h2>Current-Coupon Basis OLS Summary</h2>"
            f'<p class="subtle">{cc_subtitle}</p>'
            + df_to_html_table(cc_summary, gradient_cols=["diff", "resid_z"], fmt="{:.3f}")
            + "<h3>Current-Coupon Basis OLS Parameters</h3>"
            + df_to_html_table(cc_params, gradient_cols=["coef", "p_value"], fmt="{:.4f}")
            + "</section>"
        )
        sections.append(
            "<section><h2>Current-Coupon Basis: Full-Sample Actual / Fitted / Residual</h2>"
            + figs_to_html([
                fig_cc_fitted(df_basis, col)
                for col in ("FNCC", "G2CC") if col in df_basis.columns
            ])
            + "</section>"
        )
        print(f"Running rolling current-coupon basis OLS ({CC_ROLLING_WINDOW}d window)...")
        cc_rolling = run_rolling_current_coupon_basis(df_basis, window=CC_ROLLING_WINDOW)
        sections.append(
            f"<section><h2>Current-Coupon Basis: {CC_ROLLING_WINDOW}-Day Rolling "
            "Actual / Fitted / Residual</h2>"
            + figs_to_html([
                fig_cc_rolling(cc_rolling, cc_type)
                for cc_type in ("Conventional", "Ginnie")
            ])
            + "</section>"
        )
        sections.append(
            "<section><h2>Conv vs Ginnie Current-Coupon OAS Spread</h2>"
            + figs_to_html([fig_conv_ginnie_spread(df_basis, window=90)])
            + "</section>"
        )
        cc_ma_vol = build_cc_ma_vol_summary(df_basis)
        sections.append(
            "<section><h2>Current-Coupon Basis: Moving Averages &amp; Realized Vol</h2>"
            f'<p class="subtle">Latest: {df_basis.index[-1].date()}  |  '
            "d_ma = latest - MA  |  realized vol = std(daily change), bps/day</p>"
            + df_to_html_table(
                cc_ma_vol,
                gradient_cols=[f"vol_{w}d" for w in CC_VOL_WINDOWS]
                + [f"d_ma_{w}d" for w in CC_MA_WINDOWS],
                fmt="{:.2f}",
            )
            + figs_to_html([fig_cc_ma_vol(df_basis)])
            + "</section>"
        )
        cc_mom_history, cc_mom_summary = build_cc_momentum(df_basis)
        cc_mom_gradient = [
            "chg_5d", "chg_20d", "chg_60d",
            "slope_20d", "slope_60d", "mom_20d", "mom_60d",
        ]
        sections.append(
            "<section><h2>Current-Coupon Basis Momentum Monitor</h2>"
            f'<p class="subtle">Latest: {df_basis.index[-1].date()}  |  '
            "Positive = widening, negative = tightening</p>"
            + df_to_html_table(cc_mom_summary, gradient_cols=cc_mom_gradient, fmt="{:.2f}")
            + figs_to_html([fig_cc_momentum(cc_mom_history, cc_mom_summary)])
            + "</section>"
        )
        fn_10y_corr = build_fn_10y_rolling_corr(df_basis)
        fn_10y_all_sample = build_fn_10y_all_sample_summary(df_basis)
        fn_10y_corr_summary = build_fn_10y_corr_summary(fn_10y_corr)
        sections.append(
            "<section><h2>FN Basis vs UST 10y: Rolling Correlation Monitor</h2>"
            f'<p class="subtle">Daily changes  |  Latest: {fn_10y_corr.index[-1].date()}  |  '
            f"All sample: changes {fn_10y_all_sample['change_corr']:+.3f} "
            f"(n={fn_10y_all_sample['change_obs']}), levels "
            f"{fn_10y_all_sample['level_corr']:+.3f} "
            f"(n={fn_10y_all_sample['level_obs']})  |  "
            f"Corr z-score lookback: {CC_FN_10Y_CORR_Z_WINDOW} observations</p>"
            + df_to_html_table(fn_10y_corr_summary, fmt="{:.3f}")
            + figs_to_html([fig_fn_10y_rolling_corr(fn_10y_corr, fn_10y_all_sample)])
            + "</section>"
        )

    # 22. FN/G2 TSY OAS vs 1y history z-score
    try:
        oas_df = load_tsy_oas_data(FNAME)
        oas_summary = build_tsy_oas_summary(oas_df, lookback=TSY_OAS_LOOKBACK)
        if not oas_summary.empty:
            sections.append(
                "<section><h2>FN / G2 TSY OAS vs 1-Year History Z-Score</h2>"
                f'<p class="subtle">Latest: {oas_df.index[-1].date()}  |  '
                f"1y lookback: {TSY_OAS_LOOKBACK} trading days</p>"
                + df_to_html_table(oas_summary.reset_index(),
                                   gradient_cols=["z_score"], fmt="{:.2f}")
                + "</section>"
            )
    except Exception as exc:
        print(f"TSY OAS z-score section skipped: {exc}")

    # 23. Citi FN / G2 OAS vs 90-day history z-score
    try:
        citi_oas_df = load_citi_oas_data(CITI_FNAME)
        citi_oas_summary = build_citi_oas_summary(citi_oas_df, lookback=90)
        if not citi_oas_summary.empty:
            sections.append(
                "<section><h2>Citi FN / G2 OAS vs 90-Day History Z-Score</h2>"
                f'<p class="subtle">Latest: {citi_oas_df.index[-1].date()}  |  '
                "90-day lookback</p>"
                + df_to_html_table(citi_oas_summary.reset_index(),
                                   gradient_cols=["z_score_90d"], fmt="{:.2f}")
                + "</section>"
            )
    except Exception as exc:
        print(f"Citi OAS section skipped: {exc}")

    # 24. Per-coupon yield basis OLS
    try:
        predictors = load_ust_yield_vol_data(FNAME)
        cb_basis = build_coupon_basis_df(dat, predictors)
        if not cb_basis.empty:
            cb_subtitle = (
                f"Latest: {cb_basis.index[-1].date()}  |  Obs: {len(cb_basis)}  |  "
                f"Predictors: 10y, 2s10s, 1y10y_vol"
                f"{' (lagged 1d)' if COUPON_BASIS_LAG_PREDICTORS else ''}"
            )
            parts = [
                df_to_html_table(build_coupon_basis_summary(cb_basis),
                                 gradient_cols=["diff", "resid_z"], fmt="{:.3f}"),
                "<h3>Per-Coupon Yield Basis OLS Parameters</h3>",
                df_to_html_table(build_coupon_basis_params_table(cb_basis),
                                 gradient_cols=["coef", "p_value"], fmt="{:.4f}"),
            ]
            cb_coef_matrix = build_coupon_basis_coef_matrix(cb_basis)
            if not cb_coef_matrix.empty:
                parts.append("<h3>Per-Coupon Yield Basis Coefficients</h3>")
                for feat in COUPON_BASIS_PREDICTORS:
                    feat_cols = [c for c in cb_coef_matrix.columns
                                 if c.endswith(f" {feat}")]
                    if not feat_cols:
                        continue
                    parts.append(f"<h3>{feat}</h3>")
                    parts.append(df_to_html_table(cb_coef_matrix[feat_cols],
                                                  gradient_cols=feat_cols, fmt="{:.3f}"))
            sections.append(
                "<section><h2>Per-Coupon Yield Basis OLS Summary</h2>"
                f'<p class="subtle">{cb_subtitle}</p>'
                + "".join(parts)
                + "</section>"
            )
            sections.append(
                "<section><h2>Per-Coupon Yield Basis: Fitted vs Actual</h2>"
                + figs_to_html(fig_coupon_basis_fitted(cb_basis))
                + "</section>"
            )
    except Exception as exc:
        print(f"Per-coupon yield basis section skipped: {exc}")

    # 25-27. Appendix charts
    sections.append(
        "<section><h2>Appendix: Coupon Drops (Full History)</h2>"
        f'<p class="subtle">Z120 = trailing {COUPON_DROP_Z_WINDOW}-observation '
        "z-score  |  Black marker = current value</p>"
        + figs_to_html(figs_coupon_drops(dat))
        + "</section>"
    )
    sections.append(
        "<section><h2>Appendix: Coupon Monthly Drop Sharpe</h2>"
        f'<p class="subtle">Monthly Sharpe60 = drop / (trailing '
        f"{COUPON_SHARPE_VOL_WINDOW}-observation std of daily Perf x "
        f"sqrt({COUPON_SHARPE_MONTH_DAYS})); Perf in 32nds</p>"
        + figs_to_html(figs_coupon_drop_sharpes(dat))
        + "</section>"
    )
    sections.append(
        "<section><h2>Appendix: Coupon 1y10y Forward-Vol Sharpe Proxy</h2>"
        f'<p class="subtle">Proxy = drop (32nds) / [annualized 1y10y normal vol / '
        f"sqrt({COUPON_FORWARD_VOL_MONTHS})]; units: 32nds per bp</p>"
        + figs_to_html(figs_coupon_forward_vol_sharpes(dat))
        + "</section>"
    )

    subtitle = (
        f"Sample: {dat.index[0].date()} to {latest} ({len(dat)} observations)  |  "
        f"Latest data: {latest}  |  Generated: {datetime.now():%Y-%m-%d %H:%M}"
    )
    page = page_template("MBS RV Report", subtitle, "".join(sections))
    html_path.write_text(page, encoding="utf-8")
    return html_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    perf_audit_path = OUTPUT_DIR / "coupon_perf_vs_tsy_implied.csv"
    dat = load_mbs_rv(FNAME, perf_audit_path=perf_audit_path)

    try:
        df_basis = build_current_coupon_basis_df(FNAME)
    except Exception as exc:
        df_basis = None
        print(f"Current-coupon basis section skipped: {exc}")

    html_path = generate_html_report(dat, df_basis=df_basis)

    print("\n" + "=" * 70)
    print("MBS RV Analysis Report (HTML)")
    print("=" * 70)
    print(f"Latest date : {dat.index[-1].date()}")
    print(f"Observations: {len(dat)}")
    print(f"CC basis    : {'included' if df_basis is not None else 'skipped'}")
    print(f"HTML report saved to: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
