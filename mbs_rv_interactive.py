"""Interactive Plotly HTML report for the MBS RV monitor.

Consumes the payload dict assembled by ``mbs_rv_report._assemble_report_data``
and writes a single self-contained HTML file: plotly.js is inlined, every
figure is embedded as JSON, and the sidebar + filter UX mirrors the static
HTML report produced by ``mbs_rv_report.HtmlPages``.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

import mbs_rv_report as rv

REPORT_TITLE = "MBS RV Analysis Report"

NAV_GROUP_ORDER = [
    "Overview",
    "Coupon structures",
    "G2 & FN relative value",
    "Yield basis",
    "Performance",
    "Current-coupon basis",
    "Appendix",
]

# Matplotlib tab10 palette, used by the recent-performance charts
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
CORR_COLORS = {20: "#2563eb", 40: "#d97706", 60: "#059669"}
FN_10Y_COLORS = {20: "#dc2626", 40: "#2563eb", 60: "#475569"}
MOM_COLORS = {20: "#dc2626", 60: "#2563eb"}
MA_COLORS = {50: "orange", 100: "green", 200: "purple"}
VOL_COLORS = {20: "red", 40: "blue", 60: "gray"}
FAMILY_COLORS = {
    "Current-coupon basis": "#2563eb",
    "G2/FN relative perf": "#7c3aed",
    "Adjacent coupon swap": "#059669",
    "100bp coupon swap": "#d97706",
    "Three-price fly": "#dc2626",
}
FAMILY_ORDER = list(FAMILY_COLORS)
AGENCY_NAMES = {"FN": "Conventional", "G2": "Ginnie"}

# Cross-structure family -> sidebar group
FAMILY_GROUPS = {
    "Current-coupon basis": "Current-coupon basis",
    "G2/FN relative perf": "G2 & FN relative value",
    "Adjacent coupon swap": "Coupon structures",
    "100bp coupon swap": "Coupon structures",
    "Three-price fly": "Coupon structures",
}


# ---------------------------------------------------------------------------
# Small conversion / layout helpers
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _subplot_axis(idx: int) -> Tuple[str, str]:
    """Plotly axis names for the idx-th (1-based, row-major) simple subplot."""
    if idx == 1:
        return "x", "y"
    return f"x{idx}", f"y{idx}"


def _add_annotation(
    fig: go.Figure,
    text: str,
    xname: str,
    yname: str,
    x: float = 0.98,
    y: float = 0.04,
    align: str = "right",
    valign: str = "bottom",
    fontsize: int = 11,
    bgcolor: str = "white",
) -> None:
    fig.add_annotation(
        xref=f"{xname} domain",
        yref=f"{yname} domain",
        x=x,
        y=y,
        text=text.replace("\n", "<br>"),
        showarrow=False,
        xanchor=align,
        yanchor=valign,
        align=align,
        font=dict(size=fontsize),
        bgcolor=bgcolor,
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=4,
        opacity=0.95,
    )


def _apply_layout(fig: go.Figure, title: str, height: int = 680, subtitle: Optional[str] = None) -> go.Figure:
    title_text = title.replace("\n", "<br>")
    if subtitle:
        title_text += f'<br><span style="font-size:12px;color:#64748b">{subtitle}</span>'
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=60, r=40, t=100, b=50),
        legend=dict(font=dict(size=11)),
    )
    return fig


def _add_rangeslider(fig: go.Figure, row: int, col: int) -> None:
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        row=row,
        col=col,
    )


def _latest_marker(fig: go.Figure, series: pd.Series, color: str, row: int, col: int) -> None:
    if series.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=_xs(series.index[-1:]),
            y=[float(series.iloc[-1])],
            mode="markers",
            marker=dict(color=color, size=7),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


# ---------------------------------------------------------------------------
# Heatmap coloring (mirrors mbs_rv_report._column_color_func)
# ---------------------------------------------------------------------------
def _column_color_css(
    series: pd.Series,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    force_diverging: bool = False,
):
    s = series.dropna()
    if s.empty:
        return lambda x: None
    if vmin is None:
        vmin = float(s.min())
    if vmax is None:
        vmax = float(s.max())
    if force_diverging or (vmin < 0 < vmax):
        limit = max(abs(vmin), abs(vmax), 0.01)
        return lambda x: rv._score_to_color(float(x), -limit, limit)
    return lambda x: rv._sequential_color(float(x), vmin, vmax)


def _df_to_html_table(
    df: pd.DataFrame,
    gradient_columns: Optional[List[str]] = None,
    gradient_vranges: Optional[Dict[str, Tuple[float, float]]] = None,
    diverging_columns: Optional[List[str]] = None,
    include_index: bool = True,
) -> str:
    """Render a DataFrame as an HTML table with the same heatmap as the PDF."""
    if include_index:
        df = df.reset_index()
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    gradient_columns = gradient_columns or []
    diverging_columns = diverging_columns or []
    gradient_vranges = gradient_vranges or {}
    color_funcs = {}
    for gc in gradient_columns:
        if gc in df.columns:
            vr = gradient_vranges.get(gc)
            color_funcs[gc] = _column_color_css(
                df[gc],
                vmin=vr[0] if vr else None,
                vmax=vr[1] if vr else None,
                force_diverging=(gc in diverging_columns),
            )

    parts = ['<div class="table-wrap"><table class="data-table"><thead><tr>']
    for col in display.columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")
    for i in range(len(display)):
        parts.append("<tr>")
        for col in display.columns:
            style = ""
            if col in color_funcs:
                raw = df[col].iloc[i]
                if pd.notna(raw):
                    color = color_funcs[col](raw)
                    if color:
                        style = f' style="background:{color}"'
            parts.append(f"<td{style}>{html.escape(str(display[col].iloc[i]))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chart builders: recent performance and CT10 rolling correlations
# ---------------------------------------------------------------------------
def _fig_g2_fn_recent_jpm_perf(perf_data: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    coupons = rv.G2_FN_PERF_COUPONS
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    for r, group in enumerate([coupons[:4], coupons[4:]], start=1):
        latest_lines = []
        for coupon in group:
            color = TAB10[coupons.index(coupon) % 10]
            col = f"g2_fn_{coupon:g}_perf_32nds"
            cumulative = perf_data[col].dropna().tail(rv.G2_FN_RECENT_PLOT_WINDOW).cumsum()
            fig.add_trace(
                go.Scatter(
                    x=_xs(cumulative.index), y=_ys(cumulative), mode="lines",
                    name=f"G2/FN {coupon:g}", line=dict(color=color, width=1.5),
                ),
                row=r, col=1,
            )
            _latest_marker(fig, cumulative, color, r, 1)
            row_s = summary.loc[summary["coupon"] == f"G2/FN {coupon:g}"].iloc[0]
            latest_lines.append(
                f"{row_s['coupon']}: 5d {row_s['perf_5d']:+.2f}, "
                f"20d {row_s['perf_20d']:+.2f}, 40d {row_s['perf_40d']:+.2f}"
            )
        fig.add_hline(0, line=dict(color="#64748b", width=1, dash="dash"), row=r, col=1)
        fig.update_yaxes(title_text="Cumulative relative Perf (32nds)", row=r, col=1)
        xname, yname = _subplot_axis(r)
        _add_annotation(
            fig, "\n".join(latest_lines), xname, yname,
            x=0.02, y=0.03, align="left", valign="bottom", fontsize=10,
        )
    _apply_layout(
        fig,
        f"G2 Minus FN JPM Performance - Trailing {rv.G2_FN_RECENT_PLOT_WINDOW} Observations",
        height=780,
    )
    _add_rangeslider(fig, row=2, col=1)
    return fig


def _fig_structure_recent(
    structure_data: pd.DataFrame,
    pairs: List[Tuple[float, float]],
    key_pattern: str,
    title: str,
    panel_label: str,
    y_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[f"{agency} {panel_label}" for agency in rv.STRUCTURE_AGENCIES],
    )
    for r, agency in enumerate(rv.STRUCTURE_AGENCIES, start=1):
        for i, (high, low) in enumerate(pairs):
            key = key_pattern.format(agency=agency.lower(), high=f"{high:g}", low=f"{low:g}")
            cumulative = structure_data[key].dropna().tail(rv.STRUCTURE_RECENT_PLOT_WINDOW).cumsum()
            color = TAB10[i % 10]
            fig.add_trace(
                go.Scatter(
                    x=_xs(cumulative.index), y=_ys(cumulative), mode="lines",
                    name=f"{agency} {high:g}/{low:g}", line=dict(color=color, width=1.35),
                ),
                row=r, col=1,
            )
            _latest_marker(fig, cumulative, color, r, 1)
        fig.add_hline(0, line=dict(color="#64748b", width=1, dash="dash"), row=r, col=1)
        fig.update_yaxes(title_text=y_label, row=r, col=1)
    _apply_layout(
        fig, f"{title} - Trailing {rv.STRUCTURE_RECENT_PLOT_WINDOW} Observations", height=780
    )
    _add_rangeslider(fig, row=2, col=1)
    return fig


def _fig_coupon_fly_recent_price(fly_data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[f"{agency} coupon flies" for agency in rv.STRUCTURE_AGENCIES],
    )
    for r, agency in enumerate(rv.STRUCTURE_AGENCIES, start=1):
        for i, center in enumerate(rv.COUPON_FLY_CENTERS):
            key = f"{agency.lower()}_{center:g}_fly_level_32nds"
            recent = fly_data[key].dropna().tail(rv.STRUCTURE_RECENT_PLOT_WINDOW)
            rebased = recent - recent.iloc[0]
            color = TAB10[i % 10]
            fig.add_trace(
                go.Scatter(
                    x=_xs(rebased.index), y=_ys(rebased), mode="lines",
                    name=f"{agency} {center:g} fly", line=dict(color=color, width=1.35),
                ),
                row=r, col=1,
            )
            _latest_marker(fig, rebased, color, r, 1)
        fig.add_hline(0, line=dict(color="#64748b", width=1, dash="dash"), row=r, col=1)
        fig.update_yaxes(title_text="Fly price change (32nds, rebased)", row=r, col=1)
    _apply_layout(
        fig,
        f"Coupon Fly Price Changes - Trailing {rv.STRUCTURE_RECENT_PLOT_WINDOW} Observations",
        height=780,
    )
    _add_rangeslider(fig, row=2, col=1)
    return fig


def _window_legend_traces(fig: go.Figure) -> None:
    """One legend entry per correlation window plus the all-sample dash."""
    for window in rv.G2_FN_CORR_WINDOWS:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines", name=f"{window}d rolling",
                line=dict(color=CORR_COLORS[window], width=1.2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", name="All sample",
            line=dict(color="#111827", width=1, dash="dash"),
        )
    )


def _corr_panel_annotation(fig: go.Figure, values: Dict[str, float], idx: int) -> None:
    text = "  ".join(
        [f"{w}d {values[f'corr_{w}d']:+.2f}" for w in rv.G2_FN_CORR_WINDOWS]
        + [f"All {values['corr_all']:+.2f}"]
    )
    xname, yname = _subplot_axis(idx)
    _add_annotation(fig, text, xname, yname, x=0.02, y=0.98, align="left", valign="top", fontsize=10)


def _fig_g2_fn_ct10_rolling_corr(perf_data: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    coupons = rv.G2_FN_PERF_COUPONS
    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"G2/FN {c:g}" for c in coupons],
        vertical_spacing=0.055, horizontal_spacing=0.07,
    )
    for i, coupon in enumerate(coupons):
        r, c = i // 2 + 1, i % 2 + 1
        row_s = summary.loc[summary["coupon"] == f"G2/FN {coupon:g}"].iloc[0]
        for window in rv.G2_FN_CORR_WINDOWS:
            col = f"g2_fn_{coupon:g}_corr_{window}d"
            fig.add_trace(
                go.Scatter(
                    x=_xs(perf_data.index), y=_ys(perf_data[col]), mode="lines",
                    line=dict(color=CORR_COLORS[window], width=1.2), showlegend=False,
                ),
                row=r, col=c,
            )
        fig.add_hline(row_s["corr_all"], line=dict(color="#111827", width=1, dash="dash"), row=r, col=c)
        fig.add_hline(0, line=dict(color="#94a3b8", width=0.7), row=r, col=c)
        fig.update_yaxes(range=[-1.05, 1.05], row=r, col=c)
        _corr_panel_annotation(fig, row_s, i + 1)
    _window_legend_traces(fig)
    _apply_layout(
        fig,
        "G2 Minus FN Daily JPM Performance vs Daily CT10 Yield Change",
        height=1000,
        subtitle="Rolling correlations use paired daily observations; CT10 change is measured in bps.",
    )
    return fig


def _fig_structure_ct10_rolling_corr(
    structure_data: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    agency: str,
) -> go.Figure:
    sub = summary[summary["agency"] == agency].reset_index(drop=True)
    nrows = (len(sub) + 1) // 2
    fig = make_subplots(
        rows=nrows, cols=2, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=sub["structure"].tolist(),
        vertical_spacing=min(0.09, 0.5 / max(nrows, 1)), horizontal_spacing=0.07,
    )
    for i, row_s in sub.iterrows():
        r, c = i // 2 + 1, i % 2 + 1
        key = row_s["series_key"]
        for window in rv.G2_FN_CORR_WINDOWS:
            fig.add_trace(
                go.Scatter(
                    x=_xs(structure_data.index),
                    y=_ys(structure_data[f"{key}_corr_{window}d"]),
                    mode="lines", line=dict(color=CORR_COLORS[window], width=1.0),
                    showlegend=False,
                ),
                row=r, col=c,
            )
        fig.add_hline(row_s["corr_all"], line=dict(color="#111827", width=1, dash="dash"), row=r, col=c)
        fig.add_hline(0, line=dict(color="#94a3b8", width=0.6), row=r, col=c)
        fig.update_yaxes(range=[-1.05, 1.05], row=r, col=c)
        _corr_panel_annotation(fig, row_s, i + 1)
    _window_legend_traces(fig)
    _apply_layout(
        fig,
        f"{agency} {title} vs Daily CT10 Yield Change",
        height=int(260 * nrows + 200),
        subtitle="Correlation uses daily structure changes and daily CT10 yield changes; dashed line = full sample.",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart builders: current-coupon basis suite
# ---------------------------------------------------------------------------
def _fig_cc_fitted(df_basis: pd.DataFrame, response_col: str) -> go.Figure:
    name = {"FNCC": "Conventional", "G2CC": "Ginnie"}[response_col]
    Y, _, model = rv.run_current_coupon_ols(df_basis, response_col)
    residual = model.resid
    resid_std = float(np.std(residual))
    latest_resid = float(residual.iloc[-1])
    ratio = latest_resid / resid_std if resid_std != 0 else np.nan
    lag_text = " (lagged 1d)" if rv.CC_LAG_PREDICTORS else ""

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=_xs(Y.index), y=_ys(Y), mode="lines", name="Actual OAS",
                   line=dict(color="firebrick", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(Y.index), y=_ys(model.fittedvalues), mode="lines",
                   name="Full-sample fitted OAS", line=dict(color="royalblue", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(Y.index), y=_ys(residual), mode="lines", fill="tozeroy",
                   name="Residual = actual - fitted",
                   line=dict(color="slateblue", width=1),
                   fillcolor="rgba(106, 90, 205, 0.15)"),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    _add_annotation(
        fig,
        f"Latest actual: {Y.iloc[-1]:.2f} bps\n"
        f"Latest fitted: {model.fittedvalues.iloc[-1]:.2f} bps\n"
        f"Residual: {latest_resid:+.2f} bps  |  z = {ratio:.2f}",
        "x", "y", fontsize=10,
    )
    _apply_layout(
        fig,
        f"{name} ({rv.CC_TICKERS[name]})<br>"
        "Current-Coupon Basis: Full-Sample Actual, Fitted and Residual",
        height=760,
        subtitle=(
            f"Sample: {Y.index[0].date()} to {Y.index[-1].date()}  |  "
            f"N = {int(model.nobs):,}  |  R-squared = {model.rsquared:.3f}  |  "
            f"Predictors: 10y, 2s10s, 1y10y_vol{lag_text}  |  Intercept included"
        ),
    )
    return fig


def _fig_rolling_cc(rolling_df: pd.DataFrame, cc_type: str, window: int) -> go.Figure:
    sub = rolling_df[rolling_df["cc_type"] == cc_type].sort_index()
    if sub.empty:
        raise ValueError(f"No rolling current-coupon observations for {cc_type}.")
    latest = sub.iloc[-1]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.67, 0.33], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=_xs(sub.index), y=_ys(sub["mkt_val"]), mode="lines",
                   name="Actual OAS", line=dict(color="firebrick", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(sub.index), y=_ys(sub["mdl_val"]), mode="lines",
                   name=f"{window}-day rolling fitted OAS", line=dict(color="royalblue", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_xs(sub.index), y=_ys(sub["resid"]), mode="lines", fill="tozeroy",
                   name="Residual = actual - fitted",
                   line=dict(color="slateblue", width=1),
                   fillcolor="rgba(106, 90, 205, 0.15)"),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    _add_annotation(
        fig,
        f"Latest actual: {latest['mkt_val']:.2f} bps\n"
        f"Latest fitted: {latest['mdl_val']:.2f} bps\n"
        f"Residual: {latest['resid']:+.2f} bps  |  z = {latest['resid_z']:.2f}",
        "x", "y", fontsize=10,
    )
    _apply_layout(
        fig,
        f"{cc_type} ({rv.CC_TICKERS[cc_type]})<br>"
        f"Current-Coupon Basis: {window}-Day Rolling Actual, Fitted and Residual",
        height=760,
        subtitle=(
            f"Rolling estimates: {sub.index[0].date()} to {sub.index[-1].date()}  |  "
            f"Window = {window} observations  |  "
            "Predictors: 10y, 2s10s, 1y10y_vol  |  Intercept included"
        ),
    )
    return fig


def _fig_conv_ginnie_cc_spread(df_basis: pd.DataFrame, window: int = 90) -> go.Figure:
    spread = df_basis["FNCC"] - df_basis["G2CC"]
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    fig.add_trace(
        go.Scatter(x=_xs(spread.index), y=_ys(spread), mode="lines",
                   name="Conv - Ginnie OAS", line=dict(color="navy", width=1.2)),
        row=1, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=_xs(zscore.index), y=_ys(zscore), mode="lines",
                   name="z-score", line=dict(color="purple", width=1.2)),
        row=2, col=1,
    )
    for level in (0,):
        fig.add_hline(level, line=dict(color="black", width=0.8), row=2, col=1)
    for level in (1, -1):
        fig.add_hline(level, line=dict(color="red", width=0.8, dash="dash"), row=2, col=1)
    fig.update_yaxes(title_text="OAS spread (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    _add_annotation(
        fig,
        f"current diff={float(spread.iloc[-1]):.1f}\ncurrent z={float(zscore.iloc[-1]):.2f}",
        "x", "y", fontsize=10, bgcolor="wheat",
    )
    _apply_layout(
        fig, f"Conv vs Ginnie Current-Coupon OAS Spread ({window}-day z-score)", height=720
    )
    return fig


def _fig_cc_ma_vol(df_basis: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.09, horizontal_spacing=0.08,
        subplot_titles=[
            "Conventional (.FNCC105)", "Ginnie (.G2CC105)",
            "Conventional realized vol (daily)", "Ginnie realized vol (daily)",
        ],
    )
    for j, col in enumerate(["FNCC", "G2CC"]):
        if col not in df_basis.columns:
            continue
        s = df_basis[col]
        c = j + 1
        fig.add_trace(
            go.Scatter(x=_xs(s.index), y=_ys(s), mode="lines",
                       name=f"{col} OAS: {float(s.dropna().iloc[-1]):.1f} bps",
                       line=dict(color="navy", width=1.1)),
            row=1, col=c,
        )
        for w in rv.CC_MA_WINDOWS:
            ma = s.rolling(window=w, min_periods=w).mean()
            fig.add_trace(
                go.Scatter(x=_xs(s.index), y=_ys(ma), mode="lines",
                           name=f"MA{w}: {float(ma.dropna().iloc[-1]):.1f} bps",
                           line=dict(color=MA_COLORS[w], width=1.1)),
                row=1, col=c,
            )
        d = s.diff()
        for w in rv.CC_VOL_WINDOWS:
            vol = d.rolling(window=w, min_periods=w).std() * np.sqrt(rv.CC_VOL_ANN_FACTOR)
            fig.add_trace(
                go.Scatter(x=_xs(d.index), y=_ys(vol), mode="lines",
                           name=f"{col} {w}d: {float(vol.dropna().iloc[-1]):.2f} bps/day",
                           line=dict(color=VOL_COLORS[w], width=1.1)),
                row=2, col=c,
            )
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Vol (bps/day)", row=2, col=1)
    _apply_layout(fig, "Current-Coupon Basis: Moving Averages and Realized Vol", height=780)
    return fig


def _fig_cc_momentum(history: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.11, horizontal_spacing=0.08,
    )
    for j, (col, name, summary_name) in enumerate(
        [
            ("FNCC", "Conventional (.FNCC105)", "Conventional"),
            ("G2CC", "Ginnie (.G2CC105)", "Ginnie"),
        ]
    ):
        if f"{col}_level" not in history.columns:
            continue
        c = j + 1
        level = history[f"{col}_level"]
        fig.add_trace(
            go.Scatter(x=_xs(level.index), y=_ys(level), mode="lines",
                       name=f"{col} OAS: {level.dropna().iloc[-1]:.1f} bps",
                       line=dict(color="#111827", width=1.2)),
            row=1, col=c,
        )
        for window in rv.CC_MOM_WINDOWS:
            ema = level.ewm(span=window, adjust=False, min_periods=window).mean()
            fig.add_trace(
                go.Scatter(x=_xs(ema.index), y=_ys(ema), mode="lines",
                           name=f"{col} EMA{window}: {ema.dropna().iloc[-1]:.1f} bps",
                           line=dict(color=MOM_COLORS[window], width=1.2)),
                row=1, col=c,
            )
        trend = summary.loc[summary_name, "trend"]
        strength = summary.loc[summary_name, "strength"].lower()
        meaning = {"Upward": "widening", "Downward": "tightening", "Mixed": "turning"}[trend]
        xname, yname = _subplot_axis(c)
        _add_annotation(
            fig, f"{name}<br>Current trend: {trend} / {meaning} ({strength})",
            xname, yname, x=0.02, y=0.98, align="left", valign="top", fontsize=10,
        )
        for window in rv.CC_MOM_WINDOWS:
            score = history[f"{col}_mom_{window}d"]
            fig.add_trace(
                go.Scatter(x=_xs(score.index), y=_ys(score), mode="lines",
                           name=f"{col} mom {window} obs: {score.dropna().iloc[-1]:+.2f}",
                           line=dict(color=MOM_COLORS[window], width=1.3)),
                row=2, col=c,
            )
        fig.add_hline(0, line=dict(color="#111827", width=0.8), row=2, col=c)
        for level_ in (1, -1):
            fig.add_hline(level_, line=dict(color="#94a3b8", width=0.8, dash="dash"), row=2, col=c)
    fig.update_yaxes(title_text="OAS (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Momentum score", row=2, col=1)
    _apply_layout(
        fig,
        "Current-Coupon Basis Momentum Monitor",
        height=800,
        subtitle="Positive = basis widening; negative = basis tightening. "
                 "Score = rolling OLS slope x sqrt(window) / daily realized volatility.",
    )
    return fig


def _fig_fn_10y_rolling_corr(corr_df: pd.DataFrame, all_sample: Optional[dict]) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.07,
        subplot_titles=[
            "Full available aligned sample",
            "Most recent 250 observations",
            f"Rolling-correlation z-score over {rv.CC_FN_10Y_CORR_Z_WINDOW} observations",
        ],
    )
    for window in rv.CC_FN_10Y_CORR_WINDOWS:
        column = f"corr_{window}d"
        color = FN_10Y_COLORS[window]
        fig.add_trace(
            go.Scatter(x=_xs(corr_df.index), y=_ys(corr_df[column]), mode="lines",
                       name=f"{window} obs", line=dict(color=color, width=1.4),
                       legendgroup=f"w{window}"),
            row=1, col=1,
        )
        recent = corr_df[column].tail(250)
        fig.add_trace(
            go.Scatter(x=_xs(recent.index), y=_ys(recent), mode="lines",
                       name=f"{window} obs", line=dict(color=color, width=1.5),
                       legendgroup=f"w{window}", showlegend=False),
            row=2, col=1,
        )
        z_col = f"{column}_z{rv.CC_FN_10Y_CORR_Z_WINDOW}"
        recent_z = corr_df[z_col].dropna().tail(250)
        fig.add_trace(
            go.Scatter(x=_xs(recent_z.index), y=_ys(recent_z), mode="lines",
                       name=f"{window} obs", line=dict(color=color, width=1.4),
                       legendgroup=f"w{window}", showlegend=False),
            row=3, col=1,
        )
    for r in (1, 2):
        fig.add_hline(0, line=dict(color="#111827", width=0.8), row=r, col=1)
        fig.update_yaxes(range=[-1, 1], title_text="Correlation", row=r, col=1)
    fig.add_hline(0, line=dict(color="#111827", width=0.8), row=3, col=1)
    for level in (2, -2):
        fig.add_hline(level, line=dict(color="#b91c1c", width=0.8, dash="dash"), row=3, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)

    latest_text = "  |  ".join(
        f"{w} obs: {corr_df[f'corr_{w}d'].dropna().iloc[-1]:+.2f} "
        f"(z {corr_df[f'corr_{w}d_z{rv.CC_FN_10Y_CORR_Z_WINDOW}'].dropna().iloc[-1]:+.2f})"
        for w in rv.CC_FN_10Y_CORR_WINDOWS
    )
    subtitle = latest_text
    if all_sample is not None:
        subtitle += (
            f"<br>All sample: daily changes {all_sample['change_corr']:+.2f} "
            f"({all_sample['change_obs']} obs)  |  levels {all_sample['level_corr']:+.2f} "
            f"({all_sample['level_obs']} obs)"
        )
    _apply_layout(
        fig,
        "FN Current-Coupon Basis vs UST 10y: Rolling Correlation of Daily Changes",
        height=950,
        subtitle=subtitle,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart builders: per-coupon yield basis and distributions
# ---------------------------------------------------------------------------
def _fig_coupon_basis_fitted(cb_basis: pd.DataFrame, agency: str) -> Optional[go.Figure]:
    response_cols = [
        c for c in cb_basis.columns
        if c not in rv.COUPON_BASIS_PREDICTORS and c.split()[0] == agency
    ]
    cols_sorted = sorted(response_cols, key=lambda x: float(x.split()[1]))
    if not cols_sorted:
        return None
    n_cols = len(cols_sorted)
    nrows = int(np.ceil(n_cols / 4))
    ncols = min(n_cols, 4)
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=cols_sorted,
        vertical_spacing=0.09, horizontal_spacing=0.07,
    )
    for i, col in enumerate(cols_sorted):
        r, c = i // ncols + 1, i % ncols + 1
        Y, _, model = rv.run_coupon_basis_ols(cb_basis, col)
        show_legend = i == 0
        fig.add_trace(
            go.Scatter(x=_xs(Y.index), y=_ys(Y), mode="lines", name="Actual",
                       line=dict(color="red", width=1.0), legendgroup="act",
                       showlegend=show_legend),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(x=_xs(Y.index), y=_ys(model.fittedvalues), mode="lines",
                       name="Fitted", line=dict(color="blue", width=1.0),
                       legendgroup="fit", showlegend=show_legend),
            row=r, col=c,
        )
        resid_std = float(np.std(model.resid))
        diff = float(Y.iloc[-1] - model.fittedvalues.iloc[-1])
        xname, yname = _subplot_axis(i + 1)
        _add_annotation(
            fig,
            f"mkt={Y.iloc[-1]:.1f}\nmdl={model.fittedvalues.iloc[-1]:.1f}\n"
            f"z={diff / resid_std if resid_std else np.nan:.2f}",
            xname, yname, fontsize=9, bgcolor="wheat",
        )
        fig.update_yaxes(title_text="Basis (bps)", row=r, col=c)
    _apply_layout(fig, f"Per-Coupon Yield Basis OLS: {agency} MBS", height=int(300 * nrows + 160))
    return fig


def _fig_distribution_page(
    family: str, page_items: List[Dict], page_no: int, page_count: int
) -> go.Figure:
    color = FAMILY_COLORS[family]
    subplot_titles = []
    for item in page_items:
        values = pd.to_numeric(item["values"], errors="coerce").dropna()
        subplot_titles.append(
            f"{item['structure']}  |  N = {len(values):,}  |  Skew = {values.skew():+.2f}"
        )
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=subplot_titles + [""] * (6 - len(page_items)),
        vertical_spacing=0.09, horizontal_spacing=0.08,
    )
    for i, item in enumerate(page_items):
        r, c = i // 2 + 1, i % 2 + 1
        values = pd.to_numeric(item["values"], errors="coerce").dropna()
        bins = min(32, max(12, int(np.sqrt(len(values)) * 1.5)))
        mean = float(values.mean())
        std = float(values.std())
        p05 = float(values.quantile(0.05))
        p95 = float(values.quantile(0.95))
        fig.add_trace(
            go.Histogram(
                x=_ys(values), nbinsx=bins, marker_color=color, opacity=0.72,
                showlegend=False,
            ),
            row=r, col=c,
        )
        fig.add_vline(mean, line=dict(color="#111827", width=1.2), row=r, col=c)
        for bound in (mean - std, mean + std):
            fig.add_vline(bound, line=dict(color="#475569", width=0.9, dash="dash"), row=r, col=c)
        for bound in (p05, p95):
            fig.add_vline(bound, line=dict(color="#94a3b8", width=0.8, dash="dot"), row=r, col=c)
        xname, yname = _subplot_axis(i + 1)
        _add_annotation(
            fig,
            f"Mean {mean:+.2f}  |  Std {std:.2f}  |  P05/P95 {p05:+.2f}/{p95:+.2f}",
            xname, yname, x=0.98, y=0.98, align="right", valign="top", fontsize=9,
        )
        fig.update_xaxes(title_text=item["unit"], row=r, col=c)
        fig.update_yaxes(title_text="Observations", row=r, col=c)
    _apply_layout(
        fig,
        f"{family}: Daily Distribution ({page_no}/{page_count})",
        height=950,
        subtitle="Histogram uses the full available sample; dashed lines are mean +/- 1 standard "
                 "deviation; dotted lines are the 5th and 95th percentiles.",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart builders: appendix (historical structures and coupon-drop grids)
# ---------------------------------------------------------------------------
def _fig_historical_cpn_fly(
    dat: pd.DataFrame, mode: str, pair: Tuple[str, str], triple: Tuple[str, str, str]
) -> go.Figure:
    ref_cols = list(rv.HIST_REF_COLS)
    series_full, label = rv._build_hist_swap_fly_series(dat, mode=mode, pair=pair, triple=triple)
    latest_val = float(series_full.iloc[-1])
    hist_min = float(series_full.min())
    hist_max = float(series_full.max())

    carry_full = rv._build_hist_carry_series(dat, mode=mode, pair=pair, triple=triple)
    carry_drops_text = ""
    if carry_full is not None and not carry_full.empty:
        latest_carry = float(carry_full.iloc[-1])
        if mode == "cpn":
            cpn1, cpn2 = pair
            d1 = float(dat[f"{cpn1} Drop"].iloc[-1])
            d2 = float(dat[f"{cpn2} Drop"].iloc[-1])
            carry_drops_text = f"Carry: {latest_carry:.2f}  |  Drops: {cpn1}={d1:.2f}, {cpn2}={d2:.2f}"
        elif mode == "fly":
            front, middle, back = triple
            d_front = float(dat[f"{front} Drop"].iloc[-1])
            d_mid = float(dat[f"{middle} Drop"].iloc[-1])
            d_back = float(dat[f"{back} Drop"].iloc[-1])
            carry_drops_text = (
                f"Carry: {latest_carry:.2f}  |  "
                f"Drops: {front}={d_front:.2f}, {middle}={d_mid:.2f}, {back}={d_back:.2f}"
            )

    n_refs = len(ref_cols)
    top_titles = [
        f"{label} vs {ref_col}" + (f"<br>{carry_drops_text}" if carry_drops_text else "")
        for ref_col in ref_cols
    ]
    bottom_titles = [f"{label} vs {ref_col}" for ref_col in ref_cols]
    fig = make_subplots(
        rows=2, cols=n_refs,
        specs=[[{"secondary_y": True}] * n_refs, [{}] * n_refs],
        subplot_titles=top_titles + bottom_titles,
        vertical_spacing=0.16, horizontal_spacing=0.09,
    )
    for j, ref_col in enumerate(ref_cols):
        c = j + 1
        ref = dat[ref_col]
        df = pd.concat([series_full, ref], axis=1).dropna()
        if df.empty:
            continue
        series = df.iloc[:, 0]
        ref = df.iloc[:, 1]
        fig.add_trace(
            go.Scatter(x=_xs(series.index), y=_ys(series), mode="lines", name=label,
                       line=dict(color="navy", width=1.2), showlegend=(j == 0)),
            row=1, col=c, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=_xs(ref.index), y=_ys(ref), mode="lines", name=ref_col,
                       line=dict(color="red", width=1.0), showlegend=True),
            row=1, col=c, secondary_y=True,
        )
        fig.update_yaxes(title_text=f"{label} (32nds)", row=1, col=c, secondary_y=False)
        fig.update_yaxes(title_text=ref_col, row=1, col=c, secondary_y=True)
        fig.add_trace(
            go.Scatter(
                x=_ys(ref.iloc[:-1]), y=_ys(series.iloc[:-1]), mode="markers",
                name="Historical data", showlegend=(j == 0),
                marker=dict(color="gray", size=5, opacity=0.35),
            ),
            row=2, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[float(ref.iloc[-1])], y=[float(series.iloc[-1])], mode="markers",
                name="Latest point", showlegend=(j == 0),
                marker=dict(color="red", size=14, symbol="star",
                            line=dict(color="black", width=1.5)),
            ),
            row=2, col=c,
        )
        fig.update_xaxes(title_text=ref_col, row=2, col=c)
        fig.update_yaxes(title_text=f"{label} (32nds)", row=2, col=c)

    _add_annotation(
        fig,
        f"Latest: {latest_val:.2f}\nMax: {hist_max:.2f}\nMin: {hist_min:.2f}",
        "x", "y", x=0.98, y=0.98, align="right", valign="top", fontsize=10,
        bgcolor="wheat",
    )
    _apply_layout(fig, f"Historical {label}", height=860)
    return fig


def _fig_coupon_drop_grid(dat: pd.DataFrame, agency: str, kind: str) -> Optional[go.Figure]:
    """4x2 appendix grids: 'drop', 'sharpe' or 'fwdvol'."""
    if kind == "drop":
        colors = {"FN": "#2563eb", "G2": "#dc2626"}
        suptitle = f"{AGENCY_NAMES.get(agency, agency)} Coupon Drops - Full Available History"
        y_label = "Drop (32nds)"
    elif kind == "sharpe":
        colors = {"FN": "#7c3aed", "G2": "#059669"}
        suptitle = f"{AGENCY_NAMES.get(agency, agency)} Coupon Monthly Drop Sharpe - Full Available History"
        y_label = "Drop / monthly-equivalent vol"
    else:
        colors = {"FN": "#d97706", "G2": "#0891b2"}
        suptitle = f"{AGENCY_NAMES.get(agency, agency)} Coupon 1y10y Forward-Vol Sharpe Proxy"
        y_label = "Drop / monthly 1y10y vol"

    if kind == "fwdvol" and rv.COUPON_FORWARD_VOL_COL not in dat.columns:
        return None
    annualized_vol = pd.to_numeric(dat[rv.COUPON_FORWARD_VOL_COL], errors="coerce") if kind == "fwdvol" else None
    monthly_fwd_vol = annualized_vol / np.sqrt(rv.COUPON_FORWARD_VOL_MONTHS) if kind == "fwdvol" else None

    available = [
        coupon for coupon in rv.COUPONS
        if rv._col(agency, coupon, "Drop") in dat.columns
        and (kind != "sharpe" or rv._col(agency, coupon, "Perf") in dat.columns)
    ]
    if not available:
        return None

    panel_series = []
    panel_titles = []
    for coupon in available:
        drop = pd.to_numeric(dat[rv._col(agency, coupon, "Drop")], errors="coerce")
        if kind == "drop":
            series = drop.dropna()
            if series.empty:
                continue
            latest = float(series.iloc[-1])
            rolling_mean = series.rolling(rv.COUPON_DROP_Z_WINDOW, min_periods=rv.COUPON_DROP_Z_WINDOW).mean()
            rolling_std = series.rolling(rv.COUPON_DROP_Z_WINDOW, min_periods=rv.COUPON_DROP_Z_WINDOW).std(ddof=1)
            rolling_z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            latest_z = float(rolling_z.dropna().iloc[-1])
            title = f"{agency} {coupon:g}  |  Current: {latest:.2f}  |  Z120: {latest_z:+.2f}"
        elif kind == "sharpe":
            perf = pd.to_numeric(dat[rv._col(agency, coupon, "Perf")], errors="coerce")
            vol = perf.rolling(rv.COUPON_SHARPE_VOL_WINDOW, min_periods=rv.COUPON_SHARPE_VOL_WINDOW).std(ddof=1)
            monthly_vol = vol * np.sqrt(rv.COUPON_SHARPE_MONTH_DAYS)
            series = (drop / monthly_vol.replace(0, np.nan)).dropna()
            if series.empty:
                continue
            title = f"{agency} {coupon:g}  |  Monthly Sharpe60: {float(series.iloc[-1]):+.2f}"
        else:
            series = (drop / monthly_fwd_vol.replace(0, np.nan)).dropna()
            if series.empty:
                continue
            title = f"{agency} {coupon:g}  |  Forward-vol proxy: {float(series.iloc[-1]):+.3f}"
        panel_series.append(series)
        panel_titles.append(title)

    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=panel_titles + [""] * (8 - len(panel_titles)),
        vertical_spacing=0.055, horizontal_spacing=0.07,
    )
    for i, series in enumerate(panel_series):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Scatter(x=_xs(series.index), y=_ys(series), mode="lines",
                       line=dict(color=colors[agency], width=1.25), showlegend=False),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(x=_xs(series.index[-1:]), y=[float(series.iloc[-1])], mode="markers",
                       marker=dict(color="#111827", size=6), showlegend=False, hoverinfo="skip"),
            row=r, col=c,
        )
        fig.add_hline(0, line=dict(color="#64748b", width=0.7, dash="dash"), row=r, col=c)
        fig.update_yaxes(title_text=y_label, row=r, col=c)
    _apply_layout(fig, suptitle, height=1000)
    return fig


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
class _ReportBuilder:
    def __init__(self):
        self.pages: List[Dict[str, str]] = []
        self.fig_json: Dict[str, str] = {}
        self._fig_count = 0

    def _next_fig_id(self) -> str:
        self._fig_count += 1
        return f"fig-{self._fig_count:03d}"

    def _fig_block(
        self, fig: go.Figure, css_class: str = "chart", extra_attrs: str = "", extra_style: str = ""
    ) -> str:
        fid = self._next_fig_id()
        self.fig_json[fid] = pio.to_json(fig).replace("</", "<\\/")
        return (
            f'<div class="{css_class}" id="{fid}"{extra_attrs}{extra_style}></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{self.fig_json[fid]}</script>"
        )

    def _add(self, title: str, section: str, body: str, search_extra: str = "") -> None:
        page_id = f"page-{len(self.pages) + 1:03d}"
        search = f"{title} {section} {search_extra}".lower()
        self.pages.append(
            {"id": page_id, "title": title, "section": section, "body": body, "search": search}
        )

    def add_fig(self, title: str, section: str, fig: go.Figure, search_extra: str = "") -> None:
        self._add(title, section, self._fig_block(fig), search_extra)

    def add_table(self, title: str, section: str, table_html: str, subtitle: Optional[str] = None) -> None:
        body = ""
        if subtitle:
            body += f'<p class="view-subtitle">{html.escape(subtitle)}</p>'
        body += table_html
        self._add(title, section, body)

    def add_html(self, title: str, section: str, body: str, search_extra: str = "") -> None:
        self._add(title, section, body, search_extra)

    def add_dropdown(self, title: str, section: str, options: List[Tuple[str, go.Figure]], label: str = "Structure") -> None:
        if not options:
            return
        sel_id = f"sel-{len(self.pages) + 1:03d}"
        option_tags = "".join(
            f'<option value="{i}">{html.escape(text)}</option>' for i, (text, _) in enumerate(options)
        )
        parts = [
            f'<div class="dropdown-bar"><label>{html.escape(label)}: '
            f'<select class="dd-select" id="{sel_id}">{option_tags}</select></label></div>'
        ]
        for i, (_, fig) in enumerate(options):
            hidden = "" if i == 0 else ' style="display:none"'
            parts.append(
                self._fig_block(
                    fig,
                    css_class="chart dd-fig",
                    extra_attrs=f' data-select="{sel_id}" data-value="{i}"',
                    extra_style=hidden,
                )
            )
        search_extra = " ".join(text for text, _ in options)
        self._add(title, section, "".join(parts), search_extra)


def _overview_body(dat: pd.DataFrame, payloads: Dict) -> str:
    latest = dat.index[-1].date()
    latest_px = rv.build_latest_price_table(dat)
    if not latest_px.empty:
        price_html = _df_to_html_table(latest_px, include_index=False)
    else:
        price_html = "<p>No latest price data available.</p>"
    num_structures = len(payloads["cpn_fly_summary"])
    contents_items = [
        "Coupon swap / fly / G2-FN score tables",
        "Yield basis score tables",
        "Performance over periods",
        "Rolling performance z-scores",
        "G2/FN 3.0-6.5 JPM relative performance and CT10 correlation",
        "FN/G2 adjacent-coupon JPM performance swaps and CT10 correlation",
        "FN/G2 100-bps-wide JPM performance swaps and CT10 correlation",
        "FN/G2 three-price coupon flies and CT10 correlation",
        "Current-coupon basis OLS summary and parameters",
        "Current-coupon full-sample and rolling actual / fitted / residual",
        "Conventional vs Ginnie current-coupon OAS spread",
        "Current-coupon moving averages, realized volatility and momentum",
        "FN basis vs UST 10y rolling correlation and z-score",
        "Per-coupon yield basis OLS summary, parameters and fitted charts",
        "Cross-structure CT10 correlation, volatility and daily distributions",
        f"Appendix: historical charts for {num_structures} cpn/fly structures (dropdowns)",
        "Appendix: full-history drop, drop-Sharpe and forward-vol Sharpe charts (dropdowns)",
    ]
    contents_html = "<ol>" + "".join(f"<li>{html.escape(item)}</li>" for item in contents_items) + "</ol>"
    return (
        f'<div class="overview"><h1>{REPORT_TITLE}</h1>'
        f'<p class="ov-sub">Latest date: {latest}  |  Observations: {len(dat):,}</p>'
        f'<div class="ov-box"><h3>Latest MBS Prices ({latest})</h3>'
        f"<p>Prices shown as decimal and handle-32nds (e.g. 98.50 -&gt; 98-16).</p>{price_html}</div>"
        f'<div class="ov-box"><h3>Report contents</h3>{contents_html}</div></div>'
    )


def _perf_gradient_cols() -> List[str]:
    return ["latest", "perf_5d", "perf_20d", "perf_40d",
            "corr_all", "corr_20d", "corr_40d", "corr_60d"]


def _fly_gradient_cols() -> List[str]:
    return ["level_latest", "chg_5d", "chg_20d", "chg_40d",
            "corr_all", "corr_20d", "corr_40d", "corr_60d"]


def _corr_vol_cols() -> List[str]:
    return [
        "corr_all", *[f"corr_{w}d" for w in rv.G2_FN_CORR_WINDOWS],
        "mean", "vol_all", *[f"vol_{w}d" for w in rv.G2_FN_CORR_WINDOWS],
        "p05", "median", "p95",
    ]


def _try_view(b: _ReportBuilder, title: str, section: str, kind: str, *args, **kwargs) -> None:
    """Build one view, skipping it (never crashing) if its construction fails."""
    try:
        if kind == "fig":
            fig = args[0]() if callable(args[0]) else args[0]
            if fig is not None:
                b.add_fig(title, section, fig, **kwargs)
        elif kind == "table":
            b.add_table(title, section, *args, **kwargs)
        elif kind == "dropdown":
            b.add_dropdown(title, section, *args, **kwargs)
    except Exception as exc:
        print(f"Interactive view '{title}' skipped: {exc}")


def _build_pages(payloads: Dict) -> _ReportBuilder:
    b = _ReportBuilder()
    dat = payloads["dat"]
    latest = dat.index[-1].date()

    # Overview / contents
    b.add_html(f"{REPORT_TITLE} - Contents & Latest Prices", "Overview", _overview_body(dat, payloads))

    # Cpn/fly score tables
    for group_name, group_df in payloads["groups"].items():
        _try_view(
            b, f"{group_name}  (latest: {latest})", "Coupon structures", "table",
            _df_to_html_table(
                group_df,
                gradient_columns=payloads["score_cols"],
                gradient_vranges=payloads["gradient_vranges"],
                diverging_columns=payloads["diverging_cols"],
            ),
            subtitle=payloads["subtitle"],
        )

    # Yield basis score tables
    for group_name, group_df in payloads["basis_groups"].items():
        _try_view(
            b, f"{group_name}  (latest: {latest})", "Yield basis", "table",
            _df_to_html_table(group_df, gradient_columns=payloads["basis_score_cols"]),
            subtitle=payloads["basis_subtitle"],
        )

    # Performance tables
    b.add_table(
        "Performance Over Periods", "Performance",
        _df_to_html_table(payloads["perf_summary"], gradient_columns=payloads["perf_cols"]),
        subtitle=f"Cumulative JPM DataQuery performance (32nds) over trailing windows  |  Latest: {latest}",
    )
    b.add_table(
        f"Rolling Performance Z-Scores ({rv.ROLL_SUM_WINDOW}d sum / {rv.ROLL_SUM_Z_WINDOW}d z-score)",
        "Performance",
        _df_to_html_table(payloads["rolling_perf_z"], gradient_columns=payloads["rolling_perf_cols"]),
        subtitle=f"Performance = JPM DataQuery daily performance (32nds)  |  Latest: {latest}",
    )

    # G2/FN relative value
    perf_grad = _perf_gradient_cols()
    b.add_table(
        "G2 Minus FN JPM Performance: 3.0 to 6.5 Coupons", "G2 & FN relative value",
        _df_to_html_table(
            payloads["g2_fn_perf_summary"],
            gradient_columns=perf_grad, diverging_columns=perf_grad, include_index=False,
        ),
        subtitle="Daily relative Perf = G2 JPM Perf - FN JPM Perf (32nds); "
                 "5d/20d/40d are cumulative sums  |  Correlation uses daily CT10 yield changes",
    )
    _try_view(
        b, f"G2 Minus FN JPM Performance - Trailing {rv.G2_FN_RECENT_PLOT_WINDOW} Observations",
        "G2 & FN relative value", "fig",
        lambda: _fig_g2_fn_recent_jpm_perf(payloads["g2_fn_perf_data"], payloads["g2_fn_perf_summary"]),
    )
    _try_view(
        b, "G2 Minus FN Daily JPM Performance vs Daily CT10 Yield Change",
        "G2 & FN relative value", "fig",
        lambda: _fig_g2_fn_ct10_rolling_corr(payloads["g2_fn_perf_data"], payloads["g2_fn_perf_summary"]),
    )

    # Adjacent-coupon swaps
    b.add_table(
        "Adjacent-Coupon Swap JPM Performance", "Coupon structures",
        _df_to_html_table(
            payloads["coupon_swap_display"],
            gradient_columns=perf_grad, diverging_columns=perf_grad, include_index=False,
        ),
        subtitle="Daily swap Perf = higher-coupon JPM Perf - lower-coupon JPM Perf (32nds); "
                 "5d/20d/40d are cumulative sums  |  Correlation uses daily CT10 yield changes",
    )
    _try_view(
        b, f"Coupon Swap JPM Performance - Trailing {rv.STRUCTURE_RECENT_PLOT_WINDOW} Observations",
        "Coupon structures", "fig",
        lambda: _fig_structure_recent(
            payloads["coupon_swap_data"], rv.COUPON_SWAP_PAIRS,
            "{agency}_{high}_{low}_swap_perf_32nds",
            "Coupon Swap JPM Performance", "adjacent-coupon swaps",
            "Cumulative swap Perf (32nds)",
        ),
    )
    for agency in rv.STRUCTURE_AGENCIES:
        _try_view(
            b, f"{agency} Adjacent-Coupon Swap JPM Perf vs Daily CT10 Yield Change",
            "Coupon structures", "fig",
            lambda agency=agency: _fig_structure_ct10_rolling_corr(
                payloads["coupon_swap_data"], payloads["coupon_swap_summary"],
                "Adjacent-Coupon Swap JPM Perf", agency,
            ),
        )

    # 100-bps-wide swaps
    b.add_table(
        "100-bps-Wide Coupon Swap JPM Performance", "Coupon structures",
        _df_to_html_table(
            payloads["wide_swap_display"],
            gradient_columns=perf_grad, diverging_columns=perf_grad, include_index=False,
        ),
        subtitle="Daily swap Perf = higher-coupon JPM Perf - lower-coupon JPM Perf (32nds); "
                 "pairs span 100 bps; 5d/20d/40d are cumulative sums  |  "
                 "Correlation uses daily CT10 yield changes",
    )
    _try_view(
        b, f"100-bps-Wide Coupon Swap JPM Performance - Trailing {rv.STRUCTURE_RECENT_PLOT_WINDOW} Observations",
        "Coupon structures", "fig",
        lambda: _fig_structure_recent(
            payloads["wide_swap_data"], rv.WIDE_COUPON_SWAP_PAIRS,
            "{agency}_{high}_{low}_wide_swap_perf_32nds",
            "100-bps-Wide Coupon Swap JPM Performance", "100-bps-wide coupon swaps",
            "Cumulative swap Perf (32nds)",
        ),
    )
    for agency in rv.STRUCTURE_AGENCIES:
        _try_view(
            b, f"{agency} 100-bps-Wide Coupon Swap JPM Perf vs Daily CT10 Yield Change",
            "Coupon structures", "fig",
            lambda agency=agency: _fig_structure_ct10_rolling_corr(
                payloads["wide_swap_data"], payloads["wide_swap_summary"],
                "100-bps-Wide Coupon Swap JPM Perf", agency,
            ),
        )

    # Three-price flies
    fly_grad = _fly_gradient_cols()
    b.add_table(
        "Three-Price Coupon Fly Performance", "Coupon structures",
        _df_to_html_table(
            payloads["coupon_fly_display"],
            gradient_columns=fly_grad, diverging_columns=fly_grad, include_index=False,
        ),
        subtitle="Fly level = 2 x center price - lower price - upper price; values and changes in 32nds; "
                 "correlation uses daily fly-price changes vs daily CT10 yield changes",
    )
    _try_view(
        b, f"Coupon Fly Price Changes - Trailing {rv.STRUCTURE_RECENT_PLOT_WINDOW} Observations",
        "Coupon structures", "fig",
        lambda: _fig_coupon_fly_recent_price(payloads["coupon_fly_data"]),
    )
    for agency in rv.STRUCTURE_AGENCIES:
        _try_view(
            b, f"{agency} Three-Price Coupon Fly Changes vs Daily CT10 Yield Change",
            "Coupon structures", "fig",
            lambda agency=agency: _fig_structure_ct10_rolling_corr(
                payloads["coupon_fly_data"], payloads["coupon_fly_summary"],
                "Three-Price Coupon Fly Changes", agency,
            ),
        )

    _build_cc_pages(b, payloads)
    _build_oas_pages(b, payloads)
    _build_cb_pages(b, payloads)
    _build_appendix_pages(b, payloads)
    return b


def _build_cc_pages(b: _ReportBuilder, payloads: Dict) -> None:
    df_basis = payloads.get("df_basis")
    section = "Current-coupon basis"

    if payloads.get("cc_summary") is not None:
        b.add_table(
            "Current-Coupon Basis OLS Summary", section,
            _df_to_html_table(payloads["cc_summary"], gradient_columns=["diff", "resid_z"]),
            subtitle=payloads["cc_subtitle"],
        )
    if payloads.get("cc_params") is not None:
        b.add_table(
            "Current-Coupon Basis OLS Parameters", section,
            _df_to_html_table(
                payloads["cc_params"],
                gradient_columns=["coef", "p_value"], include_index=False,
            ),
            subtitle=payloads["cc_subtitle"],
        )
        if df_basis is not None:
            for response_col in ("FNCC", "G2CC"):
                if response_col in df_basis.columns:
                    name = {"FNCC": "Conventional", "G2CC": "Ginnie"}[response_col]
                    _try_view(
                        b,
                        f"{name} Current-Coupon Basis: Full-Sample Actual, Fitted and Residual",
                        section, "fig",
                        lambda response_col=response_col: _fig_cc_fitted(df_basis, response_col),
                    )
    if payloads.get("cc_rolling") is not None:
        for cc_type in ("Conventional", "Ginnie"):
            _try_view(
                b,
                f"{cc_type} Current-Coupon Basis: {rv.CC_ROLLING_WINDOW}-Day Rolling Actual, Fitted and Residual",
                section, "fig",
                lambda cc_type=cc_type: _fig_rolling_cc(
                    payloads["cc_rolling"], cc_type=cc_type, window=rv.CC_ROLLING_WINDOW
                ),
            )
        if df_basis is not None:
            _try_view(
                b, "Conv vs Ginnie Current-Coupon OAS Spread (90-day z-score)", section, "fig",
                lambda: _fig_conv_ginnie_cc_spread(df_basis, window=90),
            )
    if payloads.get("cc_ma_vol") is not None and df_basis is not None:
        b.add_table(
            "Current-Coupon Basis: Moving Averages & Realized Vol", section,
            _df_to_html_table(
                payloads["cc_ma_vol"],
                gradient_columns=[f"vol_{w}d" for w in rv.CC_VOL_WINDOWS],
                diverging_columns=[f"d_ma_{w}d" for w in rv.CC_MA_WINDOWS],
            ),
            subtitle=f"Latest: {df_basis.index[-1].date()}  |  d_ma = latest - MA  |  "
                     "realized vol = std(daily change), bps/day",
        )
        _try_view(
            b, "Current-Coupon Basis: Moving Averages and Realized Vol", section, "fig",
            lambda: _fig_cc_ma_vol(df_basis),
        )
    if payloads.get("cc_momentum_summary") is not None and df_basis is not None:
        mom_cols = ["chg_5d", "chg_20d", "chg_60d", "slope_20d", "slope_60d", "mom_20d", "mom_60d"]
        b.add_table(
            "Current-Coupon Basis Momentum Monitor", section,
            _df_to_html_table(
                payloads["cc_momentum_summary"],
                gradient_columns=mom_cols, diverging_columns=mom_cols,
            ),
            subtitle=f"Latest: {df_basis.index[-1].date()}  |  Positive = widening, negative = tightening  |  "
                     "Trend direction requires the 20- and 60-observation momentum scores to have the same sign",
        )
        _try_view(
            b, "Current-Coupon Basis Momentum Monitor (Charts)", section, "fig",
            lambda: _fig_cc_momentum(payloads["cc_momentum_history"], payloads["cc_momentum_summary"]),
        )
    if payloads.get("fn_10y_corr_summary") is not None:
        z_window = rv.CC_FN_10Y_CORR_Z_WINDOW
        b.add_table(
            "FN Basis vs UST 10y: Rolling Correlation Monitor", section,
            _df_to_html_table(
                payloads["fn_10y_corr_summary"],
                gradient_columns=[
                    "corr_now", "corr_5ago", "corr_20ago", f"z_now_{z_window}", "z_5ago", "z_20ago",
                ],
                diverging_columns=[
                    "corr_now", "corr_5ago", "d_corr_5", "corr_20ago", "d_corr_20",
                    f"z_now_{z_window}", "z_5ago", "z_20ago",
                ],
            ),
            subtitle=(
                f"Daily changes  |  Latest: {payloads['fn_10y_corr'].index[-1].date()}  |  "
                f"Corr z-score lookback: {z_window} observations"
            ),
        )
        _try_view(
            b, "FN Current-Coupon Basis vs UST 10y: Rolling Correlation of Daily Changes",
            section, "fig",
            lambda: _fig_fn_10y_rolling_corr(payloads["fn_10y_corr"], payloads["fn_10y_all_sample"]),
        )

    if payloads.get("cross_risk_summary") is not None:
        corr_vol_cols = _corr_vol_cols()
        diverging = [
            "latest", "corr_all", *[f"corr_{w}d" for w in rv.G2_FN_CORR_WINDOWS],
            "mean", "p05", "median", "p95",
        ]
        for family in FAMILY_ORDER:
            family_summary = payloads["cross_risk_summary"][
                payloads["cross_risk_summary"]["family"] == family
            ].drop(columns=["family"])
            if family_summary.empty:
                continue
            b.add_table(
                f"{family}: CT10 Correlation and Daily Volatility", FAMILY_GROUPS[family],
                _df_to_html_table(
                    family_summary,
                    gradient_columns=corr_vol_cols, diverging_columns=diverging,
                    include_index=False,
                ),
                subtitle="Correlation is versus daily CT10 yield change; all-sample and latest "
                         "20/40/60 paired observations. Volatility is the standard deviation "
                         "of the same daily series.",
            )
        for family in FAMILY_ORDER:
            family_series = [
                item for item in payloads["cross_risk_series"] if item["family"] == family
            ]
            page_count = (len(family_series) + 5) // 6
            for page_start in range(0, len(family_series), 6):
                page_items = family_series[page_start:page_start + 6]
                page_no = page_start // 6 + 1
                _try_view(
                    b, f"{family}: Daily Distribution ({page_no}/{page_count})",
                    FAMILY_GROUPS[family], "fig",
                    lambda family=family, page_items=page_items, page_no=page_no, page_count=page_count:
                        _fig_distribution_page(family, page_items, page_no, page_count),
                )


def _build_oas_pages(b: _ReportBuilder, payloads: Dict) -> None:
    oas_summary = payloads.get("oas_summary")
    if oas_summary is None or oas_summary.empty:
        return
    b.add_table(
        "FN / G2 TSY OAS vs 1-Year History Z-Score", "Performance",
        _df_to_html_table(oas_summary.reset_index(), gradient_columns=["z_score"], include_index=False),
        subtitle=payloads["oas_subtitle"],
    )


def _build_cb_pages(b: _ReportBuilder, payloads: Dict) -> None:
    if payloads.get("cb_summary") is None:
        return
    section = "Yield basis"
    cb_subtitle = payloads["cb_subtitle"]
    b.add_table(
        "Per-Coupon Yield Basis OLS Summary", section,
        _df_to_html_table(payloads["cb_summary"], gradient_columns=["diff", "resid_z"]),
        subtitle=cb_subtitle,
    )
    if payloads.get("cb_params") is not None:
        b.add_table(
            "Per-Coupon Yield Basis OLS Parameters", section,
            _df_to_html_table(
                payloads["cb_params"], gradient_columns=["coef", "p_value"], include_index=False
            ),
            subtitle=cb_subtitle,
        )
    cb_coef_matrix = payloads.get("cb_coef_matrix")
    if cb_coef_matrix is not None:
        if not cb_coef_matrix.empty and payloads.get("cb_feat_cols"):
            for feat, feat_cols in payloads["cb_feat_cols"].items():
                b.add_table(
                    f"Per-Coupon Yield Basis Coefficients — {feat}", section,
                    _df_to_html_table(cb_coef_matrix[feat_cols], gradient_columns=feat_cols),
                    subtitle=cb_subtitle,
                )
        for agency in rv.COUPON_BASIS_AGENCIES:
            _try_view(
                b, f"Per-Coupon Yield Basis OLS: {agency} MBS", section, "fig",
                lambda agency=agency: _fig_coupon_basis_fitted(payloads["cb_basis"], agency),
            )


def _build_appendix_pages(b: _ReportBuilder, payloads: Dict) -> None:
    dat = payloads["dat"]
    hist = payloads.get("hist_structures") or []
    adjacent: List[Tuple[str, tuple, tuple]] = []
    g2fn: List[Tuple[str, tuple, tuple]] = []
    flies: List[Tuple[str, tuple, tuple]] = []
    for mode, pair, triple in hist:
        if mode == "cpn":
            agencies = {leg.split()[0] for leg in pair}
            (adjacent if len(agencies) == 1 else g2fn).append((mode, pair, triple))
        elif mode == "fly":
            flies.append((mode, pair, triple))

    def hist_options(structures, labeler):
        options = []
        for mode, pair, triple in structures:
            try:
                options.append(
                    (labeler(pair, triple), _fig_historical_cpn_fly(dat, mode, pair, triple))
                )
            except Exception as exc:
                print(f"Interactive historical view '{labeler(pair, triple)}' skipped: {exc}")
        return options

    _try_view(
        b, "Historical Adjacent-Coupon Swaps (select structure)", "Appendix", "dropdown",
        hist_options(adjacent, lambda pair, _: f"{pair[0]}/{pair[1].split()[-1]}"),
    )
    _try_view(
        b, "Historical G2 vs FN Swaps (select structure)", "Appendix", "dropdown",
        hist_options(g2fn, lambda pair, _: f"{pair[0]} vs {pair[1]}"),
    )
    _try_view(
        b, "Historical Coupon Flies (select structure)", "Appendix", "dropdown",
        hist_options(flies, lambda _, triple: f"{triple[0]}/{triple[1]}/{triple[2]} fly"),
    )

    for kind, title in [
        ("drop", "Coupon Drops - Full History (select agency)"),
        ("sharpe", "Coupon Monthly Drop Sharpe (select agency)"),
        ("fwdvol", "Coupon 1y10y Forward-Vol Sharpe Proxy (select agency)"),
    ]:
        options = []
        for agency in rv.COUPON_BASIS_AGENCIES:
            try:
                fig = _fig_coupon_drop_grid(dat, agency, kind)
            except Exception as exc:
                print(f"Interactive appendix view '{title} {agency}' skipped: {exc}")
                fig = None
            if fig is not None:
                options.append((f"{AGENCY_NAMES.get(agency, agency)} ({agency})", fig))
        _try_view(b, title, "Appendix", "dropdown", options, label="Agency")


# ---------------------------------------------------------------------------
# HTML shell (ported from mbs_rv_report.HtmlPages._write_html)
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--panel:#fff;--bg:#eef2f6;--accent:#174a6e}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
.sidebar{position:fixed;inset:0 auto 0 0;width:290px;background:#102b3c;color:#e7f0f5;padding:22px 16px;overflow:auto}
.sidebar h1{font-size:19px;line-height:1.2;margin:0 0 7px}.subtitle{font-size:12px;color:#b9cad5;margin-bottom:16px}
#filter{width:100%;padding:9px 10px;border:1px solid #507084;border-radius:7px;background:#fff;color:#17202a;margin-bottom:14px}
.nav-group{margin:13px 0 19px}.nav-group h3{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8fc1dc;margin:0 7px 6px}
.nav-group a{display:block;color:#dce9f0;text-decoration:none;padding:6px 8px;border-radius:5px;font-size:12px}
.nav-group a:hover{background:#1e4b65;color:#fff}.nav-group a.hidden{display:none}
main{margin-left:290px;padding:22px;max-width:1700px}.topbar{display:flex;justify-content:space-between;align-items:center;margin:0 auto 16px;max-width:1380px}
.topbar strong{font-size:13px}.topbar button{border:1px solid #b6c5cf;background:#fff;padding:7px 10px;border-radius:6px;cursor:pointer}
.report-page{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 3px 12px rgba(15,23,42,.06);margin:0 auto 22px;max-width:1380px;scroll-margin-top:12px;overflow:hidden}
.page-heading{display:flex;align-items:center;gap:11px;padding:12px 15px;border-bottom:1px solid var(--line)}.page-heading h2{font-size:15px;margin:0;flex:1}
.page-number{font-size:11px;font-weight:bold;color:#fff;background:var(--accent);border-radius:12px;padding:3px 7px}
.page-body{padding:12px 15px}.report-page.hidden{display:none}
.chart{width:100%}
.view-subtitle{font-size:12px;color:var(--muted);margin:2px 0 10px}
.table-wrap{overflow:auto;max-height:82vh;border:1px solid var(--line);border-radius:6px}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 7px;position:sticky;top:0;font-weight:bold;text-align:center;white-space:nowrap;z-index:1}
.data-table td{border:1px solid #e2e8f0;padding:3px 7px;text-align:center;white-space:nowrap}
.data-table tbody tr:nth-child(even){background:#f0f0f0}
.dropdown-bar{margin:2px 0 12px;font-size:13px}
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:240px;background:#fff}
.overview h1{font-size:24px;margin:8px 0 4px;text-align:center}
.overview .ov-sub{text-align:center;color:var(--muted);margin:0 0 18px}
.ov-box{border:1px solid #40466e;border-radius:8px;background:#f7f8fc;padding:14px 18px;margin:0 auto 16px;max-width:900px}
.ov-box h3{margin:0 0 8px;font-size:14px}
.ov-box p{font-size:12px;color:var(--muted);margin:0 0 10px}
.ov-box ol{font-size:13px;margin:0;padding-left:22px;columns:2;column-gap:36px}
@media(max-width:850px){.sidebar{position:relative;width:auto;max-height:46vh}main{margin-left:0;padding:10px}.topbar{padding:0 4px}.ov-box ol{columns:1}}
@media print{.sidebar,.topbar{display:none}main{margin:0;padding:0}.report-page{break-after:page;box-shadow:none;border:0}}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
document.querySelectorAll('.chart:not(.dd-fig)').forEach(el=>renderChart(el.id));
document.querySelectorAll('.dd-fig').forEach(el=>{if(el.style.display!=='none')renderChart(el.id);});
document.querySelectorAll('select.dd-select').forEach(sel=>{sel.addEventListener('change',()=>{
document.querySelectorAll('.dd-fig[data-select="'+sel.id+'"]').forEach(d=>{const on=d.dataset.value===sel.value;d.style.display=on?'':'none';if(on)renderChart(d.id);});});});
const filter=document.getElementById('filter');
const pages=[...document.querySelectorAll('.report-page')];
const links=[...document.querySelectorAll('.nav-group a')];
filter.addEventListener('input',()=>{const q=filter.value.trim().toLowerCase();let n=0;pages.forEach(p=>{const show=!q||p.dataset.search.includes(q);p.classList.toggle('hidden',!show);if(show)n++;});links.forEach(a=>a.classList.toggle('hidden',q&&!a.dataset.search.includes(q)));document.getElementById('visibleCount').textContent=`${n} views`;});
document.getElementById('topButton').addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));
"""


def _write_html(report_path: Path, payloads: Dict, builder: _ReportBuilder) -> None:
    dat = payloads["dat"]
    pages = builder.pages

    grouped: Dict[str, List[Dict[str, str]]] = {name: [] for name in NAV_GROUP_ORDER}
    for page in pages:
        grouped.setdefault(page["section"], []).append(page)

    nav_parts = []
    for section in NAV_GROUP_ORDER:
        group_pages = grouped.get(section) or []
        if not group_pages:
            continue
        links = "".join(
            f'<a href="#{page["id"]}" data-search="{html.escape(page["search"])}">'
            f'<span>{html.escape(page["title"])}</span></a>'
            for page in group_pages
        )
        nav_parts.append(f'<div class="nav-group"><h3>{html.escape(section)}</h3>{links}</div>')

    page_parts = []
    for number, page in enumerate(pages, start=1):
        title = html.escape(page["title"])
        page_parts.append(
            f'<section class="report-page" id="{page["id"]}" '
            f'data-search="{html.escape(page["search"])}">'
            f'<div class="page-heading"><span class="page-number">{number:02d}</span>'
            f"<h2>{title}</h2></div>"
            f'<div class="page-body">{page["body"]}</div>'
            f"</section>"
        )

    subtitle = f"Latest data {dat.index[-1].date()} · {len(dat):,} observations"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(REPORT_TITLE)} (Interactive)</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<aside class="sidebar"><h1>{html.escape(REPORT_TITLE)}</h1><div class="subtitle">{html.escape(subtitle)} · {len(pages)} views (interactive)</div>
<input id="filter" type="search" placeholder="Filter sections and charts" aria-label="Filter report">
<nav>{''.join(nav_parts)}</nav></aside>
<main><div class="topbar"><strong id="visibleCount">{len(pages)} views</strong><button id="topButton">Back to top</button></div>{''.join(page_parts)}</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_interactive_report(payloads: Dict, report_path: Optional[Path] = None) -> Path:
    """Build the interactive Plotly HTML report from assembled payloads."""
    if report_path is None:
        report_path = rv._default_report_path("MBS_RV_Report_Interactive", "html")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    builder = _build_pages(payloads)
    _write_html(report_path, payloads, builder)
    print(f"Interactive report saved to: {report_path}")
    return report_path
