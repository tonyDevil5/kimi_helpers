"""Interactive Plotly HTML report for the UST fair-value monitor.

Consumes the payload dict assembled by ``ust_fair_value_report._assemble_report_data``
and writes a single self-contained HTML file: plotly.js is inlined, every
figure is embedded as JSON, and the sidebar + filter UX mirrors the static
HTML report produced by ``mbs_rv_report.HtmlPages`` (ported via
``mbs_rv_interactive``).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

import ust_fair_value_report as ufr

REPORT_TITLE = "UST 10Y Fair Value OLS Regression Report"

NAV_GROUP_ORDER = [
    "Overview",
    "10Y fair value model",
    "Rolling fair value",
    "Curve spreads",
    "Butterfly spreads",
]


# ---------------------------------------------------------------------------
# Small conversion / layout helpers (ported from mbs_rv_interactive)
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _assert_aligned_series(actual: pd.Series, fitted: pd.Series, label: str) -> None:
    """Fail fast instead of drawing a plausible-looking, misaligned chart."""
    if len(actual) != len(fitted):
        raise ValueError(
            f"{label}: actual/fitted length mismatch ({len(actual)} vs {len(fitted)})"
        )
    if not actual.index.equals(fitted.index):
        raise ValueError(f"{label}: actual/fitted date indexes are not identical")
    if not actual.index.is_monotonic_increasing:
        raise ValueError(f"{label}: dates are not sorted in ascending order")


def _observed_trace(index, values, name: str = "Observed") -> go.Scatter:
    return go.Scatter(
        x=_xs(index),
        y=_ys(values),
        mode="lines",
        name=name,
        line=dict(color="red", width=1.4),
        opacity=0.75,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Observed: %{y:.2f}<extra></extra>",
    )


def _fitted_trace(index, values, name: str = "Model fitted") -> go.Scatter:
    return go.Scatter(
        x=_xs(index),
        y=_ys(values),
        mode="lines",
        name=name,
        line=dict(color="blue", width=1.4),
        opacity=0.75,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Model fitted: %{y:.2f}<extra></extra>",
    )


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
    bgcolor: str = "wheat",
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
        opacity=0.85,
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
        hovermode="closest",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def _add_rangeslider(fig: go.Figure) -> None:
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Heatmap tables (mirror ust_fair_value_report._df_to_figure coloring)
# ---------------------------------------------------------------------------
def _df_to_html_table(
    df: pd.DataFrame,
    gradient_columns: Optional[List[str]] = None,
    include_index: bool = True,
) -> str:
    """Render a DataFrame as an HTML table with the same heatmap as the PDF.

    Uses ``ust_fair_value_report._column_color_func`` per gradient column, so
    the diverging/sequential scales and ranges match the PDF exactly. Floats
    are formatted to three decimals, as in ``_df_to_figure``.
    """
    if include_index:
        df = df.reset_index()
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    gradient_columns = gradient_columns or []
    color_funcs = {}
    for gc in gradient_columns:
        if gc in df.columns:
            color_funcs[gc] = ufr._column_color_func(df[gc])

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
                    style = f' style="background:{color_funcs[col](raw)}"'
            parts.append(f"<td{style}>{html.escape(str(display[col].iloc[i]))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chart builders: 10Y fair-value model
# ---------------------------------------------------------------------------
def _fig_fitted_vs_actual(payloads: Dict) -> go.Figure:
    """Full-sample view matching the MBS RV fitted-plus-residual layout."""
    model = payloads["model"]
    dat_final = payloads["dat_final"]
    summary = payloads["summary"]
    actual = dat_final.iloc[:, ufr.RESPONSE_IDX]
    fitted = pd.Series(model.fittedvalues, index=actual.index)
    _assert_aligned_series(actual, fitted, "10Y fair-value chart")

    ratio = (
        summary["latest_residual"] / summary["resid_std"]
        if summary["resid_std"] != 0
        else float("nan")
    )
    annot_text = (
        f"Latest observed: {summary['latest_actual']:.3f}%\n"
        f"Latest fitted: {summary['latest_fitted']:.3f}%\n"
        f"Residual: {summary['latest_residual'] * 100:+.2f} bps  |  z = {ratio:.2f}"
    )

    residual_bps = (actual - fitted) * 100.0
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.67, 0.33], vertical_spacing=0.06,
    )
    fig.add_trace(_observed_trace(actual.index, actual, "Observed 10Y yield"), row=1, col=1)
    fig.add_trace(_fitted_trace(fitted.index, fitted, "Full-sample fitted 10Y yield"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=_xs(residual_bps.index), y=_ys(residual_bps), mode="lines",
            fill="tozeroy", name="Residual = observed − fitted",
            line=dict(color="slateblue", width=1),
            fillcolor="rgba(106, 90, 205, 0.15)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Residual: %{y:+.2f} bps<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="10Y yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    _add_annotation(
        fig, annot_text, "x", "y", x=0.02, y=0.98, align="left", valign="top",
        fontsize=11,
    )
    _apply_layout(
        fig, "10Y UST Fair Value: Full-Sample Observed, Fitted and Residual",
        height=760, subtitle=payloads["subtitle"] + "  |  Residual = observed − fitted",
    )
    return fig


def _fig_native_rolling_fair_value(rolling_df: pd.DataFrame, window: int) -> go.Figure:
    """MBS-style rolling observed/fitted chart with raw residual beneath it."""
    if rolling_df.empty:
        raise ValueError("No rolling fair-value observations available")
    rolling_df = rolling_df.sort_index()
    latest = rolling_df.iloc[-1]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.67, 0.33], vertical_spacing=0.06,
    )
    fig.add_trace(
        _observed_trace(rolling_df.index, rolling_df["mkt_val"], "Observed 10Y yield"),
        row=1, col=1,
    )
    fig.add_trace(
        _fitted_trace(
            rolling_df.index, rolling_df["mdl_val"],
            f"{window}-observation rolling fitted yield",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=_xs(rolling_df.index), y=_ys(rolling_df["resid_bps"]), mode="lines",
            fill="tozeroy", name="Residual = observed − fitted",
            line=dict(color="slateblue", width=1),
            fillcolor="rgba(106, 90, 205, 0.15)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Residual: %{y:+.2f} bps<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(0, line=dict(color="black", width=0.8), row=2, col=1)
    fig.update_yaxes(title_text="10Y yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
    _add_annotation(
        fig,
        f"Latest observed: {latest['mkt_val']:.3f}%\n"
        f"Latest fitted: {latest['mdl_val']:.3f}%\n"
        f"Residual: {latest['resid_bps']:+.2f} bps  |  z = {latest['resid_z']:.2f}",
        "x", "y", fontsize=10,
    )
    _apply_layout(
        fig,
        f"10Y UST Fair Value: {window}-Observation Rolling Observed, Fitted and Residual",
        height=760,
        subtitle=(
            f"Rolling estimates: {rolling_df.index[0].date()} to "
            f"{rolling_df.index[-1].date()}  |  Window = {window} observations  |  "
            "Same response and predictors as the full-sample OLS; intercept included"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart builders: rolling fair value (USTDurationStrategy)
# ---------------------------------------------------------------------------
def _fig_rolling_mkt_vs_mdl(rolling_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_rolling_mkt_vs_mdl."""
    latest = rolling_df.iloc[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(rolling_df.index), y=_ys(rolling_df["mkt_val"]), mode="lines",
            name="Market 10Y Yield", line=dict(color="red", width=1.4), opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_xs(rolling_df.index), y=_ys(rolling_df["mdl_val"]), mode="lines",
            name="Model Fair Value", line=dict(color="blue", width=1.4), opacity=0.7,
        )
    )
    fig.update_yaxes(title_text="Yield (%)")
    _add_annotation(
        fig,
        f"Latest: mkt={latest['mkt_val']:.3f}, mdl={latest['mdl_val']:.3f}, "
        f"z={latest['resid_z']:.2f}",
        "x", "y", x=0.02, y=0.98, align="left", valign="top", fontsize=11,
    )
    _apply_layout(fig, "Rolling Market vs Model 10Y UST Yield", height=640)
    _add_rangeslider(fig)
    return fig


def _fig_rolling_residual(rolling_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_rolling_residual."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(rolling_df.index), y=_ys(rolling_df["resid_z"]), mode="lines",
            name="Residual z-score", line=dict(color="purple", width=1.4), opacity=0.85,
        )
    )
    fig.add_hline(0, line=dict(color="black", width=0.8))
    fig.add_hline(1.25, line=dict(color="red", width=0.8, dash="dash"))
    fig.add_hline(-1.25, line=dict(color="red", width=0.8, dash="dash"))
    # Legend entry for the open-threshold band (mirrors the matplotlib legend)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", name="open threshold",
            line=dict(color="red", width=0.8, dash="dash"),
        )
    )
    fig.update_yaxes(title_text="Residual / In-Sample Std")
    _apply_layout(
        fig,
        f"Rolling Fair-Value Residual Z-Score ({ufr.ROLLING_WINDOW}-day rolling OLS)",
        height=560,
    )
    _add_rangeslider(fig)
    return fig


def _fig_recent_rolling_mkt_vs_mdl(recent_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_recent_rolling_mkt_vs_mdl."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(recent_df.index), y=_ys(recent_df["mkt_val"]), mode="lines",
            name="Market 10Y Yield", line=dict(color="red", width=1.4), opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_xs(recent_df.index), y=_ys(recent_df["mdl_val"]), mode="lines",
            name="Model Fair Value", line=dict(color="blue", width=1.4), opacity=0.7,
        )
    )
    fig.update_yaxes(title_text="Yield (%)")
    _apply_layout(
        fig,
        "Recent Rolling Market vs Model 10Y UST Yield",
        height=640,
        subtitle=(
            f"current mkt={recent_df['mkt_val'].iloc[-1]:.3f}, "
            f"mdl={recent_df['mdl_val'].iloc[-1]:.3f}, "
            f"resid ratio={recent_df['resid_z'].iloc[-1]:.3f}  |  "
            f"{ufr.ROLLING_WINDOW}-day rolling window"
        ),
    )
    _add_rangeslider(fig)
    return fig


def _fig_recent_rolling_residual(recent_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_recent_rolling_residual."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(recent_df.index), y=_ys(recent_df["resid_bps"]), mode="lines",
            name="residual (bps)", line=dict(color="purple", width=1.4), opacity=0.85,
        )
    )
    fig.add_hline(0, line=dict(color="black", width=0.8))
    fig.update_yaxes(title_text="Residual (bps)")
    _apply_layout(
        fig,
        f"Recent Rolling Fair-Value Residual ({ufr.ROLLING_WINDOW}-day rolling window)",
        height=560,
    )
    _add_rangeslider(fig)
    return fig


# ---------------------------------------------------------------------------
# Chart builders: curve spreads and butterflies (fitted vs actual grids)
# ---------------------------------------------------------------------------
def _grid_panel_annotation(fig: go.Figure, idx: int, y_val: float, mdl_val: float, resid_std: float) -> None:
    diff = y_val - mdl_val
    ratio = diff / resid_std if resid_std != 0 else float("nan")
    xname, yname = _subplot_axis(idx)
    _add_annotation(
        fig,
        f"mkt={y_val:.1f}\nmdl={mdl_val:.1f}\ndiff={diff:.1f}\nratio={ratio:.2f}",
        xname, yname, fontsize=9,
    )


def _fig_all_curves(merged_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_all_curves (2x3 grid)."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[name for _, _, name in ufr.CURVE_PAIRS],
        vertical_spacing=0.12, horizontal_spacing=0.06,
    )
    import numpy as np

    for i, (crv1, crv2, name) in enumerate(ufr.CURVE_PAIRS):
        r, c = i // 3 + 1, i % 3 + 1
        Y, model = ufr.run_curve_ols(merged_df, crv1, crv2)
        fitted = pd.Series(model.fittedvalues, index=Y.index)
        _assert_aligned_series(Y, fitted, f"{name} curve chart")
        show_legend = i == 0
        fig.add_trace(
            _observed_trace(Y.index, Y),
            row=r, col=c,
        )
        fig.data[-1].update(legendgroup="observed", showlegend=show_legend, line_width=1.1)
        fig.add_trace(
            _fitted_trace(fitted.index, fitted),
            row=r, col=c,
        )
        fig.data[-1].update(legendgroup="fitted", showlegend=show_legend, line_width=1.1)
        _grid_panel_annotation(
            fig, i + 1, float(Y.iloc[-1]),
            float(fitted.iloc[-1]), float(np.std(model.resid)),
        )
    fig.update_yaxes(title_text="Spread (bps)", zeroline=True, zerolinecolor="#94a3b8")
    _apply_layout(
        fig,
        "UST Curve Spreads: Observed vs Model Fitted",
        height=820,
        subtitle="Observed spread = long-end yield minus short-end yield, in basis points",
    )
    return fig


def _fig_all_flies(merged_df: pd.DataFrame) -> go.Figure:
    """Mirror ust_fair_value_report.plot_all_flies (1x3 grid)."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[name for _, _, _, name in ufr.FLY_TRIPLES],
        horizontal_spacing=0.06,
    )
    import numpy as np

    for i, (short, belly, long, name) in enumerate(ufr.FLY_TRIPLES):
        Y, model = ufr.run_fly_ols(merged_df, short, belly, long)
        fitted = pd.Series(model.fittedvalues, index=Y.index)
        _assert_aligned_series(Y, fitted, f"{name} butterfly chart")
        show_legend = i == 0
        fig.add_trace(
            _observed_trace(Y.index, Y),
            row=1, col=i + 1,
        )
        fig.data[-1].update(legendgroup="observed", showlegend=show_legend, line_width=1.1)
        fig.add_trace(
            _fitted_trace(fitted.index, fitted),
            row=1, col=i + 1,
        )
        fig.data[-1].update(legendgroup="fitted", showlegend=show_legend, line_width=1.1)
        _grid_panel_annotation(
            fig, i + 1, float(Y.iloc[-1]),
            float(fitted.iloc[-1]), float(np.std(model.resid)),
        )
    fig.update_yaxes(title_text="Butterfly (bps)", zeroline=True, zerolinecolor="#94a3b8")
    _apply_layout(
        fig,
        "UST Butterfly Spreads: Observed vs Model Fitted",
        height=520,
        subtitle="Observed butterfly = 2 × belly yield − short-end yield − long-end yield, in basis points",
    )
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


def _overview_body(payloads: Dict) -> str:
    dat_final = payloads["dat_final"]
    latest = ufr._idx_date(dat_final.index[-1])
    summary = payloads["summary"]

    stats_rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("R-squared", f"{summary['r_squared']:.4f}"),
            ("Adj R-squared", f"{summary['adj_r_squared']:.4f}"),
            ("F-statistic", f"{summary['f_stat']:.2f}"),
            ("F p-value", f"{summary['f_pvalue']:.4e}"),
            ("Residual Std", f"{summary['resid_std'] * 100:.2f} bps"),
            ("Latest Observed", f"{summary['latest_actual']:.3f}%"),
            ("Latest Fitted", f"{summary['latest_fitted']:.3f}%"),
            (
                "Latest Residual",
                f"{summary['latest_residual'] * 100:+.2f} bps ({summary['latest_resid_z']:.2f} std)",
            ),
        ]
    )

    contents_items = [
        "Full-sample OLS regression parameters and statistics",
        "Full-sample observed, fitted and residual 10Y UST yield",
    ]
    rolling_120_df = payloads.get("rolling_120_df")
    if rolling_120_df is not None and not rolling_120_df.empty:
        contents_items.append(
            f"{ufr.FAIR_VALUE_ROLLING_WINDOW}-observation rolling observed, fitted and residual chart"
        )
    rolling_df = payloads.get("rolling_df")
    if rolling_df is not None and not rolling_df.empty:
        contents_items += [
            "Rolling market vs model value chart",
            "Rolling residual z-score chart",
        ]
    recent_df = payloads.get("recent_df")
    if recent_df is not None and not recent_df.empty:
        contents_items += [
            "Recent rolling market vs model chart",
            "Recent rolling residual (bps) chart",
        ]
    if payloads.get("curve_summary") is not None:
        contents_items += [
            "UST curve-spread fair-value summary table",
            "Curve-spread actual vs fitted grid (2s5s, 2s10s, 5s10s, 5s30s, 7s30s, 10s30s)",
        ]
    if payloads.get("fly_summary") is not None:
        contents_items += [
            "UST butterfly fair-value summary table",
            "Butterfly actual vs fitted grid (2s5s10s, 5s10s30s, 10s20s30s)",
        ]
    contents_html = "<ol>" + "".join(f"<li>{html.escape(item)}</li>" for item in contents_items) + "</ol>"

    return (
        f'<div class="overview"><h1>{REPORT_TITLE}</h1>'
        f'<p class="ov-sub">Latest date: {latest}  |  Observations: {len(dat_final):,}</p>'
        f'<div class="ov-box"><h3>Key Regression Statistics</h3>'
        f'<table class="stats-table">{stats_rows}</table></div>'
        f'<div class="ov-box"><h3>Report contents</h3>{contents_html}</div></div>'
    )


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

    # Overview / contents
    b.add_html(f"{REPORT_TITLE} - Contents & Key Stats", "Overview", _overview_body(payloads))

    # Full-sample 10Y fair-value model
    section = "10Y fair value model"
    if payloads.get("params_table") is not None:
        _try_view(
            b, "Full-Sample OLS Regression Parameters", section, "table",
            _df_to_html_table(payloads["params_table"], gradient_columns=["coef", "p_value"]),
            subtitle=payloads.get("subtitle"),
        )
    if payloads.get("stats_table") is not None:
        _try_view(
            b, "Regression Statistics", section, "table",
            _df_to_html_table(payloads["stats_table"], gradient_columns=["value"]),
            subtitle=payloads.get("subtitle"),
        )
    if payloads.get("model") is not None:
        _try_view(
            b, "Full-Sample Observed, Fitted and Residual 10Y UST Yield", section, "fig",
            lambda: _fig_fitted_vs_actual(payloads),
        )

    # Native rolling fair value, using the same OLS specification as full sample.
    section = "Rolling fair value"
    rolling_120_df = payloads.get("rolling_120_df")
    if rolling_120_df is not None and not rolling_120_df.empty:
        _try_view(
            b,
            f"{ufr.FAIR_VALUE_ROLLING_WINDOW}-Observation Rolling Observed, Fitted and Residual",
            section,
            "fig",
            lambda: _fig_native_rolling_fair_value(
                rolling_120_df, ufr.FAIR_VALUE_ROLLING_WINDOW
            ),
        )

    # Optional legacy strategy-based rolling views.
    rolling_df = payloads.get("rolling_df")
    if rolling_df is not None and not rolling_df.empty:
        _try_view(
            b, "Rolling Market vs Model 10Y UST Yield", section, "fig",
            lambda: _fig_rolling_mkt_vs_mdl(rolling_df),
        )
        _try_view(
            b,
            f"Rolling Fair-Value Residual Z-Score ({ufr.ROLLING_WINDOW}-day rolling OLS)",
            section, "fig",
            lambda: _fig_rolling_residual(rolling_df),
        )
    recent_df = payloads.get("recent_df")
    if recent_df is not None and not recent_df.empty:
        _try_view(
            b, "Recent Rolling Market vs Model 10Y UST Yield", section, "fig",
            lambda: _fig_recent_rolling_mkt_vs_mdl(recent_df),
        )
        _try_view(
            b,
            f"Recent Rolling Fair-Value Residual ({ufr.ROLLING_WINDOW}-day rolling window)",
            section, "fig",
            lambda: _fig_recent_rolling_residual(recent_df),
        )

    # Curve spreads
    if payloads.get("curve_summary") is not None:
        section = "Curve spreads"
        _try_view(
            b, "UST Curve Spread Fair-Value Summary", section, "table",
            _df_to_html_table(payloads["curve_summary"], gradient_columns=["diff", "ratio"]),
            subtitle=payloads.get("curve_subtitle"),
        )
        _try_view(
            b, "UST Curve Spreads: Actual vs Model Fitted", section, "fig",
            lambda: _fig_all_curves(payloads["merged_df"]),
        )

    # Butterflies
    if payloads.get("fly_summary") is not None:
        section = "Butterfly spreads"
        _try_view(
            b, "UST Butterfly Spread Fair-Value Summary", section, "table",
            _df_to_html_table(
                payloads["fly_summary"], gradient_columns=["diff", "ratio", "beta_10y"]
            ),
            subtitle=payloads.get("fly_subtitle"),
        )
        _try_view(
            b, "UST Butterfly Spreads: Actual vs Model Fitted", section, "fig",
            lambda: _fig_all_flies(payloads["merged_df"]),
        )

    return b


# ---------------------------------------------------------------------------
# HTML shell (ported from mbs_rv_interactive)
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
.stats-table{border-collapse:collapse;font-size:13px;font-family:Consolas,monospace}
.stats-table td{padding:3px 16px 3px 0}
.stats-table td:first-child{color:var(--muted)}
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
    dat_final = payloads["dat_final"]
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

    subtitle = (
        f"Latest data {ufr._idx_date(dat_final.index[-1])} · "
        f"{len(dat_final):,} observations"
    )
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
        report_path = ufr._default_report_path("UST_FairValue_Report_Interactive", "html")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    builder = _build_pages(payloads)
    _write_html(report_path, payloads, builder)
    print(f"Interactive report saved to: {report_path}")
    return report_path
