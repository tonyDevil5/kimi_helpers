"""Interactive Plotly HTML report for the DM GOVY monitor.

Consumes the payload dict assembled by ``dm_govy_report._assemble_report_data``
and writes a single self-contained HTML file: plotly.js is inlined, every
figure is embedded as JSON, and the sidebar + filter UX mirrors the MBS RV
interactive report (``mbs_rv_interactive``), whose shell helpers are reused
here.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import dm_govy_report as dg
from mbs_rv_interactive import (
    _CSS,
    _JS,
    _add_annotation,
    _add_rangeslider,
    _apply_layout,
    _subplot_axis,
    _xs,
    _ys,
    get_plotlyjs,
)

REPORT_TITLE = "DM Government Bond Performance Comparison"

NAV_GROUP_ORDER = [
    "Overview",
    "Summary tables",
    "Spread analysis",
    "Yield curves",
    "Quantiles",
]

YIELD_CURVE_COLORS = {
    "us": "blue",
    "frf": "red",
    "dem": "green",
    "jpy": "orange",
    "uk": "purple",
}
SPREAD_COLORS = {"GT10 - Bund10": "green", "GT10 - OAT10Y": "red", "GT10 - Bono10Y": "orange"}


# ---------------------------------------------------------------------------
# Heatmap tables (mirrors dm_govy_report._df_to_figure cell coloring)
# ---------------------------------------------------------------------------
def _df_to_html_table(
    df: pd.DataFrame,
    gradient_columns: Optional[List[str]] = None,
    include_index: bool = True,
) -> str:
    """Render a DataFrame as an HTML table with the same heatmap as the PDF."""
    if include_index:
        df = df.reset_index()
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    color_funcs = {}
    for gc in gradient_columns or []:
        if gc in df.columns:
            color_funcs[gc] = dg._column_color_func(df[gc])

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
# Chart builders (mirror the matplotlib pages in dm_govy_report)
# ---------------------------------------------------------------------------
def _fig_yield_curves(df: pd.DataFrame, countries: List[str]) -> go.Figure:
    """Plotly version of dm_govy_report.plot_yield_curves."""
    tenors_num = [2, 5, 7, 10, 20, 30]
    fig = go.Figure()
    for country in countries:
        yields = []
        for tenor in dg.TENORS:
            col = f"{country}_{tenor}"
            yields.append(float(df[col].iloc[-1]) if col in df.columns else None)
        fig.add_trace(
            go.Scatter(
                x=tenors_num,
                y=yields,
                mode="lines+markers",
                name=country.upper(),
                line=dict(color=YIELD_CURVE_COLORS.get(country, "gray"), width=1.6),
                marker=dict(size=7),
                connectgaps=False,
            )
        )
    _apply_layout(fig, "DM Government Yield Curves (Latest)", height=640)
    fig.update_xaxes(title_text="Tenor (years)", tickvals=tenors_num)
    fig.update_yaxes(title_text="Yield (%)")
    return fig


def _fig_spread_history_1y(df: pd.DataFrame) -> go.Figure:
    """Plotly version of dm_govy_report.plot_spread_history_1y."""
    required = {"us_10y", "dem_10y", "frf_10y", "esp_10y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for spread history plot: {missing}")

    recent = df.tail(252)
    spreads = {
        "GT10 - Bund10": recent["us_10y"] - recent["dem_10y"],
        "GT10 - OAT10Y": recent["us_10y"] - recent["frf_10y"],
        "GT10 - Bono10Y": recent["us_10y"] - recent["esp_10y"],
    }

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"{name} 1-Year History" for name in spreads],
    )
    for r, (name, spread) in enumerate(spreads.items(), start=1):
        color = SPREAD_COLORS.get(name, "blue")
        spread_bps = (spread * 100).dropna()
        fig.add_trace(
            go.Scatter(
                x=_xs(spread_bps.index), y=_ys(spread_bps), mode="lines",
                name=name, line=dict(color=color, width=1.4), showlegend=False,
            ),
            row=r, col=1,
        )
        fig.add_hline(0, line=dict(color="black", width=0.8, dash="dash"), row=r, col=1)

        min_val = float(spread_bps.min())
        max_val = float(spread_bps.max())
        min_date = spread_bps.idxmin()
        max_date = spread_bps.idxmax()
        for level in (min_val, max_val):
            fig.add_hline(
                level, line=dict(color=color, width=0.8, dash="dot"), opacity=0.7, row=r, col=1
            )
        fig.add_trace(
            go.Scatter(
                x=_xs([min_date]), y=[min_val], mode="markers",
                marker=dict(color=color, size=8, symbol="triangle-down"),
                showlegend=False, hoverinfo="skip",
            ),
            row=r, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_xs([max_date]), y=[max_val], mode="markers",
                marker=dict(color=color, size=8, symbol="triangle-up"),
                showlegend=False, hoverinfo="skip",
            ),
            row=r, col=1,
        )

        # Tight y-axis: from min to max with small padding only for the zero line
        pad = (max_val - min_val) * 0.02 if max_val != min_val else 1.0
        fig.update_yaxes(range=[min_val - pad, max_val + pad], title_text="Spread (bps)", row=r, col=1)

        xname, yname = _subplot_axis(r)
        _add_annotation(
            fig,
            f"Latest: {float(spread_bps.iloc[-1]):.1f} bps\n"
            f"Min: {min_val:.1f} bps ({dg._idx_date(min_date):%Y-%m-%d})\n"
            f"Max: {max_val:.1f} bps ({dg._idx_date(max_date):%Y-%m-%d})",
            xname, yname, x=0.02, y=0.95, align="left", valign="top",
            fontsize=10, bgcolor="wheat",
        )
    _apply_layout(fig, "GT10 vs Bund / OAT / Bono 10Y Spreads — 1-Year History", height=860)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    _add_rangeslider(fig, row=3, col=1)
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

    def add_dropdown(
        self,
        title: str,
        section: str,
        options: List[Tuple[str, str]],
        label: str = "Country",
        subtitle: Optional[str] = None,
        search_extra: str = "",
    ) -> None:
        """Dropdown of pre-rendered HTML blocks (tables), shown one at a time.

        Uses the same ``.dd-fig`` / ``data-select`` mechanism as the Plotly
        dropdowns in the shared JS, so no extra scripting is required.
        """
        if not options:
            return
        sel_id = f"sel-{len(self.pages) + 1:03d}"
        option_tags = "".join(
            f'<option value="{i}">{html.escape(text)}</option>' for i, (text, _) in enumerate(options)
        )
        parts = []
        if subtitle:
            parts.append(f'<p class="view-subtitle">{html.escape(subtitle)}</p>')
        parts.append(
            f'<div class="dropdown-bar"><label>{html.escape(label)}: '
            f'<select class="dd-select" id="{sel_id}">{option_tags}</select></label></div>'
        )
        for i, (_, block_html) in enumerate(options):
            hidden = "" if i == 0 else ' style="display:none"'
            block_id = self._next_fig_id()
            parts.append(
                f'<div class="dd-fig" id="{block_id}" data-select="{sel_id}" '
                f'data-value="{i}"{hidden}>{block_html}</div>'
            )
        search = f"{search_extra} {' '.join(text for text, _ in options)}".strip()
        self._add(title, section, "".join(parts), search)


def _overview_body(payloads: Dict) -> str:
    df = payloads["df"]
    latest = dg._idx_date(df.index[-1])
    summary = payloads["summary"]

    movers_html = ""
    if "chg_1m" in summary.columns:
        chg_1m = summary["chg_1m"].dropna().sort_values()
        worst = chg_1m.head(3)
        best = chg_1m.tail(3)
        movers_html = (
            "<p>Largest Yield Risers (1m, bps)<br>"
            + "<br>".join(f"&nbsp;&nbsp;{idx}: +{val:.1f}" for idx, val in best.iloc[::-1].items())
            + "<br><br>Largest Yield Fallers (1m, bps)<br>"
            + "<br>".join(f"&nbsp;&nbsp;{idx}: {val:.1f}" for idx, val in worst.items())
            + "</p>"
        )

    contents_items = [
        "Compact DM bond summary tables by tenor group "
        "(yield, 1d/1w/1m/3m/6m/1y changes, RSI, BB position, momentum, carry)",
        "GT10 vs 10Y Bund / OAT / Bono spread summary table",
        "1-year GT10 spread history charts",
        "DM yield curve snapshot",
        f"{dg.QUANTILE_WINDOW}-day yield quantiles by country "
        "(current yield vs min/25%/50%/75%/max)",
    ]
    contents_html = "<ol>" + "".join(f"<li>{html.escape(item)}</li>" for item in contents_items) + "</ol>"
    return (
        f'<div class="overview"><h1>{REPORT_TITLE}</h1>'
        f'<p class="ov-sub">Latest date: {latest}  |  Observations: {len(df):,}</p>'
        f'<div class="ov-box"><h3>Top Movers (1m yield change)</h3>{movers_html}</div>'
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
    df = payloads["df"]

    # Overview / contents
    try:
        b.add_html(f"{REPORT_TITLE} — Contents & Top Movers", "Overview", _overview_body(payloads))
    except Exception as exc:
        print(f"Interactive overview view skipped: {exc}")

    # Summary tables split by tenor group
    for group_name, sub in payloads.get("tenor_groups") or []:
        _try_view(
            b, f"DM Government Bonds — {group_name}", "Summary tables", "table",
            _df_to_html_table(sub, gradient_columns=payloads["gradient_cols"]),
            subtitle=payloads["subtitle"],
        )

    # GT10 spread summary table + 1-year history chart
    if payloads.get("spread_summary") is not None:
        _try_view(
            b, "GT10 Spread Analysis", "Spread analysis", "table",
            _df_to_html_table(
                payloads["spread_summary"],
                gradient_columns=payloads["spread_gradient_cols"],
            ),
            subtitle=payloads["spread_subtitle"],
        )
        _try_view(
            b, "GT10 vs Bund / OAT / Bono 10Y Spreads — 1-Year History",
            "Spread analysis", "fig",
            lambda: _fig_spread_history_1y(df),
        )

    # Yield curve chart
    _try_view(
        b, "DM Government Yield Curves (Latest)", "Yield curves", "fig",
        lambda: _fig_yield_curves(df, list(dg.COUNTRIES.keys())),
    )

    # Recent-window quantile tables consolidated into a country dropdown
    quantile_options = []
    for country, country_name, q_df_sorted in payloads.get("quantile_tables") or []:
        try:
            quantile_options.append(
                (
                    f"{country_name} ({country.upper()})",
                    _df_to_html_table(q_df_sorted, gradient_columns=["range_pct"]),
                )
            )
        except Exception as exc:
            print(f"Interactive quantile table '{country_name}' skipped: {exc}")
    _try_view(
        b,
        f"{dg.QUANTILE_WINDOW}-Day Yield Quantiles by Country (select country)",
        "Quantiles", "dropdown",
        quantile_options,
        label="Country",
        subtitle=payloads["q_subtitle"],
        search_extra="yield quantiles current min p25 p50 p75 max range",
    )
    return b


# ---------------------------------------------------------------------------
# HTML shell (ported from mbs_rv_interactive._write_html)
# ---------------------------------------------------------------------------
def _write_html(report_path: Path, payloads: Dict, builder: _ReportBuilder) -> None:
    df = payloads["df"]
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

    subtitle = f"Latest data {dg._idx_date(df.index[-1])} · {len(df):,} observations"
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
        report_path = dg._default_report_path("DM_GOVY_Report_Interactive", "html")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    builder = _build_pages(payloads)
    _write_html(report_path, payloads, builder)
    print(f"Interactive report saved to: {report_path}")
    return report_path
