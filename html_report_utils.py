"""Shared helpers for single-file interactive HTML reports.

Page chrome mirrors the CTA comparison report style; table gradient coloring
replicates the ``_column_color_func`` convention used by the matplotlib PDF
reports (diverging blue-white-red when a column spans zero, sequential
white-to-red otherwise) without a matplotlib dependency.
"""

from __future__ import annotations

import html
import re

import pandas as pd
import plotly.io as pio

PAGE_CSS = """
:root{--bg:#eef3f7;--ink:#17212c;--muted:#536574;--navy:#12344d;--line:#dbe4ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}section{background:white;border:1px solid var(--line);border-radius:10px;margin:0 0 18px;padding:20px;box-shadow:0 2px 8px #17212c0b;scroll-margin-top:240px}
nav{position:sticky;top:0;z-index:50;background:var(--navy);border-radius:8px;margin:0 0 18px;padding:8px 12px;display:flex;flex-wrap:wrap;gap:4px;box-shadow:0 2px 8px #17212c33}
nav a{color:#d8e8f8;text-decoration:none;font-size:12px;padding:4px 9px;border-radius:5px;white-space:nowrap}
nav a:hover{background:#ffffff26;color:white}
h1{font-size:26px;margin:0 0 8px}h2{font-size:18px;margin:0 0 14px}h3{font-size:15px;margin:14px 0 8px}p{margin:8px 0}.subtle{color:var(--muted)}.table-wrap{overflow:auto;max-height:80vh}
table{border-collapse:collapse;width:100%;font-size:12px}th{background:var(--navy);color:white;padding:7px;text-align:right;position:sticky;top:0}
th:first-child,td:first-child{text-align:left}td{border-bottom:1px solid var(--line);padding:6px 7px;text-align:right;white-space:nowrap}
tbody tr:nth-child(even){background:#f5f8fa}.side-by-side{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
.badge-pos{color:#15803d;font-weight:bold}.badge-neg{color:#b91c1c;font-weight:bold}.badge-zero{color:#6b7280;font-weight:bold}
@media(max-width:700px){main{padding:10px}section{padding:12px}}
"""

_BADGE_CLASS = {
    "bull": "badge-pos", "bullish": "badge-pos", "long": "badge-pos",
    "bear": "badge-neg", "bearish": "badge-neg", "short": "badge-neg",
    "mixed": "badge-zero", "flat": "badge-zero",
}


def _hex_lerp(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = (int(c1[i:i + 2], 16) for i in (1, 3, 5))
    r2, g2, b2 = (int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(
        round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t)
    )


def _stop_color(stops: list[str], norm: float) -> str:
    """Evenly-spaced color stops, matching matplotlib's from_list colormap."""
    norm = max(0.0, min(1.0, norm))
    n = len(stops) - 1
    pos, frac = divmod(norm * n, 1.0)
    idx = min(int(pos), n - 1)
    return _hex_lerp(stops[idx], stops[idx + 1], frac)


def _diverging_color(x: float, limit: float) -> str:
    norm = (float(x) + limit) / (2 * limit)
    return _stop_color(["#4575b4", "#ffffff", "#d73027"], norm)


def _sequential_color(x: float, vmin: float, vmax: float) -> str:
    if vmax == vmin:
        return "#ffffff"
    norm = (float(x) - vmin) / (vmax - vmin)
    return _stop_color(["#ffffff", "#fee0d2", "#fc9272", "#de2d26"], norm)


def _column_color_map(series: pd.Series) -> dict[float, str]:
    """Value -> hex color for one numeric column, PDF-report convention."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    vmin, vmax = float(s.min()), float(s.max())
    out = {}
    for value in s.unique():
        if vmin < 0 < vmax:
            out[value] = _diverging_color(value, max(abs(vmin), abs(vmax), 0.01))
        else:
            out[value] = _sequential_color(value, vmin, vmax)
    return out


def df_to_html_table(
    df: pd.DataFrame,
    gradient_cols: tuple[str, ...] | list[str] = (),
    badge_cols: tuple[str, ...] | list[str] = (),
    fmt: str = "{:.3f}",
) -> str:
    """Render a DataFrame as an HTML table with PDF-style gradient cells."""
    frame = df.copy()
    show_index = not isinstance(frame.index, pd.RangeIndex)
    index_name = html.escape(str(frame.index.name or frame.columns.name or ""))
    columns = [html.escape(str(c)) for c in frame.columns]
    color_maps = {
        col: _column_color_map(frame[col]) for col in gradient_cols if col in frame.columns
    }

    def _index_text(idx) -> str:
        if isinstance(idx, tuple):
            return " / ".join(str(part) for part in idx)
        return str(idx)

    def _cell(col: str, value) -> str:
        if pd.isna(value):
            return "<td></td>"
        if col in badge_cols:
            cls = _BADGE_CLASS.get(str(value).lower(), "badge-zero")
            return f'<td class="{cls}">{html.escape(str(value))}</td>'
        if isinstance(value, (int, float)):
            text = fmt.format(value)
            color = color_maps.get(col, {}).get(value)
            style = f' style="background-color:{color}"' if color else ""
            return f"<td{style}>{text}</td>"
        return f"<td>{html.escape(str(value))}</td>"

    header = "".join(f"<th>{c}</th>" for c in columns)
    if show_index:
        header = f"<th>{index_name}</th>" + header
    header = f"<tr>{header}</tr>"
    rows = []
    for idx, row in frame.iterrows():
        cells = ""
        if show_index:
            cells = f"<td>{html.escape(_index_text(idx))}</td>"
        cells += "".join(_cell(col, row[col]) for col in frame.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def figs_to_html(figs: list) -> str:
    """Embed Plotly figures; plotly.js inlined once with the first figure."""
    parts = []
    for i, fig in enumerate(figs):
        parts.append(pio.to_html(
            fig,
            include_plotlyjs="inline" if i == 0 else False,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        ))
    return "".join(parts)


def page_template(title: str, subtitle: str, sections_html: str, nav: bool = True) -> str:
    """Assemble the page; when nav=True, add a sticky section navigation bar.

    Sections are auto-anchored: every `<section><h2>Title</h2>` gets an id and
    a nav link.
    """
    if nav:
        links = []
        count = 0

        def _anchor(match: "re.Match") -> str:
            nonlocal count
            count += 1
            sec_id = f"sec-{count}"
            links.append(f'<a href="#{sec_id}">{match.group(1)}</a>')
            return f'<section id="{sec_id}"><h2>{match.group(1)}</h2>'

        sections_html = re.sub(
            r"<section><h2>(.*?)</h2>", _anchor, sections_html
        )
        nav_html = f"<nav>{''.join(links)}</nav>" if links else ""
    else:
        nav_html = ""
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{PAGE_CSS}</style></head><body><main>"
        f"<section><h1>{html.escape(title)}</h1><p class=\"subtle\">{subtitle}</p></section>"
        f"{nav_html}{sections_html}</main></body></html>"
    )
