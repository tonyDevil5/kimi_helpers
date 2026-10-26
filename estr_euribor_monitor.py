r"""
ESTR vs Euribor 6M Forward Basis Monitor
========================================
Interactive monitor for the implied €STR-vs-Euribor-6M forward basis across
the nY2Y forward structures (n = 1..9), built from Citi EUR forward basis
spreads in mbs_rv_citi.xlsx (sheet EUR_ESTR_EURIBOR).

Methodology:
    estr6m(nY2Y) = estr3m(nY2Y) + 3s6s(nY2Y)
where estr3m is the €STR-vs-3M-Euribor basis (EUROSTR_EURIBOR_BASIS family) and
3s6s is the 3M-vs-6M Euribor basis (3S6S_BASIS family). All values in bp.

Report content per structure: latest estr6m level, 1/5/20d changes, z-scores
of the level over 60/90/120/250 trading days, a full-history level chart and
a four-window z-score panel.

Run:
    python estr_euribor_monitor.py

Output:
    ./estr_euribor_output/ESTR_Euribor_Forward_Basis_<latest data date>.html
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FNAME = r"D:\python\notebook\data\mbs\mbs_rv_citi.xlsx"
SHEET_NAME = "EUR_ESTR_EURIBOR"
OUTPUT_DIR = Path(__file__).parent / "estr_euribor_output"

REPORT_TITLE = "€STR vs Euribor 6M Forward Basis Monitor"
METHODOLOGY_NOTE = (
    "estr6m = (€STR−3M Euribor) + (3M−6M Euribor) forward basis, "
    "Citi EUR forward basis spreads"
)

Z_WINDOWS = [60, 90, 120, 250]
Z_CHART_DAYS = 2 * 365  # trailing x-range for the z-score panels
MIN_OBS = 60  # below this a structure is skipped with a note

# Forward structures nY2Y
STRUCTURES = list(range(1, 10))

# Citi column families
ESTR3M_FAMILY = "EUROSTR_EURIBOR_BASIS"   # €STR vs 3M Euribor leg
S3S6_FAMILY = "3S6S_BASIS"                # 3M vs 6M Euribor leg
TENOR_RE = re.compile(r"\.(\d+)Y\.2Y\.")

STRUCT_COLORS = {
    1: "#2563eb", 2: "#0891b2", 3: "#059669", 4: "#d97706",
    5: "#dc2626", 6: "#7c3aed", 7: "#db2777", 8: "#475569",
    9: "#0d9488",
}
Z_COLORS = {60: "#dc2626", 90: "#d97706", 120: "#2563eb", 250: "#7c3aed"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_basis_data(fpath: str = FNAME) -> Dict[int, Dict[str, object]]:
    """Load the Citi EUR basis sheet; return per-structure data dicts.

    Each entry holds the raw leg series (estr3m / s3s6, independently dropna'd)
    plus "df", a date-aligned frame of both legs and their sum estr6m (bp).
    Structures missing a leg (or with too little aligned history) are omitted.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"Workbook not found: {fpath}")

    try:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME, header=1)
    except ValueError as exc:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' not found in {fpath}. "
            "Check the sheet name (exact case)."
        ) from exc

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    legs: Dict[str, Dict[int, pd.Series]] = {ESTR3M_FAMILY: {}, S3S6_FAMILY: {}}
    for col in df.columns:
        m = TENOR_RE.search(str(col))
        if not m:
            continue
        n = int(m.group(1))
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if ESTR3M_FAMILY in col:
            legs[ESTR3M_FAMILY][n] = series
        elif S3S6_FAMILY in col:
            legs[S3S6_FAMILY][n] = series

    out: Dict[int, Dict[str, object]] = {}
    for n in STRUCTURES:
        estr3m = legs[ESTR3M_FAMILY].get(n)
        s3s6 = legs[S3S6_FAMILY].get(n)
        if estr3m is None or s3s6 is None or estr3m.empty or s3s6.empty:
            continue
        data = pd.concat(
            [estr3m.rename("estr3m"), s3s6.rename("s3s6")], axis=1
        ).dropna()
        if len(data) < MIN_OBS:
            print(f"  WARNING: {n}y2y has only {len(data)} aligned observations; skipped")
            continue
        data["estr6m"] = data["estr3m"] + data["s3s6"]
        out[n] = {"df": data, "estr3m": estr3m, "s3s6": s3s6}

    if out:
        print(f"Loaded {len(out)} structures "
              f"({min(d['df'].index[0] for d in out.values()).date()} to "
              f"{max(d['df'].index[-1] for d in out.values()).date()})")
    # Flag stale legs: the estr6m sum needs both legs on the same date
    for family, leg_key, label in (
        (ESTR3M_FAMILY, "estr3m", "estr3m"),
        (S3S6_FAMILY, "s3s6", "3s6s"),
    ):
        for n in sorted(out):
            raw = out[n][leg_key]
            aligned_last = out[n]["df"].index[-1]
            if raw.index[-1] > aligned_last:
                print(
                    f"  WARNING: {n}y2y {label} leg ends {raw.index[-1].date()} but the "
                    f"other leg ends {aligned_last.date()}; estr6m series stops at "
                    f"{aligned_last.date()}"
                )
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _zscore(s: pd.Series, window: int) -> float:
    """Z of the latest level vs the trailing window (inclusive, ddof=1)."""
    recent = s.dropna().tail(window)
    if len(recent) < 2:
        return np.nan
    std = recent.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float((s.dropna().iloc[-1] - recent.mean()) / std)


def _chg(s: pd.Series, days: int) -> float:
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - days])


def build_structure_stats(data: Dict[int, Dict[str, object]]) -> Dict[int, Dict[str, float]]:
    """Latest level, changes and z-scores per structure.

    Leg latest values come from each leg's own (raw) series, so a stale leg
    is visible as such rather than silently truncated to the aligned sample.
    """
    stats: Dict[int, Dict[str, float]] = {}
    for n, entry in data.items():
        df: pd.DataFrame = entry["df"]  # type: ignore[assignment]
        estr3m: pd.Series = entry["estr3m"]  # type: ignore[assignment]
        s3s6: pd.Series = entry["s3s6"]  # type: ignore[assignment]
        s = df["estr6m"]
        row: Dict[str, float] = {
            "latest": float(s.iloc[-1]),
            "last_date": df.index[-1],
            "estr3m_latest": float(estr3m.iloc[-1]),
            "estr3m_last_date": estr3m.index[-1],
            "s3s6_latest": float(s3s6.iloc[-1]),
            "s3s6_last_date": s3s6.index[-1],
            "chg_1d": _chg(s, 1),
            "chg_5d": _chg(s, 5),
            "chg_20d": _chg(s, 20),
            "vol_20d": float(s.diff().iloc[-20:].std(ddof=1)),
        }
        for w in Z_WINDOWS:
            row[f"z_{w}d"] = _zscore(s, w)
        stats[n] = row
    return stats


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: go.Figure, title: str, height: int = 460, subtitle: Optional[str] = None) -> go.Figure:
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


def _fig_level(n: int, df: pd.DataFrame) -> go.Figure:
    s = df["estr6m"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(s.index), y=_ys(s), mode="lines", name="estr6m",
            line=dict(color=STRUCT_COLORS.get(n, "#475569"), width=1.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_xs(s.index[-1:]), y=[float(s.iloc[-1])], mode="markers",
            marker=dict(color="#111827", size=7), showlegend=False, hoverinfo="skip",
        )
    )
    fig.update_yaxes(title_text="bp")
    _apply_layout(
        fig, f"{n}Y2Y estr6m Forward Basis - Full History",
        subtitle=f"Latest: {float(s.iloc[-1]):.2f} bp  |  "
                 f"{s.index[0].date()} to {s.index[-1].date()}",
    )
    return fig


def _fig_zscores(n: int, df: pd.DataFrame) -> go.Figure:
    s = df["estr6m"]
    fig = go.Figure()
    for w in Z_WINDOWS:
        mean = s.rolling(window=w, min_periods=w).mean()
        std = s.rolling(window=w, min_periods=w).std(ddof=1)
        z = ((s - mean) / std.replace(0, np.nan)).dropna()
        if z.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_xs(z.index), y=_ys(z), mode="lines", name=f"z{w}",
                line=dict(color=Z_COLORS[w], width=1.2),
            )
        )
    end = s.index[-1]
    start = end - pd.Timedelta(days=Z_CHART_DAYS)
    x0, x1 = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    fig.update_xaxes(range=[x0, x1])
    for level, color, dash in ((0, "#111827", "solid"), (2, "#b91c1c", "dash"), (-2, "#b91c1c", "dash")):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                      line=dict(color=color, width=0.9, dash=dash))
    fig.update_yaxes(title_text="Z-score")
    _apply_layout(
        fig, f"{n}Y2Y estr6m Z-Scores ({'/'.join(map(str, Z_WINDOWS))}d)", height=400,
        subtitle=f"Last ~{Z_CHART_DAYS // 365} years  |  z = (level - window mean) / window std (ddof=1)",
    )
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _z_color(z: float) -> Optional[str]:
    """Diverging blue-white-red shading for z-scores, capped at +/-3."""
    if pd.isna(z):
        return None
    t = max(-1.0, min(1.0, float(z) / 3.0))
    if t >= 0:
        # white -> red
        r, g, b = 255, int(255 - 150 * t), int(255 - 150 * t)
    else:
        # white -> blue
        r, g, b = int(255 + 150 * t), int(255 + 150 * t), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(x: float, fmt: str = "{:.2f}", na: str = "n/a") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _summary_table_html(stats: Dict[int, Dict[str, float]]) -> str:
    cols = (
        ["Structure", "estr6m (bp)", "Chg 1d (bp)", "Chg 5d (bp)", "Chg 20d (bp)",
         "Vol 20d (bp/day)"]
        + [f"z{w}" for w in Z_WINDOWS]
        + ["estr3m leg (bp)", "3s6s leg (bp)"]
    )
    z_cols = [f"z{w}" for w in Z_WINDOWS]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for n in STRUCTURES:
        row = stats.get(n)
        if row is None:
            cells = [f"{n}y2y", "N.A."] + ["—"] * (len(cols) - 2)
            parts.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
            continue
        estr3m_txt = (
            f"{_fmt(row['estr3m_latest'])} "
            f"({pd.Timestamp(row['estr3m_last_date']).date()})"
        )
        s3s6_txt = (
            f"{_fmt(row['s3s6_latest'])} "
            f"({pd.Timestamp(row['s3s6_last_date']).date()})"
        )
        values = {
            "Structure": f"{n}y2y",
            "estr6m (bp)": _fmt(row["latest"]),
            "Chg 1d (bp)": _fmt(row["chg_1d"], "{:+.2f}"),
            "Chg 5d (bp)": _fmt(row["chg_5d"], "{:+.2f}"),
            "Chg 20d (bp)": _fmt(row["chg_20d"], "{:+.2f}"),
            "Vol 20d (bp/day)": _fmt(row["vol_20d"]),
            "estr3m leg (bp)": estr3m_txt,
            "3s6s leg (bp)": s3s6_txt,
        }
        for w in Z_WINDOWS:
            values[f"z{w}"] = _fmt(row[f"z_{w}d"], "{:+.2f}")
        parts.append("<tr>")
        for c in cols:
            style = ""
            if c in z_cols and pd.notna(row[f"z_{c[1:]}d"]):
                color = _z_color(row[f"z_{c[1:]}d"])
                if color:
                    style = f' style="background:{color}"'
            parts.append(f"<td{style}>{html.escape(values[c])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
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
.section h3{font-size:15px;margin:2px 0 10px;color:var(--ink)}
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
.note{font-size:12px;color:var(--muted);font-style:italic}
.dropdown-bar{margin:2px 0 14px;font-size:13px}
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:180px;background:#fff}
@media print{.header{position:static}.section{break-after:page}}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
function renderBlock(block){if(!block)return;block.querySelectorAll('.chart').forEach(el=>renderChart(el.id));}
function showStructure(slug){
document.querySelectorAll('.struct-block').forEach(bl=>{
const on=bl.dataset.struct===slug;bl.style.display=on?'':'none';if(on)renderBlock(bl);
});
}
const structSel=document.getElementById('struct-select');
if(structSel){
structSel.addEventListener('change',()=>showStructure(structSel.value));
showStructure(structSel.value);
}else{
document.querySelectorAll('.chart').forEach(el=>renderChart(el.id));
}
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text===''||text==='—'||text==='N.A.'||text==='n/a')return {blank:true,num:false,value:''};const n=Number(text.replace(/(%|bp)$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
function csvCell(value){return '"'+value.replace(/"/g,'""')+'"';}
document.querySelectorAll('.sortable-table').forEach((table)=>{
const wrap=table.closest('.table-wrap');if(!wrap)return;const rows=[...table.tBodies[0].rows];const tools=document.createElement('div');tools.className='table-tools';
const search=document.createElement('input');search.type='search';search.placeholder='Filter this table';search.setAttribute('aria-label','Filter this table');
const reset=document.createElement('button');reset.type='button';reset.textContent='Reset sort';
const csv=document.createElement('button');csv.type='button';csv.textContent='Download CSV';
const count=document.createElement('span');count.className='table-count';tools.append(search,reset,csv,count);wrap.parentNode.insertBefore(tools,wrap);
const refresh=()=>{let shown=0;rows.forEach(row=>{if(row.style.display!=='none')shown++;});count.textContent=`${shown} of ${rows.length} rows`;};refresh();
search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();rows.forEach(row=>{row.style.display=(!q||row.textContent.toLowerCase().includes(q))?'':'none';});refresh();});
table.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{const asc=!th.classList.contains('sort-asc');table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));th.classList.add(asc?'sort-asc':'sort-desc');const sorted=[...rows].sort((a,b)=>{const av=cellValue(a.cells[col]),bv=cellValue(b.cells[col]);if(av.blank!==bv.blank)return av.blank?1:-1;let c;if(av.num&&bv.num)c=av.value-bv.value;else c=String(av.value).localeCompare(String(bv.value));return asc?c:-c;});sorted.forEach(row=>table.tBodies[0].appendChild(row));}));
reset.addEventListener('click',()=>{rows.forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));});
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='estr-6m-forward-basis.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
});
"""


class _ReportBuilder:
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


def build_html(
    data: Dict[int, Dict[str, object]],
    stats: Dict[int, Dict[str, float]],
    report_path: Path,
) -> Path:
    builder = _ReportBuilder()
    as_of = max(e["df"].index[-1] for e in data.values()).date()

    # Staleness note: estr6m can only be computed where both legs overlap
    stale: Dict[Tuple[str, object, object], List[str]] = {}
    for n in sorted(data):
        entry = data[n]
        df: pd.DataFrame = entry["df"]  # type: ignore[assignment]
        for leg_key, leg_name in (("estr3m", "estr3m"), ("s3s6", "3s6s")):
            leg: pd.Series = entry[leg_key]  # type: ignore[assignment]
            if leg.index[-1] > df.index[-1]:
                key = (leg_name, leg.index[-1].date(), df.index[-1].date())
                stale.setdefault(key, []).append(f"{n}y2y")
    stale_html = ""
    if stale:
        notes = [
            f"{leg} leg is current to {leg_date} but the other leg ends {other_date} "
            f"({', '.join(structs)})"
            for (leg, leg_date, other_date), structs in stale.items()
        ]
        stale_html = (
            '<p class="note" style="color:#b91c1c">Data staleness: '
            + "; ".join(html.escape(t) for t in notes)
            + ". estr6m levels and z-scores are computed only over dates where both "
            "legs overlap; leg latest values in the table show each leg's own as-of date.</p>"
        )

    # Summary section
    summary_parts = [
        "<h2>Summary - estr6m Forward Basis (bp)</h2>",
        _summary_table_html(stats),
        stale_html,
    ]

    # Per-structure charts: dropdown selector + hidden blocks (lazy render)
    default_slug = f"{STRUCTURES[0]}y2y"
    options = "".join(
        f'<option value="{n}y2y"{" selected" if f"{n}y2y" == default_slug else ""}>{n}y2y</option>'
        for n in STRUCTURES
    )
    block_parts = []
    for n in STRUCTURES:
        slug = f"{n}y2y"
        hidden = "" if slug == default_slug else ' style="display:none"'
        if n not in data:
            block_parts.append(
                f'<div class="struct-block" data-struct="{slug}"{hidden}>'
                '<p class="note">No data available for this structure: one or both '
                "Citi basis legs are missing from the workbook.</p></div>"
            )
            continue
        block_parts.append(
            f'<div class="struct-block" data-struct="{slug}"{hidden}>'
            f"{builder.fig_block(_fig_level(n, data[n]['df']))}"
            f"{builder.fig_block(_fig_zscores(n, data[n]['df']))}"
            f"</div>"
        )
    charts_section = (
        '<section class="section"><h2>Structure Charts</h2>'
        f'<div class="dropdown-bar"><label>Structure: '
        f'<select id="struct-select">{options}</select></label></div>'
        + "".join(block_parts)
        + "</section>"
    )

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
<div class="tb-sub">as of {as_of}  |  {html.escape(METHODOLOGY_NOTE)}</div></nav>
<main>
<section class="section">{''.join(summary_parts)}</section>
{charts_section}
</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_console_summary(
    data: Dict[int, Dict[str, object]],
    stats: Dict[int, Dict[str, float]],
    report_path: Path,
) -> None:
    print("\nSanity check (latest aligned date): estr3m + 3s6s = estr6m")
    for n in sorted(stats):
        df: pd.DataFrame = data[n]["df"]  # type: ignore[assignment]
        last = df.iloc[-1]
        print(
            f"  {n}y2y: {last['estr3m']:8.4f} + {last['s3s6']:8.4f} "
            f"= {last['estr3m'] + last['s3s6']:8.4f} "
            f"(series: {last['estr6m']:.4f})"
        )
        e3: pd.Series = data[n]["estr3m"]  # type: ignore[assignment]
        s6: pd.Series = data[n]["s3s6"]  # type: ignore[assignment]
        if e3.index[-1] > df.index[-1] or s6.index[-1] > df.index[-1]:
            print(
                f"        raw leg latests: estr3m {e3.iloc[-1]:.4f} ({e3.index[-1].date()}), "
                f"3s6s {s6.iloc[-1]:.4f} ({s6.index[-1].date()})"
            )

    print("\n" + "=" * 88)
    print(f"{REPORT_TITLE} - summary")
    print("=" * 88)
    z_headers = "".join(f"{'z' + str(w):>8}" for w in Z_WINDOWS)
    header = f"{'Struct':<8}{'estr6m':>9}{'chg1d':>8}{'chg5d':>8}{'chg20d':>8}{'vol20':>8}{z_headers}"
    print(header)
    print("-" * len(header))
    for n in STRUCTURES:
        row = stats.get(n)
        if row is None:
            print(f"{n}y2y{'':<4}{'N.A.':>9}")
            continue
        z_vals = "".join(f"{_fmt(row[f'z_{w}d'], '{:+.2f}'):>8}" for w in Z_WINDOWS)
        print(
            f"{n}y2y{'':<4}{_fmt(row['latest']):>9}"
            f"{_fmt(row['chg_1d'], '{:+.2f}'):>8}"
            f"{_fmt(row['chg_5d'], '{:+.2f}'):>8}"
            f"{_fmt(row['chg_20d'], '{:+.2f}'):>8}"
            f"{_fmt(row['vol_20d']):>8}"
            f"{z_vals}"
        )
    print(f"\nHTML report saved to: {report_path}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> Path:
    data = load_basis_data(FNAME)
    if not data:
        raise RuntimeError("No computable structures found in the workbook.")

    print("Computing summary statistics...")
    stats = build_structure_stats(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    as_of = max(e["df"].index[-1] for e in data.values()).strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"ESTR_Euribor_Forward_Basis_{as_of}.html"

    print("Building interactive HTML report...")
    build_html(data, stats, report_path)

    print_console_summary(data, stats, report_path)
    return report_path


if __name__ == "__main__":
    main()
