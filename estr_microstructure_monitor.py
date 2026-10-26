r"""
€STR Swap-Rate Microstructure Monitor
=====================================
Interactive monitor for adjacent spreads and butterflies on the Citi €STR OIS
curve families: spot par rates and the 1y/2y/3y/5y/10y forward rate ladders.

Data: mbs_rv_citi.xlsx sheet ESTR (header row index 1, dates descending ->
sorted ascending). Par and forward rates are quoted in percent and converted
to bp for spread/fly arithmetic.

Monitors per family:
    - Adjacent spreads: consecutive-tenor differences along the family ladder
    - Adjacent flies: 2 x mid - left - right on consecutive ladder triples
    - Stats per series: latest, 1/5/20/60d changes, 252d z-score

Run:
    python estr_microstructure_monitor.py

Output:
    ./estr_microstructure_output/ESTR_Microstructure_Monitor_<latest data date>.html
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
SHEET_NAME = "ESTR"
OUTPUT_DIR = Path(__file__).parent / "estr_microstructure_output"

REPORT_TITLE = "€STR Swap Microstructure Monitor"
METHODOLOGY_NOTE = (
    "€STR OIS par/forward rates converted to bp; spread = right − left tenor; "
    "fly = 2×mid − wings; z = (latest − 252d mean) / 252d std (ddof=1)"
)

Z_WINDOW = 252
RECENT_DAYS = 365  # trailing window for the "recent" charts

PAR_RE = re.compile(r"RATES\.OIS\.EUR_EUROSTR\.PAR\.(\d+)Y - CLOSE")
FWD_RE = re.compile(r"RATES\.OIS\.EUR_EUROSTR\.FWD\.(\d+)Y\.(\d+)Y - CLOSE")

FAMILY_ORDER = ["Spot", "1y fwd", "2y fwd", "3y fwd", "5y fwd", "10y fwd"]

# Extended tab10-ish palette for spread/fly overlay lines
LINE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#2563eb", "#dc2626", "#059669",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_curve_families(fpath: str = FNAME) -> Dict[str, pd.DataFrame]:
    """Load the ESTR sheet; return family -> wide DataFrame of tenor columns (%).

    Families: "Spot" (PAR columns) and "{s}y fwd" for each FWD start tenor.
    Tenor ladders are discovered from the columns actually present.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    try:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME, header=1)
    except ValueError as exc:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' not found in {fpath}. Check the sheet name (exact case)."
        ) from exc

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date").sort_index()
    df = df[df.index.notna()]  # drop trailing blank rows in the sheet

    families: Dict[str, Dict[int, pd.Series]] = {}
    for col in df.columns:
        col = str(col)
        m = PAR_RE.match(col)
        if m:
            families.setdefault("Spot", {})[int(m.group(1))] = pd.to_numeric(df[col], errors="coerce")
            continue
        m = FWD_RE.match(col)
        if m:
            families.setdefault(f"{int(m.group(1))}y fwd", {})[int(m.group(2))] = (
                pd.to_numeric(df[col], errors="coerce")
            )

    out: Dict[str, pd.DataFrame] = {}
    for name in FAMILY_ORDER:
        tenors = families.get(name)
        if not tenors:
            print(f"  WARNING: family '{name}' has no columns; skipped")
            continue
        wide = pd.DataFrame({t: s for t, s in sorted(tenors.items())}).dropna(how="all")
        wide.columns = [f"{t}y" for t in wide.columns]
        out[name] = wide
        print(f"  {name}: {wide.shape[1]} tenors, {wide.shape[0]} rows "
              f"({wide.index[0].date()} to {wide.index[-1].date()})")
    return out


# ---------------------------------------------------------------------------
# Spread / fly construction and stats
# ---------------------------------------------------------------------------
def build_spreads(family_bp: pd.DataFrame) -> pd.DataFrame:
    """Adjacent-tenor spreads along the ladder, in bp."""
    tenors = list(family_bp.columns)
    out = {}
    for left, right in zip(tenors, tenors[1:]):
        out[f"{right}-{left}"] = family_bp[right] - family_bp[left]
    return pd.DataFrame(out).dropna(how="all")


def build_flies(family_bp: pd.DataFrame) -> pd.DataFrame:
    """Three-point flies on consecutive ladder triples: 2 x mid - wings, in bp."""
    tenors = list(family_bp.columns)
    out = {}
    for left, mid, right in zip(tenors, tenors[1:], tenors[2:]):
        out[f"{left}/{mid}/{right}"] = 2.0 * family_bp[mid] - family_bp[left] - family_bp[right]
    return pd.DataFrame(out).dropna(how="all")


def _chg(s: pd.Series, days: int) -> float:
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - days])


def _zscore(s: pd.Series, window: int = Z_WINDOW) -> float:
    recent = s.dropna().tail(window)
    if len(recent) < 2:
        return np.nan
    std = recent.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float((recent.iloc[-1] - recent.mean()) / std)


def build_series_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Latest / changes / z252 per column of a spread or fly frame."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            print(f"  WARNING: series '{col}' is empty; skipped")
            continue
        rows.append({
            "series": col,
            "latest": float(s.iloc[-1]),
            "chg_1d": _chg(s, 1),
            "chg_5d": _chg(s, 5),
            "chg_20d": _chg(s, 20),
            "chg_60d": _chg(s, 60),
            "z252": _zscore(s),
        })
    return pd.DataFrame(rows).set_index("series")


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
        legend=dict(font=dict(size=10)),
    )
    return fig


def _fig_overlay(df: pd.DataFrame, family: str, kind: str, days: Optional[int] = None) -> go.Figure:
    """Level overlay, one thin line per spread/fly; legend carries latest values."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_xs(s.index), y=_ys(s), mode="lines",
                name=f"{col} ({float(s.iloc[-1]):+.1f})",
                line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=1.1),
            )
        )
    if days is not None:
        end = df.index[-1]
        start = end - pd.Timedelta(days=days)
        fig.update_xaxes(range=[start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")])
        title = f"{family} {kind} - Trailing {days} Days"
    else:
        title = f"{family} {kind} - Full History"
    fig.add_hline(0, line=dict(color="#94a3b8", width=0.7))
    fig.update_yaxes(title_text="bp")
    _apply_layout(fig, title, subtitle="Legend shows latest level (bp)")
    return fig


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def _z_color(z: float) -> Optional[str]:
    """Diverging blue-white-red shading for z-scores, capped at +/-3."""
    if pd.isna(z):
        return None
    t = max(-1.0, min(1.0, float(z) / 3.0))
    if t >= 0:
        r, g, b = 255, int(255 - 150 * t), int(255 - 150 * t)
    else:
        r, g, b = int(255 + 150 * t), int(255 + 150 * t), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(x: float, fmt: str = "{:.2f}", na: str = "n/a") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _stats_table_html(stats: pd.DataFrame) -> str:
    cols = ["series", "latest", "chg_1d", "chg_5d", "chg_20d", "chg_60d", "z252"]
    header_labels = ["Series", "Latest (bp)", "Chg 1d", "Chg 5d", "Chg 20d", "Chg 60d", "z252"]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{c}</th>" for c in header_labels]
    parts.append("</tr></thead><tbody>")
    display = stats.reset_index()
    for i in range(len(display)):
        parts.append("<tr>")
        for c in cols:
            val = display[c].iloc[i]
            if c == "series":
                parts.append(f"<td>{html.escape(str(val))}</td>")
                continue
            if c == "latest":
                txt = _fmt(val)
            elif c == "z252":
                txt = _fmt(val, "{:+.2f}")
            else:
                txt = _fmt(val, "{:+.1f}")
            style = ""
            if c == "z252":
                color = _z_color(val)
                if color:
                    style = f' style="background:{color}"'
            parts.append(f"<td{style}>{txt}</td>")
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
.section h3{font-size:15px;margin:14px 0 8px;color:var(--ink)}
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
.dropdown-bar{margin:2px 0 14px;font-size:13px}
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:180px;background:#fff}
.note{font-size:12px;color:var(--muted);font-style:italic}
@media print{.header{position:static}.family-block{display:block !important;break-after:page}}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
function renderBlock(block){if(!block)return;block.querySelectorAll('.chart').forEach(el=>renderChart(el.id));}
function showFamily(slug){
document.querySelectorAll('.family-block').forEach(bl=>{
const on=bl.dataset.family===slug;bl.style.display=on?'':'none';if(on)renderBlock(bl);
});
}
const famSel=document.getElementById('family-select');
if(famSel){
famSel.addEventListener('change',()=>showFamily(famSel.value));
showFamily(famSel.value);
}else{
document.querySelectorAll('.chart').forEach(el=>renderChart(el.id));
}
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text===''||text==='n/a')return {blank:true,num:false,value:''};const n=Number(text.replace(/(%|bp)$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
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
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`estr-microstructure-table-${String(tableIndex+1).padStart(2,'0')}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
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


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def build_html(families: Dict[str, Dict[str, object]], report_path: Path) -> Path:
    builder = _ReportBuilder()
    as_of = max(f["spreads"].index[-1] for f in families.values()).date()

    default_slug = _slug(FAMILY_ORDER[0])
    options = "".join(
        f'<option value="{_slug(name)}"{" selected" if _slug(name) == default_slug else ""}>'
        f"{html.escape(name)}</option>"
        for name in FAMILY_ORDER
    )

    block_parts = []
    for name in FAMILY_ORDER:
        slug = _slug(name)
        hidden = "" if slug == default_slug else ' style="display:none"'
        fam = families.get(name)
        if fam is None:
            block_parts.append(
                f'<div class="family-block" data-family="{slug}"{hidden}>'
                '<p class="note">No data available for this family.</p></div>'
            )
            continue
        block_parts.append(
            f'<div class="family-block" data-family="{slug}"{hidden}>'
            f"<h3>Adjacent spreads</h3>{_stats_table_html(fam['spread_stats'])}"
            f"{builder.fig_block(_fig_overlay(fam['spreads'], name, 'Adjacent Spreads'))}"
            f"{builder.fig_block(_fig_overlay(fam['spreads'], name, 'Adjacent Spreads', days=RECENT_DAYS))}"
            f"<h3>Adjacent flies</h3>{_stats_table_html(fam['fly_stats'])}"
            f"{builder.fig_block(_fig_overlay(fam['flies'], name, 'Adjacent Flies'))}"
            f"{builder.fig_block(_fig_overlay(fam['flies'], name, 'Adjacent Flies', days=RECENT_DAYS))}"
            f"</div>"
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
<section class="section"><h2>Curve Families</h2>
<div class="dropdown-bar"><label>Family: <select id="family-select">{options}</select></label></div>
{''.join(block_parts)}
</section>
</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_console_summary(families: Dict[str, Dict[str, object]], report_path: Path) -> None:
    for name in FAMILY_ORDER:
        fam = families.get(name)
        if fam is None:
            continue
        print("\n" + "=" * 78)
        print(f"{name} - adjacent spreads (bp)")
        print("=" * 78)
        print(fam["spread_stats"].round(2).to_string())
        print(f"\n{name} - adjacent flies (bp)")
        print(fam["fly_stats"].round(2).to_string())
    print(f"\nHTML report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> Path:
    raw = load_curve_families(FNAME)
    if not raw:
        raise RuntimeError("No curve families found in the workbook.")

    print("Building spreads and flies...")
    families: Dict[str, Dict[str, object]] = {}
    for name, wide in raw.items():
        family_bp = wide * 100.0  # percent -> bp
        spreads = build_spreads(family_bp)
        flies = build_flies(family_bp)
        families[name] = {
            "spreads": spreads,
            "flies": flies,
            "spread_stats": build_series_stats(spreads),
            "fly_stats": build_series_stats(flies),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    as_of = max(f["spreads"].index[-1] for f in families.values()).strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"ESTR_Microstructure_Monitor_{as_of}.html"

    print("Building interactive HTML report...")
    build_html(families, report_path)

    print_console_summary(families, report_path)
    return report_path


if __name__ == "__main__":
    main()
