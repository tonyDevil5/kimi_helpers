r"""
Ad-hoc Rates / Vol / Correlation Report (interactive HTML)
===========================================================
Analyzes 7 daily series over the past 3 years, aligned on common trading days:

    1. UST 10Y yield                 USGG10YR Index             (%  -> bp)
    2. FNCC CC OAS basis             .FNCC105 G Index           (%  -> bp)
    2b. US IG corporate OAS          LUACOAS Index              (%  -> bp)
    3. UST 10Y swap spread           USSFCT10 BGN Curncy        (bp; fallbacks USSC10/USWS10)
    4. LCH-CME SOFR 10Y basis        USLCSS10 Curncy            (bp)
    5. EUR 5Y Euribor 6M-3M basis    EUSA5 - EUSW5V3 BGN Curncy (%  -> bp)
    6. SOFR 2s10s swap spread        USOSFR10 - USOSFR2 Curncy  (%  -> bp)
    7. SOFR 2s5s10s swap fly         2*USOSFR5 - USOSFR2 - USOSFR10 Curncy (% -> bp)

Analysis (daily changes on the aligned panel):
    - Annualized volatility of daily changes (std x sqrt(252)), in bp/yr.
    - Full pairwise correlation matrix of daily changes (heatmap).
    - 95th-percentile absolute 21-trading-day change, signed 5th/95th
      percentiles, and the latest 1m change for context.

Run:
    python adhoc_rates_vol_corr.py

Output:
    ./adhoc_output/rates_vol_corr/Adhoc_Rates_Vol_Corr_<latest data date>.html
"""

from __future__ import annotations

import html
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "adhoc_output" / "rates_vol_corr"
REPORT_TITLE = "Ad-hoc Rates Vol & Correlation Report"
LOOKBACK_YEARS = 3
MOVE_WINDOW = 21          # 1-month change window (trading days)
MIN_OBS = 30

# Raw Bloomberg inputs: (name, ticker candidates in preference order, scale -> display, units)
INPUT_SPECS: List[Tuple[str, List[str], float, str]] = [
    ("UST10Y", ["USGG10YR Index", "GT10 Govt"], 100.0, "bp"),
    ("FNCC", [".FNCC105 G Index"], 100.0, "bp"),
    ("IGOAS", ["LUACOAS Index"], 100.0, "bp"),
    ("SwapSprd10Y", ["USSFCT10 BGN Curncy", "USSFCT10 Curncy",
                     "USSC10 Curncy", "USWS10 Curncy"], 1.0, "bp"),
    ("LCHCME10Y", ["USLCSS10 Curncy"], 1.0, "bp"),
    ("Eur6m5Y", ["EUSA5 BGN Curncy", "EUSA5 Curncy"], 100.0, "bp"),
    ("Eur3m5Y", ["EUSW5V3 BGN Curncy", "EUSW5V3 Curncy"], 100.0, "bp"),
    ("SOFR2Y", ["USOSFR2 Curncy"], 100.0, "bp"),
    ("SOFR5Y", ["USOSFR5 Curncy"], 100.0, "bp"),
    ("SOFR10Y", ["USOSFR10 Curncy"], 100.0, "bp"),
]

# Report series built from the inputs: (display name, formula on input names, units)
REPORT_SPECS: List[Tuple[str, str]] = [
    ("UST 10Y", "UST10Y"),
    ("FNCC CC OAS", "FNCC"),
    ("US IG Corp OAS", "IGOAS"),
    ("UST 10Y swap sprd", "SwapSprd10Y"),
    ("LCH-CME SOFR 10Y", "LCHCME10Y"),
    ("EUR 5Y 6M-3M", "Eur6m5Y - Eur3m5Y"),
    ("SOFR 2s10s", "SOFR10Y - SOFR2Y"),
    ("SOFR 2s5s10s fly", "2*SOFR5Y - SOFR2Y - SOFR10Y"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_bdh(df, tickers=None) -> pd.DataFrame:
    """Normalize xbbg bdh output (narwhals long or wide MultiIndex) to date x ticker."""
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
        if tickers is not None:
            df = df.reindex(columns=[t for t in tickers if t in df.columns])
    elif isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_raw() -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """One bdh call for all candidate tickers. Returns (wide df, name -> ticker, failed names)."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    tickers = sorted({t for _, cands, _, _ in INPUT_SPECS for t in cands})
    end_date = date.today()
    start_date = end_date - timedelta(days=int(LOOKBACK_YEARS * 365.25) + 14)
    print(f"Downloading {len(tickers)} candidate tickers from Bloomberg "
          f"({start_date} to {end_date})...")
    df = blp.bdh(tickers=tickers, flds=["px_last"],
                 start_date=start_date.strftime("%Y-%m-%d"),
                 end_date=end_date.strftime("%Y-%m-%d"))
    df = _normalize_bdh(df, tickers=tickers)
    df = df.apply(pd.to_numeric, errors="coerce")

    resolved: Dict[str, str] = {}
    failed: List[str] = []
    for name, cands, _, _ in INPUT_SPECS:
        hit = next((t for t in cands if t in df.columns and df[t].notna().sum() > MIN_OBS), None)
        if hit:
            resolved[name] = hit
            if hit != cands[0]:
                print(f"  {name}: primary {cands[0]} failed, using {hit}")
        else:
            failed.append(name)
            print(f"  WARNING: {name} unresolved (tried {', '.join(cands)})")
    return df, resolved, failed


def build_panel(raw: pd.DataFrame, resolved: Dict[str, str]
                ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], List[str]]:
    """Assemble the 7 report series in display units (bp).

    Returns (panel, units, ticker-note map, failed report series).
    """
    inputs: Dict[str, pd.Series] = {}
    for name, cands, scale, _ in INPUT_SPECS:
        if name in resolved:
            inputs[name] = pd.to_numeric(raw[resolved[name]], errors="coerce") * scale

    series: Dict[str, pd.Series] = {}
    notes: Dict[str, str] = {}
    failed: List[str] = []
    for display, formula in REPORT_SPECS:
        try:
            s = eval(formula, {"__builtins__": {}}, inputs)
        except KeyError:
            failed.append(display)
            continue
        s = s.dropna()
        if len(s) < MIN_OBS:
            failed.append(display)
            continue
        series[display] = s
        notes[display] = formula

    # Align on common trading days, keeping each series' full history for obs reporting
    panel = pd.concat(series, axis=1, sort=True).dropna()
    units = {c: "bp" for c in panel.columns}
    print(f"Aligned panel: {panel.shape[0]} obs x {panel.shape[1]} series, "
          f"{panel.index[0].date()} to {panel.index[-1].date()}")
    return panel, units, notes, failed


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_stats(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (summary per series, daily changes, rolling 21d changes)."""
    changes = panel.diff().dropna()
    rolls = panel.diff(MOVE_WINDOW).dropna()

    rows = []
    for col in panel.columns:
        d = changes[col]
        r = rolls[col]
        rows.append({
            "series": col,
            "latest": float(panel[col].iloc[-1]),
            "ann_vol": float(d.std(ddof=1) * np.sqrt(252.0)),
            "abs1m_p95": float(r.abs().quantile(0.95)),
            "chg1m_p05": float(r.quantile(0.05)),
            "chg1m_p95": float(r.quantile(0.95)),
            "latest_1m_chg": float(r.iloc[-1]),
            "obs": int(panel[col].count()),
            "first": panel[col].first_valid_index().date().isoformat(),
            "last": panel[col].last_valid_index().date().isoformat(),
        })
    summary = pd.DataFrame(rows).set_index("series")
    return summary, changes, rolls


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: go.Figure, title: str, height: int = 460,
                  subtitle: Optional[str] = None) -> go.Figure:
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


def fig_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = list(corr.columns)
    z = corr.values
    text = [[f"{v:+.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        text=text, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="corr"),
    ))
    fig.update_yaxes(autorange="reversed")
    _apply_layout(
        fig, "Pairwise Correlation - Daily Changes",
        height=560,
        subtitle=f"{corr.index.name or ''}{len(labels)} series, aligned daily changes "
                 f"({corr.attrs.get('start', '')} to {corr.attrs.get('end', '')}, "
                 f"n={corr.attrs.get('n', '')})",
    )
    return fig


def fig_series_history(name: str, s: pd.Series, unit: str) -> go.Figure:
    s = s.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_xs(s.index), y=_ys(s), mode="lines", name=name,
        line=dict(color="#174a6e", width=1.4),
        hovertemplate="%{x}: %{y:.1f} bp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index[-1:]), y=[float(s.iloc[-1])], mode="markers",
        marker=dict(color="#dc2626", size=8), showlegend=False, hoverinfo="skip",
    ))
    fig.update_yaxes(title_text=unit)
    _apply_layout(
        fig, f"{name} - Level History",
        subtitle=f"Latest: {float(s.iloc[-1]):+.1f} {unit}  |  "
                 f"{s.index[0].date()} to {s.index[-1].date()}  |  {len(s)} obs",
    )
    return fig


def fig_vol_bars(summary: pd.DataFrame) -> go.Figure:
    df = summary.sort_values("ann_vol", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["ann_vol"], y=df.index, orientation="h",
        marker_color="#174a6e",
        text=[f"{v:.1f}" for v in df["ann_vol"]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f} bp/yr<extra></extra>",
    ))
    fig.update_xaxes(title_text="annualized vol of daily changes (bp/yr)")
    _apply_layout(fig, "Annualized Volatility of Daily Changes (bp/yr)", height=380)
    return fig


# ---------------------------------------------------------------------------
# HTML shell (repo conventions)
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
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:240px;background:#fff}
@media print{.header{position:static}.section{break-after:page}}
"""

_JS = r"""
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
function renderBlock(block){if(!block)return;block.querySelectorAll('.chart').forEach(el=>renderChart(el.id));}
function showStruct(group,slug){
document.querySelectorAll('.struct-block[data-group="'+group+'"]').forEach(bl=>{
const on=bl.dataset.struct===slug;bl.style.display=on?'':'none';if(on)renderBlock(bl);
});
}
document.querySelectorAll('select.struct-select').forEach(sel=>{
sel.addEventListener('change',()=>showStruct(sel.dataset.group,sel.value));
showStruct(sel.dataset.group,sel.value);
});
document.querySelectorAll('.chart-always').forEach(el=>renderChart(el.id));
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text===''||text==='—'||text==='N.A.'||text==='n/a')return {blank:true,num:false,value:''};const lead=text.match(/^[-+]?\d*\.?\d+/);if(lead)return {blank:false,num:true,value:Number(lead[0])};const n=Number(text.replace(/(%|bp)$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
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
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='rates-vol-corr.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
});
"""


class _ReportBuilder:
    """Collects figures as JSON and emits self-contained HTML blocks."""

    def __init__(self) -> None:
        self._fig_count = 0

    def fig_block(self, fig: go.Figure, always: bool = False) -> str:
        self._fig_count += 1
        fid = f"fig-{self._fig_count:03d}"
        fig_json = pio.to_json(fig).replace("</", "<\\/")
        cls = "chart chart-always" if always else "chart"
        return (
            f'<div class="{cls}" id="{fid}"></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{fig_json}</script>"
        )


def _fmt(x: float, fmt: str = "{:.1f}", na: str = "—") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _summary_table_html(summary: pd.DataFrame, tickers_used: Dict[str, str],
                        formulas: Dict[str, str]) -> str:
    cols = ["Series", "Definition / ticker", "Latest (bp)", "Ann Vol (bp/yr)",
            "|1m| p95 (bp)", "1m p05 (bp)", "1m p95 (bp)", "Latest 1m chg (bp)",
            "Obs", "From", "To"]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for name, row in summary.iterrows():
        values = {
            "Series": html.escape(name),
            "Definition / ticker": html.escape(f"{formulas.get(name, '')}  ({tickers_used.get(name, '')})"),
            "Latest (bp)": _fmt(row["latest"], "{:+.1f}"),
            "Ann Vol (bp/yr)": _fmt(row["ann_vol"], "{:.1f}"),
            "|1m| p95 (bp)": _fmt(row["abs1m_p95"], "{:.1f}"),
            "1m p05 (bp)": _fmt(row["chg1m_p05"], "{:+.1f}"),
            "1m p95 (bp)": _fmt(row["chg1m_p95"], "{:+.1f}"),
            "Latest 1m chg (bp)": _fmt(row["latest_1m_chg"], "{:+.1f}"),
            "Obs": str(int(row["obs"])),
            "From": row["first"],
            "To": row["last"],
        }
        parts.append("<tr>")
        for c in cols:
            style = ' style="font-weight:bold"' if c == "Series" else ""
            parts.append(f"<td{style}>{values[c]}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def build_html(panel: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame,
               tickers_used: Dict[str, str], formulas: Dict[str, str],
               notes: List[str], report_path: Path) -> Path:
    builder = _ReportBuilder()
    as_of = panel.index[-1].date()

    summary_section = (
        '<section class="section"><h2>Summary - Volatility &amp; 1-Month Moves (bp)</h2>'
        + _summary_table_html(summary, tickers_used, formulas)
        + '<p class="note">All series in bp. Ann Vol = std of daily changes × √252. '
          f'|1m| p95 = 95th percentile of |{MOVE_WINDOW}-trading-day change|; '
          f'1m p05/p95 = signed {MOVE_WINDOW}d change percentiles; Latest 1m chg = most recent '
          f'{MOVE_WINDOW}d change. Series are aligned on common trading days for the '
          "correlation analysis; per-series obs/date range reflect the aligned panel.</p></section>"
    )

    vol_section = (
        '<section class="section"><h2>Annualized Volatility</h2>'
        + builder.fig_block(fig_vol_bars(summary), always=True)
        + "</section>"
    )

    corr_section = (
        '<section class="section"><h2>Correlation Matrix - Daily Changes</h2>'
        + builder.fig_block(fig_heatmap(corr), always=True)
        + "</section>"
    )

    # Time-series dropdown
    names = list(panel.columns)
    options = "".join(
        f'<option value="{html.escape(n)}"{" selected" if n == names[0] else ""}>{html.escape(n)}</option>'
        for n in names
    )
    block_parts = []
    for n in names:
        hidden = "" if n == names[0] else ' style="display:none"'
        block_parts.append(
            f'<div class="struct-block" data-group="ts" data-struct="{html.escape(n)}"{hidden}>'
            f"{builder.fig_block(fig_series_history(n, panel[n], 'bp'))}"
            f"</div>"
        )
    ts_section = (
        '<section class="section"><h2>Series History (levels, bp)</h2>'
        f'<div class="dropdown-bar"><label>Series: '
        f'<select class="struct-select" data-group="ts">{options}</select></label></div>'
        + "".join(block_parts)
        + "</section>"
    )

    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes) or "<li>(none)</li>"
    notes_section = (
        '<section class="section"><h2>Data Notes</h2><ul class="note">'
        + notes_html + "</ul></section>"
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
<div class="tb-sub">as of {as_of}  |  {LOOKBACK_YEARS}y daily sample, aligned on common trading days  |  Bloomberg PX_LAST</div></nav>
<main>
{summary_section}
{vol_section}
{corr_section}
{ts_section}
{notes_section}
</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    raw, resolved, load_failed = load_raw()
    panel, units, formulas, build_failed = build_panel(raw, resolved)
    failed = sorted(set(load_failed + build_failed))

    summary, changes, rolls = compute_stats(panel)
    corr = changes.corr()
    corr.attrs = {
        "start": str(changes.index[0].date()), "end": str(changes.index[-1].date()),
        "n": len(changes),
    }

    tickers_used = {}
    for display, formula in formulas.items():
        parts = []
        for inp in re_names(formula):
            if inp in resolved:
                parts.append(resolved[inp])
        tickers_used[display] = ", ".join(dict.fromkeys(parts))

    notes = []
    for display in panel.columns:
        obs = int(panel[display].count())
        if obs < len(panel):
            notes.append(f"{display}: shorter history ({obs} aligned obs)")
    for f in failed:
        notes.append(f"Unresolved / excluded: {f}")
    notes.append("Unit conversions: %-quoted series (USGG10YR, .FNCC105, LUACOAS, EUSA5/EUSW5V3, "
                 "USOSFR2/5/10) are scaled ×100 to bp; USSFCT10 and USLCSS10 are already bp.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = panel.index[-1].strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"Adhoc_Rates_Vol_Corr_{report_date}.html"

    def _render(path: Path) -> Path:
        return build_html(panel, summary, corr, tickers_used, formulas, notes, path)

    try:
        out = _render(report_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUT_DIR / f"Adhoc_Rates_Vol_Corr_{report_date}_{ts}.html"
        print(f"Warning: could not write dated file (may be open). Saving to {report_path}")
        out = _render(report_path)

    print("\n" + "=" * 96)
    print(REPORT_TITLE)
    print("=" * 96)
    print("Resolved tickers:")
    for name, ticker in resolved.items():
        print(f"  {name:<14} -> {ticker}")
    if failed:
        print("Failed / excluded series:")
        for f in failed:
            print(f"  - {f}")
    print("\nSummary table (bp):")
    print(summary.round(2).to_string())
    print("\nCorrelation of daily changes:")
    print(corr.round(2).to_string())

    hv = summary["ann_vol"].idxmax()
    lv = summary["ann_vol"].idxmin()
    off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    print(f"\nHeadline:")
    print(f"  Highest vol : {hv} ({summary.loc[hv, 'ann_vol']:.1f} bp/yr)")
    print(f"  Lowest vol  : {lv} ({summary.loc[lv, 'ann_vol']:.1f} bp/yr)")
    print(f"  Strongest corr: {off_diag.idxmax()[0]} vs {off_diag.idxmax()[1]} "
          f"({off_diag.max():+.2f})")
    print(f"  Weakest corr  : {off_diag.idxmin()[0]} vs {off_diag.idxmin()[1]} "
          f"({off_diag.min():+.2f})")
    print(f"\nHTML report saved to: {out}")
    print("=" * 96)


def re_names(formula: str) -> List[str]:
    import re
    return [tok for tok in re.split(r"[^A-Za-z0-9_]+", formula) if tok and not tok[0].isdigit()]


if __name__ == "__main__":
    main()
