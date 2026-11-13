"""Interactive HTML report: MBS origination vs JPM coupon performance.

Loads daily Citi securitized-MBS origination trackers (sheet 'MBS_ORIG' in
mbs_rv_citi.xlsx, $mm notional per FN/G2 coupon, NaN filled with 0) and daily
JPM DataQuery coupon performance (32nds) from mbs_rv.xlsx.  For each
agency-coupon and rolling window (10/20/30d), runs OLS of rolling-sum perf on
rolling-sum origination, and reports beta / t-stat / R2 / correlation in a
summary table.  Charts: daily origination per agency (last 250 trading days,
legend shows latest value), recent-days bar chart, and a dual-axis
rolling-orig vs rolling-perf chart switchable by coupon and window.

Run: .venv/Scripts/python.exe mbs_origination_analysis.py
"""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.offline import get_plotlyjs

import mbs_rv_report as rv

REPORT_TITLE = "MBS Origination vs Coupon Performance"
OUTPUT_DIR = Path(__file__).parent / "mbs_origination_output"

AGENCIES = rv.COUPON_BASIS_AGENCIES          # ["FN", "G2"]
COUPONS = rv.COUPONS                          # 2.0 ... 6.5
WINDOWS = [10, 20, 30]                        # rolling-sum windows (days)
RECENT_DAYS = 250                             # trading days in daily-orig charts
BAR_DAYS = 30                                 # most recent days in bar chart
MIN_OBS = 250                                 # minimum common history for a coupon

_AGENCY_MAP = {"FNMA": "FN", "GNMAII": "G2"}
_ORIG_RE = re.compile(
    r"ORIG_TRACKER\.OT_NOTIONAL\.OT_(FNMA|GNMAII)(?:\.30Y)?\.OT_(\d+)_(\d+)_OT")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_origination(fpath: str | Path = rv.CITI_FNAME) -> pd.DataFrame:
    """Daily origination per agency-coupon ($mm), columns 'FN 5.0' etc., NaN->0."""
    raw = pd.read_excel(fpath, sheet_name="MBS_ORIG", header=1, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    series: Dict[str, pd.Series] = {}
    for col in raw.columns:
        m = _ORIG_RE.search(str(col))
        if not m:
            continue
        agency = _AGENCY_MAP[m.group(1)]
        coupon = float(f"{m.group(2)}.{m.group(3)}")
        name = f"{agency} {coupon:g}"
        series[name] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
    df = pd.DataFrame(series)
    df = df[sorted(df.columns, key=rv.coupon_sort_key)]
    print(f"Origination: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()} (NaN filled with 0)")
    return df


def load_perf() -> pd.DataFrame:
    """Daily JPM perf (32nds) per agency-coupon, columns 'FN 5.0' etc."""
    dat = rv.load_mbs_rv(rv.FNAME)
    series = {}
    for agency in AGENCIES:
        for col in rv.get_block_columns(dat, agency, "Perf"):
            name = col.replace(" Perf", "")
            series[name] = pd.to_numeric(dat[col], errors="coerce")
    df = pd.DataFrame(series)
    return df[sorted(df.columns, key=rv.coupon_sort_key)]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def rolling_ols_stats(orig: pd.Series, perf: pd.Series, window: int) -> Optional[dict]:
    """OLS of rolling-x-day perf sum on rolling-x-day orig sum over common dates."""
    ro = orig.rolling(window).sum()
    rp = perf.rolling(window).sum()
    sample = pd.concat([rp.rename("y"), ro.rename("x")], axis=1, sort=True).dropna()
    if sample.shape[0] < MIN_OBS or sample["x"].std() < 1e-12:
        return None
    X = sm.add_constant(sample["x"], has_constant="add")
    model = sm.OLS(sample["y"], X).fit()
    return {
        "beta": float(model.params["x"]),
        "t": float(model.tvalues["x"]),
        "R2": float(model.rsquared),
        "corr": float(sample["y"].corr(sample["x"])),
        "obs": int(model.nobs),
    }


def build_summary(orig: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in orig.columns:
        if name not in perf.columns:
            continue
        row = {"structure": name}
        for w in WINDOWS:
            stats = rolling_ols_stats(orig[name], perf[name], w)
            for k in ("beta", "t", "R2", "corr"):
                row[f"{k}_{w}"] = stats[k] if stats else np.nan
        row["obs"] = stats["obs"] if stats else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("structure")


# ---------------------------------------------------------------------------
# HTML helpers (conventions mirror mbs_perf_regression.py)
# ---------------------------------------------------------------------------
def _rg_color(value: float, limit: float) -> str:
    """Diverging red/green: negative red, positive green, white near zero."""
    if pd.isna(value):
        return "#ffffff"
    norm = max(-1.0, min(1.0, value / max(limit, 1e-9)))
    if norm >= 0:
        g, r_b = 180, int(255 - 155 * norm)
        return f"#{r_b:02x}{g:02x}{r_b:02x}"
    r, g_b = 220, int(255 - 155 * (-norm))
    return f"#{r:02x}{g_b:02x}{g_b:02x}"


def _summary_table_html(summary: pd.DataFrame) -> str:
    beta_cols = [f"beta_{w}" for w in WINDOWS]
    t_cols = [f"t_{w}" for w in WINDOWS]
    r2_cols = [f"R2_{w}" for w in WINDOWS]
    corr_cols = [f"corr_{w}" for w in WINDOWS]
    beta_limit = max(summary[beta_cols].abs().max().max(), 0.01)
    t_limit = max(summary[t_cols].abs().max().max(), 2.0)
    r2_limit = max(summary[r2_cols].abs().max().max(), 0.05)
    corr_limit = max(summary[corr_cols].abs().max().max(), 0.05)

    def fmt(col: str, v: float) -> str:
        if pd.isna(v):
            return ""
        if col == "obs":
            return f"{int(v)}"
        if col.startswith(("t_", "corr_")):
            return f"{v:.2f}"
        if col.startswith("R2_"):
            return f"{v:.3f}"
        return f"{v:.3f}"

    group = (f'<tr><th rowspan="2">Agency-Coupon</th>'
             + "".join(f'<th colspan="4">{w}d</th>' for w in WINDOWS)
             + '<th rowspan="2">obs</th></tr>')
    sub = "<tr>" + "".join(
        f"<th>beta</th><th>t</th><th>R²</th><th>corr</th>" for _ in WINDOWS) + "</tr>"
    parts = ['<div class="table-wrap"><table class="data-table"><thead>', group, sub,
             "</thead><tbody>"]
    for name, row in summary.iterrows():
        parts.append(f"<tr><td>{html.escape(name)}</td>")
        for w in WINDOWS:
            for prefix, limit in (("beta", beta_limit), ("t", t_limit),
                                  ("R2", r2_limit), ("corr", corr_limit)):
                col = f"{prefix}_{w}"
                style = f' style="background:{_rg_color(row[col], limit)}"'
                parts.append(f"<td{style}>{fmt(col, row[col])}</td>")
        parts.append(f"<td>{fmt('obs', row['obs'])}</td></tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


class _FigBuilder:
    """Embeds figures as JSON blocks; charts render lazily on first visibility."""

    def __init__(self):
        self._count = 0

    def fig_block(self, fig: go.Figure, hidden: bool = False,
                  select: Tuple[str, str] = ("", ""), value: int = 0) -> str:
        self._count += 1
        fid = f"fig-{self._count:03d}"
        js = pio.to_json(fig).replace("</", "<\\/")
        attrs = ""
        if select[0]:
            attrs = f' data-select="{select[0]}" data-value="{select[1]}"'
        style = ' style="display:none"' if hidden else ""
        return (
            f'<div class="chart" id="{fid}"{attrs}{style}></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">{js}</script>'
        )


def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _apply_layout(fig: go.Figure, title: str, height: int = 560,
                  subtitle: Optional[str] = None) -> go.Figure:
    title_text = title.replace("\n", "<br>")
    if subtitle:
        title_text += f'<br><span style="font-size:12px;color:#64748b">{subtitle}</span>'
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(font=dict(size=11)),
    )
    return fig


DAILY_VIEWS = ["FN", "G2", "Both"]            # view selector options for the daily-orig chart


def daily_orig_fig(orig: pd.DataFrame, view: str = "FN") -> go.Figure:
    """Daily origination per coupon over the last RECENT_DAYS trading days.

    view: "FN"/"G2" shows that agency's coupons; "Both" overlays both
    agencies with the agency prefix kept in the trace names.
    """
    if view == "Both":
        cols = list(orig.columns)
        title = f"FN + G2 daily origination - last {RECENT_DAYS} trading days"
    else:
        cols = [c for c in orig.columns if c.startswith(f"{view} ")]
        title = f"{view} daily origination - last {RECENT_DAYS} trading days"
    recent = orig[cols].tail(RECENT_DAYS)
    fig = go.Figure()
    for col in cols:
        s = recent[col]
        latest = orig[col].iloc[-1]
        fig.add_trace(go.Scatter(
            x=_xs(s.index), y=s.values, mode="lines",
            name=f"{col} ({latest:,.0f})", line=dict(width=1.3)))
    _apply_layout(
        fig, title,
        subtitle="Citi securitized MBS origination tracker ($mm per day, NaN filled with 0); "
                 "legend shows latest daily value.")
    fig.update_yaxes(title="Origination ($mm)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    return fig


BAR_VIEWS = ["FN", "G2", "Combined"]           # view selector options for the bar chart


def recent_bar_fig(orig: pd.DataFrame, view: str = "Combined") -> go.Figure:
    """Stacked bars of origination over the most recent BAR_DAYS days.

    view: "FN"/"G2" stack that agency's coupons; "Combined" sums FN and G2
    origination for the same coupon before stacking.
    """
    if view == "Combined":
        cols = []
        frame = pd.DataFrame(index=orig.index)
        for coupon in COUPONS:
            parts = [f"{a} {coupon:g}" for a in AGENCIES]
            parts = [p for p in parts if p in orig.columns]
            if not parts:
                continue
            label = f"{coupon:g}"
            frame[label] = orig[parts].sum(axis=1)
            cols.append(label)
        subtitle = "FN + G2 origination summed per coupon, stacked ($mm per day)"
        title = f"Daily origination - last {BAR_DAYS} trading days (stacked, FN+G2 combined)"
    else:
        cols = [c for c in orig.columns if c.startswith(f"{view} ")]
        frame = orig
        subtitle = f"Stacked by {view} coupon ($mm per day)"
        title = f"Daily origination - last {BAR_DAYS} trading days (stacked, {view})"
    recent = frame[cols].tail(BAR_DAYS)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Bar(x=_xs(recent.index), y=recent[col].values,
                             name=col))
    _apply_layout(fig, title, subtitle=subtitle)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="Origination ($mm)")
    return fig


def rolling_dual_axis_fig(orig: pd.DataFrame, perf: pd.DataFrame,
                          name: str, window: int) -> go.Figure:
    """Rolling-x-day orig sum vs rolling-x-day perf sum, dual axis."""
    ro = orig[name].rolling(window).sum()
    rp = perf[name].rolling(window).sum()
    stats = rolling_ols_stats(orig[name], perf[name], window)
    subtitle = (f"{window}-day rolling sums; full-sample OLS: "
                f"beta={stats['beta']:.3f}, t={stats['t']:.2f}, "
                f"R2={stats['R2']:.3f}, corr={stats['corr']:.2f}"
                if stats else f"{window}-day rolling sums; insufficient history")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_xs(ro.index), y=ro.values, mode="lines",
                             name=f"Orig {window}d sum ($mm)",
                             line=dict(color="#1f77b4", width=1.4)))
    fig.add_trace(go.Scatter(x=_xs(rp.index), y=rp.values, mode="lines",
                             name=f"Perf {window}d sum (32nds)",
                             line=dict(color="#d62728", width=1.3), yaxis="y2"))
    _apply_layout(fig, f"{name} rolling {window}d origination vs performance",
                  subtitle=subtitle)
    fig.update_yaxes(title="Origination ($mm)")
    fig.update_layout(yaxis2=dict(title="Perf (32nds)", overlaying="y", side="right",
                                  showgrid=False))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    return fig


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
_CSS = """
:root{--panel:#fff;--line:#dbe4ea;--muted:#64748b;--accent:#40466e}
*{box-sizing:border-box}body{font:14px Arial,sans-serif;color:#183247;background:#eef3f7;margin:0}
header{background:#12344d;color:#fff;padding:16px 22px}header h1{font-size:20px;margin:0 0 4px}
header .meta{color:#c7d7e3;font-size:12px;margin:0}
main{max-width:1380px;margin:0 auto;padding:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 3px 12px rgba(15,23,42,.06);margin-bottom:22px;overflow:hidden}
.panel h2{font-size:15px;margin:0;padding:12px 15px;border-bottom:1px solid var(--line)}
.panel .body{padding:12px 15px}
.subtitle{font-size:12px;color:var(--muted);margin:2px 0 10px}
.chart{width:100%}
.table-wrap{overflow:auto;max-height:82vh;border:1px solid var(--line);border-radius:6px}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 18px 5px 7px;position:sticky;top:0;text-align:center;white-space:nowrap;cursor:pointer;user-select:none}
.data-table th::after{content:"↕";position:absolute;margin-left:5px;color:#b8c2d8;font-size:10px}
.data-table th.sort-asc::after{content:"▲";color:#fff}.data-table th.sort-desc::after{content:"▼";color:#fff}
.data-table td{border:1px solid #e2e8f0;padding:3px 7px;text-align:center;white-space:nowrap}
.data-table tbody tr:nth-child(even){background:#f0f0f0}
.dropdown-bar{margin:2px 0 12px;font-size:13px}
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:220px;background:#fff}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered='1';Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
const io=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){renderChart(e.target.id);io.unobserve(e.target);}});},{rootMargin:'200px'});
document.querySelectorAll('.chart').forEach(el=>{if(el.style.display==='none')return;if(el.dataset.rendered)return;io.observe(el);});
function updateDualVisibility(){
document.querySelectorAll('.chart[data-select]').forEach(d=>{
const sels=d.dataset.select.split('|'),vals=(d.dataset.value||'').split('|');
const on=sels.every((s,k)=>{const el=document.getElementById(s);return el&&el.value===vals[k];});
d.style.display=on?'':'none';if(on)renderChart(d.id);});}
document.querySelectorAll('select.dd-select').forEach(sel=>{sel.addEventListener('change',updateDualVisibility);});
updateDualVisibility();
function cellNum(t){const n=parseFloat(t.replace(/,/g,''));return t!==''&&!isNaN(n)?n:null;}
document.querySelectorAll('.data-table').forEach(table=>{
const tbody=table.tBodies[0];if(!tbody)return;
table.querySelectorAll('th').forEach((th,col)=>{th.title='Click to sort';th.addEventListener('click',()=>{
const asc=!th.classList.contains('sort-asc');
table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
th.classList.add(asc?'sort-asc':'sort-desc');
const rows=[...tbody.rows];
rows.sort((a,b)=>{const ta=a.cells[col].textContent.trim(),tb=b.cells[col].textContent.trim();
const na=cellNum(ta),nb=cellNum(tb);
if(na!==null&&nb!==null)return asc?na-nb:nb-na;
return asc?ta.localeCompare(tb,undefined,{numeric:true}):tb.localeCompare(ta,undefined,{numeric:true});});
rows.forEach(r=>tbody.appendChild(r));});});});
"""


def _default_report_path() -> Path:
    report_path = OUTPUT_DIR / f"MBS_Origination_{date.today().strftime('%Y%m%d')}.html"
    if report_path.exists():
        try:
            with open(report_path, "ab"):
                pass
        except PermissionError:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = OUTPUT_DIR / f"MBS_Origination_{date.today().strftime('%Y%m%d')}_{ts}.html"
    return report_path


def build_report(output: Optional[Path] = None) -> Tuple[Path, pd.DataFrame]:
    orig = load_origination()
    perf = load_perf()
    structures = [c for c in orig.columns if c in perf.columns]
    if not structures:
        raise ValueError("No agency-coupon common to origination and perf data.")
    summary = build_summary(orig, perf)

    fb = _FigBuilder()

    table_panel = (
        '<div class="panel"><h2>Origination → performance regression summary</h2><div class="body">'
        '<p class="subtitle">Per agency-coupon, OLS of rolling x-day JPM perf sum (32nds) on '
        'rolling x-day origination sum ($mm), common dates, full sample. '
        'Colors: red (negative) / green (positive), scaled per column group.</p>'
        + _summary_table_html(summary) + "</div></div>"
    )

    sel_daily = "sel-daily"
    daily_opts = "".join(
        f'<option value="{v}"{ " selected" if v == "FN" else ""}>{v}</option>'
        for v in DAILY_VIEWS)
    daily_parts = [
        f'<div class="panel"><h2>Daily origination (last {RECENT_DAYS} trading days)</h2><div class="body">'
        '<div class="dropdown-bar"><label>View: '
        f'<select class="dd-select" id="{sel_daily}">{daily_opts}</select></label></div>'
    ]
    for v in DAILY_VIEWS:
        daily_parts.append(fb.fig_block(
            daily_orig_fig(orig, v), hidden=v != "FN",
            select=(sel_daily, v)))
    daily_parts.append("</div></div>")

    sel_bar = "sel-bar"
    bar_opts = "".join(
        f'<option value="{v}"{ " selected" if v == "Combined" else ""}>{v}'
        f'{" (FN+G2 same coupon)" if v == "Combined" else ""}</option>'
        for v in BAR_VIEWS)
    bar_parts = [
        f'<div class="panel"><h2>Recent daily origination (stacked, last {BAR_DAYS}d)</h2><div class="body">'
        '<div class="dropdown-bar"><label>View: '
        f'<select class="dd-select" id="{sel_bar}">{bar_opts}</select></label></div>'
    ]
    for v in BAR_VIEWS:
        bar_parts.append(fb.fig_block(
            recent_bar_fig(orig, v), hidden=v != "Combined",
            select=(sel_bar, v)))
    bar_parts.append("</div></div>")

    # Dual-axis rolling chart: two dropdowns (coupon, window)
    coupon_opts = "".join(
        f'<option value="{i}">{html.escape(c)}</option>'
        for i, c in enumerate(structures))
    window_opts = "".join(
        f'<option value="{w}"{ " selected" if w == 20 else ""}>{w}d</option>'
        for w in WINDOWS)
    sel_coupon, sel_window = "sel-coupon", "sel-window"
    dual_parts = [
        '<div class="panel"><h2>Rolling origination vs rolling performance</h2><div class="body">'
        '<div class="dropdown-bar"><label>Coupon: '
        f'<select class="dd-select" id="{sel_coupon}">{coupon_opts}</select></label> '
        '<label>Window: '
        f'<select class="dd-select" id="{sel_window}">{window_opts}</select></label></div>'
    ]
    for i, name in enumerate(structures):
        for w in WINDOWS:
            hidden = not (i == 0 and w == 20)
            dual_parts.append(fb.fig_block(
                rolling_dual_axis_fig(orig, perf, name, w), hidden=hidden,
                select=(f"{sel_coupon}|{sel_window}", f"{i}|{w}")))
    dual_parts.append("</div></div>")

    if output is None:
        output = _default_report_path()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    latest = min(orig.index[-1], perf.index[-1]).date()
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
<header><h1>{html.escape(REPORT_TITLE)}</h1>
<p class="meta">Latest data {latest} · {len(structures)} agency-coupon series · windows {"/".join(str(w) for w in WINDOWS)}d</p></header>
<main>
{table_panel}
{''.join(daily_parts)}
{''.join(bar_parts)}
{''.join(dual_parts)}
</main>
<script>{_JS}</script>
</body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Report saved to: {output}")
    return output, summary


def print_console_summary(summary: pd.DataFrame) -> None:
    ranked = summary.sort_values("R2_20", ascending=False)
    print("\nStrongest orig->perf relationships (20d window, by R2):")
    for name, row in ranked.head(5).iterrows():
        print(f"  {name:8s} beta={row['beta_20']:+.4f} (t={row['t_20']:+.1f}), "
              f"R2={row['R2_20']:.3f}, corr={row['corr_20']:+.2f}")
    print(f"R2_20 range: {summary['R2_20'].min():.3f} - {summary['R2_20'].max():.3f}")


def main():
    _, summary = build_report()
    print_console_summary(summary)


if __name__ == "__main__":
    main()
