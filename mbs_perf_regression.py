"""Interactive HTML report: daily JPM MBS coupon performance regressed on rates factors.

For each FN/G2 coupon (2.0-6.5), runs full-sample and rolling 120d OLS of daily
JPM DataQuery performance (32nds) on daily changes of the 10y yield, the 2s10s
slope and 1y10y swaption vol, reusing the data loading and predictor conventions
of mbs_rv_report.py.  Output is a single self-contained Plotly HTML file with a
sortable summary table, rolling-beta charts, actual-vs-fitted cumulative perf
charts and a beta heatmap.

Run: .venv/Scripts/python.exe mbs_perf_regression.py
"""

from __future__ import annotations

import html
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

REPORT_TITLE = "MBS Performance Rates Regression"
OUTPUT_DIR = Path(__file__).parent / "mbs_perf_regression_output"

AGENCIES = rv.COUPON_BASIS_AGENCIES          # ["FN", "G2"]
COUPONS = rv.COUPONS                          # 2.0 ... 6.5
PREDICTORS = rv.COUPON_BASIS_PREDICTORS       # ["10y", "2s10s", "1y10y_vol"]
ROLLING_WINDOW = 120                          # days for rolling rates-beta OLS
MIN_OBS = 250                                 # minimum history for a coupon

# Predictors are used as daily changes: yields and slope in bp (x100 of the
# pct-unit diff, matching the ct10_change_bps convention in mbs_rv_report),
# swaption vol in vol points.  Response is daily JPM perf in 32nds.
FACTOR_LABELS = {"d10y": "Δ10y (bp)", "d2s10s": "Δ2s10s (bp)", "dvol": "Δ1y10y vol (vol pts)"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_regression_dataset() -> pd.DataFrame:
    """Daily perf per FN/G2 coupon (32nds) plus daily changes of the factors."""
    dat = rv.load_mbs_rv(rv.FNAME)
    predictors = rv.load_ust_yield_vol_data(rv.FNAME)

    factors = pd.DataFrame(
        {
            "d10y": predictors["10y"].diff() * 100.0,
            "d2s10s": predictors["2s10s"].diff() * 100.0,
            "dvol": predictors["1y10y_vol"].diff(),
        },
        index=predictors.index,
    )

    frames = [factors]
    for agency in AGENCIES:
        for col in rv.get_block_columns(dat, agency, "Perf"):
            frames.append(pd.to_numeric(dat[col], errors="coerce").rename(col))

    df = pd.concat(frames, axis=1).dropna(how="all").sort_index()
    print(f"Regression dataset: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


def perf_structures(df: pd.DataFrame) -> List[str]:
    """Agency-coupon perf columns with sufficient history, sorted FN then G2."""
    cols = [c for c in df.columns if str(c).endswith(" Perf")]
    cols = [c for c in cols if df[c].dropna().shape[0] >= MIN_OBS]
    return sorted(cols, key=rv.coupon_sort_key)


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------
def run_full_ols(df: pd.DataFrame, response_col: str):
    """Full-sample OLS of daily perf on factor changes."""
    sample = df[[response_col, "d10y", "d2s10s", "dvol"]].dropna()
    Y = sample[response_col]
    X = sm.add_constant(sample[["d10y", "d2s10s", "dvol"]], has_constant="add")
    model = sm.OLS(Y, X).fit()
    return sample, model


def run_rolling_betas(df: pd.DataFrame, response_col: str, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling-window OLS betas on all factors over time (one column per factor)."""
    factors = ["d10y", "d2s10s", "dvol"]
    sample = df[[response_col] + factors].dropna()
    dates, rows = [], []
    for end in range(window, len(sample) + 1):
        chunk = sample.iloc[end - window:end]
        X = sm.add_constant(chunk[factors], has_constant="add")
        model = sm.OLS(chunk[response_col], X).fit()
        dates.append(chunk.index[-1])
        rows.append([float(model.params[f]) for f in factors])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=factors)


def build_summary(df: pd.DataFrame, structures: List[str]) -> pd.DataFrame:
    rows = []
    for col in structures:
        sample, model = run_full_ols(df, col)
        resid_std = float(np.std(model.resid, ddof=1))
        latest_resid = float(model.resid.iloc[-1])
        rows.append(
            {
                "structure": col.replace(" Perf", ""),
                "beta_10y": float(model.params["d10y"]),
                "t_10y": float(model.tvalues["d10y"]),
                "beta_2s10s": float(model.params["d2s10s"]),
                "t_2s10s": float(model.tvalues["d2s10s"]),
                "beta_vol": float(model.params["dvol"]),
                "t_vol": float(model.tvalues["dvol"]),
                "R2": float(model.rsquared),
                "resid_std": resid_std,
                "obs": int(model.nobs),
                "latest_resid": latest_resid,
                "latest_resid_z": latest_resid / resid_std if resid_std > 1e-10 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("structure")


# ---------------------------------------------------------------------------
# HTML helpers (conventions mirror mbs_rv_interactive.py)
# ---------------------------------------------------------------------------
def _rg_color(value: float, limit: float) -> str:
    """Diverging red/green: negative red, positive green, white near zero."""
    if pd.isna(value):
        return "#ffffff"
    norm = max(-1.0, min(1.0, value / max(limit, 1e-9)))
    if norm >= 0:
        # white -> green
        g, r_b = 180, int(255 - 155 * norm)
        return f"#{r_b:02x}{g:02x}{r_b:02x}"
    r, g_b = 220, int(255 - 155 * (-norm))
    return f"#{r:02x}{g_b:02x}{g_b:02x}"


def _summary_table_html(summary: pd.DataFrame) -> str:
    beta_cols = ["beta_10y", "beta_2s10s", "beta_vol"]
    z_cols = ["latest_resid", "latest_resid_z"]
    beta_limit = max(summary[beta_cols].abs().max().max(), 0.01)
    z_limit = max(summary[z_cols].abs().max().max(), 0.5)

    def fmt(col: str, v: float) -> str:
        if pd.isna(v):
            return ""
        if col == "obs":
            return f"{int(v)}"
        if col in ("t_10y", "t_2s10s", "t_vol", "latest_resid_z"):
            return f"{v:.2f}"
        if col == "R2":
            return f"{v:.3f}"
        return f"{v:.2f}"

    headers = ["Agency-Coupon"] + list(summary.columns)
    parts = ['<div class="table-wrap"><table class="data-table"><thead><tr>']
    parts += [f"<th>{html.escape(h)}</th>" for h in headers]
    parts.append("</tr></thead><tbody>")
    for name, row in summary.iterrows():
        parts.append(f"<tr><td>{html.escape(name)}</td>")
        for col in summary.columns:
            style = ""
            if col in beta_cols:
                style = f' style="background:{_rg_color(row[col], beta_limit)}"'
            elif col in z_cols:
                style = f' style="background:{_rg_color(row[col], z_limit)}"'
            parts.append(f"<td{style}>{fmt(col, row[col])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


class _FigBuilder:
    """Embeds figures as JSON blocks; charts render lazily on first visibility."""

    def __init__(self):
        self._count = 0

    def fig_block(self, fig: go.Figure, hidden: bool = False, select: str = "", value: int = 0) -> str:
        self._count += 1
        fid = f"fig-{self._count:03d}"
        js = pio.to_json(fig).replace("</", "<\\/")
        attrs = ""
        if select:
            attrs = f' data-select="{select}" data-value="{value}"'
        style = ' style="display:none"' if hidden else ""
        return (
            f'<div class="chart" id="{fid}"{attrs}{style}></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">{js}</script>'
        )


def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _apply_layout(fig: go.Figure, title: str, height: int = 560, subtitle: Optional[str] = None) -> go.Figure:
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


_FACTOR_META = [
    ("d10y", "rates (Δ10y)", "beta_10y (32nds per bp)"),
    ("d2s10s", "curve (Δ2s10s)", "beta_2s10s (32nds per bp)"),
    ("dvol", "vol (Δ1y10y)", "beta_vol (32nds per vol pt)"),
]


def rolling_beta_fig(agency: str, betas: Dict[str, pd.DataFrame]) -> go.Figure:
    """Rolling betas for every factor, switchable via a dropdown."""
    cols = sorted(betas, key=rv.coupon_sort_key)
    fig = go.Figure()
    for f_idx, (factor, _label, _axis) in enumerate(_FACTOR_META):
        for col in cols:
            series = betas[col][factor]
            fig.add_trace(go.Scatter(x=_xs(series.index), y=series.values, mode="lines",
                                     name=col.replace(" Perf", ""), line=dict(width=1.4),
                                     visible=(f_idx == 0),
                                     legendgroup=col, showlegend=(f_idx == 0)))
    n = len(cols)
    buttons = []
    for f_idx, (_factor, label, axis) in enumerate(_FACTOR_META):
        active = [f_idx == j // n for j in range(len(_FACTOR_META) * n)]
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": active, "showlegend": active},
                                  {"yaxis.title.text": axis}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right",
                                        y=1.14, yanchor="top", showactive=True,
                                        bgcolor="#ffffff", bordercolor="#b6c5cf")])
    _apply_layout(fig, f"{agency} rolling {ROLLING_WINDOW}d betas",
                  subtitle=f"OLS betas on daily factor changes, {ROLLING_WINDOW}-day window; perf in 32nds. "
                           "Use the dropdown to switch factor.")
    fig.update_yaxes(title=_FACTOR_META[0][2])
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    return fig


def cumulative_fit_fig(df: pd.DataFrame, col: str) -> go.Figure:
    sample, model = run_full_ols(df, col)
    actual = sample[col].cumsum()
    fitted = model.fittedvalues.cumsum()
    resid = (sample[col] - model.fittedvalues)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_xs(actual.index), y=actual.values, mode="lines",
                             name="Actual cum perf", line=dict(color="#d62728", width=1.3)))
    fig.add_trace(go.Scatter(x=_xs(fitted.index), y=fitted.values, mode="lines",
                             name="Fitted cum perf", line=dict(color="#1f77b4", width=1.3)))
    resid_recent = resid.tail(ROLLING_WINDOW)
    fig.add_trace(go.Bar(x=_xs(resid_recent.index), y=resid_recent.values,
                         name=f"Daily resid (last {ROLLING_WINDOW}d)", marker_color="#94a3b8",
                         opacity=0.6, yaxis="y2"))
    _apply_layout(fig, f"{col.replace(' Perf', '')} actual vs fitted cumulative performance",
                  subtitle=f"Full-sample OLS, R2={model.rsquared:.3f}; bars: daily residuals, last {ROLLING_WINDOW}d (right axis)")
    fig.update_yaxes(title="Cumulative perf (32nds)")
    fig.update_layout(yaxis2=dict(title="Daily resid (32nds)", overlaying="y", side="right",
                                  showgrid=False, zeroline=False))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    return fig


def beta_heatmap_fig(summary: pd.DataFrame) -> go.Figure:
    labels = list(summary.index)
    factors = ["beta_10y", "beta_2s10s", "beta_vol"]
    z = summary[factors].values
    text = [[f"{v:.3f}" for v in row] for row in z]
    limit = float(np.nanmax(np.abs(z)))
    fig = go.Figure(go.Heatmap(
        z=z, x=[FACTOR_LABELS[{"beta_10y": "d10y", "beta_2s10s": "d2s10s", "beta_vol": "dvol"}[f]] for f in factors],
        y=labels, text=text, texttemplate="%{text}",
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-limit, zmax=limit,
        colorbar=dict(title="beta"),
    ))
    _apply_layout(fig, "Full-sample factor betas by coupon", height=max(420, 26 * len(labels) + 160),
                  subtitle="OLS of daily JPM perf (32nds) on factor daily changes")
    fig.update_yaxes(autorange="reversed")
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
document.querySelectorAll('select.dd-select').forEach(sel=>{sel.addEventListener('change',()=>{
document.querySelectorAll('.chart[data-select="'+sel.id+'"]').forEach(d=>{const on=d.dataset.value===sel.value;d.style.display=on?'':'none';if(on)renderChart(d.id);});});});
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
    report_path = OUTPUT_DIR / f"MBS_Perf_Regression_{date.today().strftime('%Y%m%d')}.html"
    if report_path.exists():
        try:
            with open(report_path, "ab"):
                pass
        except PermissionError:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = OUTPUT_DIR / f"MBS_Perf_Regression_{date.today().strftime('%Y%m%d')}_{ts}.html"
    return report_path


def build_report(output: Optional[Path] = None) -> Tuple[Path, pd.DataFrame]:
    df = build_regression_dataset()
    structures = perf_structures(df)
    if not structures:
        raise ValueError("No coupon perf series with sufficient history.")
    summary = build_summary(df, structures)

    rolling: Dict[str, pd.DataFrame] = {col: run_rolling_betas(df, col) for col in structures}
    fb = _FigBuilder()

    # Summary table panel
    table_panel = (
        '<div class="panel"><h2>Full-sample OLS summary</h2><div class="body">'
        '<p class="subtitle">Daily JPM perf (32nds) ~ const + Δ10y (bp) + Δ2s10s (bp) + Δ1y10y vol (vol pts). '
        'Betas colored red (negative) / green (positive); latest residual = today\'s unexplained perf, '
        'z-scored on full-sample residual std.</p>'
        + _summary_table_html(summary) + "</div></div>"
    )

    # Rolling beta panels, one per agency
    beta_panels = []
    for agency in AGENCIES:
        agency_betas = {c: rolling[c] for c in structures if c.startswith(f"{agency} ")}
        if agency_betas:
            beta_panels.append(
                '<div class="panel"><h2>Rolling betas (rates / curve / vol)</h2><div class="body">'
                + fb.fig_block(rolling_beta_fig(agency, agency_betas)) + "</div></div>"
            )

    # Actual vs fitted dropdown panel
    options = "".join(
        f'<option value="{i}">{html.escape(c.replace(" Perf", ""))}</option>'
        for i, c in enumerate(structures)
    )
    fit_parts = [
        '<div class="panel"><h2>Actual vs fitted cumulative performance</h2><div class="body">'
        '<div class="dropdown-bar"><label>Coupon: '
        f'<select class="dd-select" id="sel-fit">{options}</select></label></div>'
    ]
    for i, col in enumerate(structures):
        fit_parts.append(fb.fig_block(cumulative_fit_fig(df, col), hidden=i != 0, select="sel-fit", value=i))
    fit_parts.append("</div></div>")

    heatmap_panel = (
        '<div class="panel"><h2>Factor beta heatmap</h2><div class="body">'
        + fb.fig_block(beta_heatmap_fig(summary)) + "</div></div>"
    )

    if output is None:
        output = _default_report_path()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    latest = df.index[-1].date()
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
<p class="meta">Latest data {latest} · {len(structures)} coupon series · rolling window {ROLLING_WINDOW}d</p></header>
<main>
{table_panel}
{''.join(beta_panels)}
{''.join(fit_parts)}
{heatmap_panel}
</main>
<script>{_JS}</script>
</body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Report saved to: {output}")
    return output, summary


def print_console_summary(summary: pd.DataFrame) -> None:
    ranked = summary.sort_values("beta_10y")
    print("\nRates beta (beta_10y, 32nds per bp) - lowest / highest:")
    for name, row in list(ranked.head(3).iterrows()) + list(ranked.tail(3).iterrows()):
        print(f"  {name:10s} beta_10y={row['beta_10y']:+.3f} (t={row['t_10y']:+.1f}, R2={row['R2']:.3f})")
    print(f"R2 range: {summary['R2'].min():.3f} - {summary['R2'].max():.3f}")
    extremes = summary.reindex(summary["latest_resid_z"].abs().sort_values(ascending=False).index)
    print("Latest residual z highlights:")
    for name, row in extremes.head(4).iterrows():
        print(f"  {name:10s} resid={row['latest_resid']:+.2f} (z={row['latest_resid_z']:+.2f})")


def main():
    _, summary = build_report()
    print_console_summary(summary)


if __name__ == "__main__":
    main()
