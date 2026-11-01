"""Selectable JPM performance and rolling fixed-coupon yield-basis views."""
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mbs_rv_report as rv


def performance_payload(dat):
    dat = dat.sort_index()
    latest = dat.index.max()
    start = latest.replace(month=1, day=1).normalize()
    year = dat.loc[dat.index >= start]
    series = {}
    for agency in ("FN", "G2"):
        for coupon in rv.COUPONS:
            col = rv._col(agency, coupon, "Perf")
            if col in year and year[col].notna().any():
                series[f"{agency} {coupon:g}"] = [None if pd.isna(v) else float(v) for v in year[col]]
    return dict(dates=year.index.strftime("%Y-%m-%d").tolist(), series=series,
                minimum=start.strftime("%Y-%m-%d"), maximum=latest.strftime("%Y-%m-%d"),
                default=latest.replace(day=1).strftime("%Y-%m-%d"))


def add_performance(builder, dat):
    payload = performance_payload(dat)
    encoded = json.dumps(payload).replace("</", "<\\/")
    body = f'''<div class="dropdown-bar"><label>Start date (inclusive):
      <input type="date" id="jpm-perf-start" min="{payload['minimum']}" max="{payload['maximum']}" value="{payload['default']}"></label>
      <label>Agency: <select id="jpm-perf-agency"><option>FN</option><option>G2</option></select></label></div>
      <p class="view-subtitle">Cumulative JPM hedged daily performance in ticks (32nds), not raw price change or dollar P&amp;L.
      Includes the selected start date's daily performance. Available from January 1 of the latest data year.
      Missing observations are shown as gaps; cumulative values thereafter are unavailable rather than treating missing performance as zero.</p>
      <div class="chart" id="jpm-perf-chart"></div><p id="jpm-perf-status"></p>
      <script type="application/json" id="jpm-perf-data">{encoded}</script>'''
    builder.add_html("JPM Coupon Cumulative Performance — Select Start Date", "Performance", body)


def add_rolling(builder, basis, table_formatter):
    if basis is None:
        return
    history = rv.run_rolling_coupon_basis(basis, rv.CC_ROLLING_WINDOW)
    if history.empty:
        return
    options = []
    summary = []
    for name in sorted(history.structure.unique(), key=rv.coupon_sort_key):
        sub = history.loc[history.structure == name].sort_index()
        latest = sub.iloc[-1]
        state = "Cheap" if latest.resid > 0 else "Rich" if latest.resid < 0 else "Fair"
        summary.append(dict(structure=name, date=sub.index[-1].date(), actual_bps=latest.mkt_val,
                            fitted_bps=latest.mdl_val, residual_bps=latest.resid,
                            residual_z=latest.resid_z, r2=latest.r2, valuation=state))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=.07)
        for col, label, color in [("mkt_val", "Actual yield basis", "firebrick"), ("mdl_val", "120-observation fitted basis", "royalblue")]:
            fig.add_trace(go.Scatter(x=sub.index, y=sub[col], name=label, line=dict(color=color)), row=1, col=1)
        for multiple in (2, -2):
            fig.add_trace(go.Scatter(x=sub.index, y=multiple*sub.resid_std,
                                    name=f"{multiple:+d} residual std", line=dict(color="#94a3b8", dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=sub.index, y=sub.resid, customdata=sub.resid_z,
                                name="Residual: actual minus fitted", fill="tozeroy",
                                hovertemplate="%{x|%Y-%m-%d}<br>Residual %{y:.2f} bps<br>Residual z %{customdata:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_color="black", line_width=1)
        ztext = f"{latest.resid_z:+.2f}" if np.isfinite(latest.resid_z) else "unavailable"
        fig.update_layout(template="plotly_white", height=760, hovermode="x unified",
                          title=dict(text=f"{html.escape(name)}: 120-Observation Rolling Yield-Basis OLS<br><sup>"
                                     f"{sub.index[-1].date()} | Actual {latest.mkt_val:.2f} | Fitted {latest.mdl_val:.2f} bps | "
                                     f"Residual {latest.resid:+.2f} bps | z {ztext} | {state}</sup>"),
                          margin=dict(t=105), legend=dict(orientation="h", y=-.15))
        fig.update_yaxes(title_text="Yield basis (bps)", row=1, col=1)
        fig.update_yaxes(title_text="Residual (bps)", row=2, col=1)
        options.append((name, fig))
    builder.add_table("Fixed-Coupon Basis: Latest 120-Observation Rich/Cheap", "Current-coupon basis",
                      table_formatter(pd.DataFrame(summary), include_index=False,
                                                                  gradient_columns=["residual_z"]),
                      subtitle="Basis = coupon yield minus CT10, in bps. Positive residual = cheap; negative = rich. "
                      "Intercept + 10Y, 2s10s and 1y10y vol. Each fit includes its endpoint, matching CC OLS; "
                      "z = endpoint residual / population std of that fit's 120 in-window residuals. "
                      "These are model-relative valuations, not out-of-sample trading signals. Fixed coupons avoid changing CC interpolation weights.")
    builder.add_dropdown("Fixed-Coupon Basis: 120-Observation Rolling Actual, Fitted and Residual",
                         "Current-coupon basis", options, label="Agency + coupon")


JS = r'''
const jpmDataNode=document.getElementById('jpm-perf-data');
if(jpmDataNode){
 const source=JSON.parse(jpmDataNode.textContent),start=document.getElementById('jpm-perf-start'),agency=document.getElementById('jpm-perf-agency');
 function updateJpmPerformance(){
  if(!start.value||start.value<source.minimum||start.value>source.maximum){document.getElementById('jpm-perf-status').textContent='Choose a date within the available year.';return;}
  const first=source.dates.findIndex(d=>d>=start.value),dates=first<0?[]:source.dates.slice(first),traces=[];
  Object.entries(source.series).filter(([name])=>name.startsWith(agency.value+' ')).forEach(([name,values])=>{
   let total=0,complete=true;
   const y=(first<0?[]:values.slice(first)).map(v=>{if(v===null)complete=false;if(!complete)return null;total+=v;return total;});
   traces.push({x:dates,y,name,type:'scatter',mode:'lines',connectgaps:false,hovertemplate:'%{x}<br>'+name+': %{y:.2f} ticks<extra></extra>'});
  });
  Plotly.react('jpm-perf-chart',traces,{template:'plotly_white',height:600,title:agency.value+' cumulative JPM performance from '+start.value,yaxis:{title:'Cumulative performance (ticks / 32nds)',zeroline:true},xaxis:{title:'Date'},hovermode:'x unified'},{responsive:true,displaylogo:false});
  document.getElementById('jpm-perf-status').textContent=dates.length?'First included observation: '+dates[0]+' | Latest: '+source.maximum:'No observations in selected range.';
 }
 start.addEventListener('change',updateJpmPerformance);agency.addEventListener('change',updateJpmPerformance);updateJpmPerformance();
}
'''
