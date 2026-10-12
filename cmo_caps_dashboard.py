"""Self-contained, single-chart explorer for the CMO cap valuation model."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


def refresh_html(out_path: Path) -> None:
    """Rebuild presentation from a report's saved model payload, without refitting."""
    previous = out_path.read_text(encoding="utf-8")
    marker = '<script type="application/json" id="data">'
    start = previous.index(marker)+len(marker)
    payload = json.loads(previous[start:previous.index('</script>', start)])
    history_cache = pd.read_csv(out_path.parent / "cmo_caps_data.csv", index_col=0, parse_dates=True)
    for item in payload:
        history = pd.to_numeric(history_cache[item["ticker"]], errors="coerce").dropna()
        item["price_history"] = {"dates": history.index.strftime("%Y-%m-%d").tolist(), "prices": _values(history)}
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    out_path.write_text(_PAGE.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", raw).replace("__COMPARISON__", _comparison_json({b['ticker'] for b in payload})), encoding="utf-8")
    print(f"Updated HTML using saved model results: {out_path}")


def _values(series):
    return [float(v) if pd.notna(v) and np.isfinite(v) else None for v in series]


def _comparison_json(tickers):
    from cmo_price_comparison import load_comparison
    return json.dumps(load_comparison(tickers), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def build_dashboard(results: dict, bonds: list, out_path: Path,
                    summary_name: str = "cmo_caps_model_summary.csv") -> None:
    payload = []
    summary = []
    history_cache = pd.read_csv(out_path.parent / "cmo_caps_data.csv", index_col=0, parse_dates=True)
    for ticker, cap, agency in bonds:
        r = results[ticker]
        data = r["data"]
        item = {"ticker": ticker, "label": ticker.replace(" Mtge", ""),
                "cap": cap, "agency": agency, "cc": r["cc_col"],
                "group": r.get("group", "analysis"),
                "dates": data.index.strftime("%Y-%m-%d").tolist(),
                "series": {c: _values(data[c]) for c in data}, "models": {},
                "validation": r.get("validation", {}), "error": r.get("error")}
        item["reference"] = r.get("reference", {"flt_spread": None, "asof": "", "status": "Unavailable"})
        history = r.get("price_history", pd.to_numeric(history_cache[ticker], errors="coerce").dropna())
        item["price_history"] = {"dates": history.index.strftime("%Y-%m-%d").tolist(), "prices": _values(history)}
        if "baseline_fit" in r:
            item["baseline"] = _values(data.px-r["baseline_fit"])
        for name, m in r.get("models", {}).items():
            item["models"][name] = {
                "fit": _values(m["fit"]), "resid": _values(m["resid"]),
                "sigma": _values(m["sigma_path"]) if "sigma_path" in m else [float(m["sigma"])]*len(data),
                "n": m["n"], "start": str(m["start"].date()), "end": str(m["end"].date()),
                "r2": float(m["r2"]), "adj_r2": float(m["adj_r2"]),
                "condition": m["condition"], "a": m["a"],
                "b": {k: float(v) for k, v in m["b"].items()},
            }
            residual = m["resid"].iloc[-1]
            sigma = m["sigma"]
            summary.append({"ticker": ticker, "cap": cap, "model": name,
                            "group": r.get("group", "analysis"),
                            "date": item["dates"][-1], "n_train": m["n"],
                            "training_end": item["models"][name]["end"],
                            "price": data.px.iloc[-1], "fair": m["fit"].iloc[-1],
                            "residual_ticks": residual*32,
                            "z": residual/sigma if sigma > 1e-12 else 0,
                            "adj_r2": m["adj_r2"], "flt_spread": item["reference"]["flt_spread"],
                            "flt_spread_asof": item["reference"]["asof"],
                            "mtg_collat_typ": item["reference"].get("mtg_collat_typ"),
                            "underlying_cpn": item["reference"].get("underlying_cpn"),
                            "collateral_asof": item["reference"].get("collateral_asof", ""),
                            **r.get("validation", {})})
        payload.append(item)
    pd.DataFrame(summary).to_csv(out_path.with_name(summary_name), index=False)
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = _PAGE.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", raw).replace("__COMPARISON__", _comparison_json({b['ticker'] for b in payload}))
    out_path.write_text(page, encoding="utf-8")
    print(f"HTML written: {out_path}")


_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMO caps · valuation explorer</title><script>__PLOTLY__</script>
<style>
:root{--navy:#163449;--ink:#203445;--muted:#667c8d;--line:#dce5ed}
*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:var(--ink);font:14px/1.5 'Segoe UI',Arial,sans-serif}
main{max-width:1450px;margin:auto;padding:26px}header{margin:4px 0 22px}h1{font-size:29px;margin:0}h2{font-size:18px;margin:0 0 12px}.muted{color:var(--muted);font-size:12px}
.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 3px 12px #142c4010}
.controls{display:flex;gap:16px;flex-wrap:wrap}.controls label{font-size:12px;color:var(--muted)}select,input{display:block;padding:9px;border:1px solid #bdcddb;border-radius:6px;background:#fff;color:var(--ink);font:14px 'Segoe UI',Arial}select{min-width:150px}#ticker{min-width:260px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:18px}.card{background:#f3f7fa;border-radius:8px;padding:13px}.card strong{display:block;font-size:22px}.cheap{color:#167250}.rich{color:#bc4252}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th{background:var(--navy);color:white;font-weight:500;text-align:right;padding:11px}td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{text-align:left}tr.selected td{background:#e9f3fa}tbody tr{cursor:pointer}tbody tr:hover td{background:#f1f6fa}
#chart{min-height:470px}details p{max-width:1000px}summary{cursor:pointer;font-weight:600}code{background:#eff4f8;padding:2px 5px}.warning{color:#a56217}#diagnostics{display:flex;gap:20px;flex-wrap:wrap}#scenario-tables h3{margin-top:28px;color:var(--navy)}#scenario-tables h4{margin:18px 0 8px}
@media(max-width:700px){main{padding:10px}.panel{padding:12px}h1{font-size:23px}}
</style></head><body><main>
<header><div class="muted">AGENCY CMO · COUPON CAP WATCHLIST</div><h1>CMO valuation explorer</h1><p id="asof" class="muted"></p></header>
<section class="panel"><h2>Price change comparison</h2>
<div class="controls"><label>Start date<input id="perf-start" type="date" value="2026-08-01"></label><label>End date<input id="perf-end" type="date" value="2026-10-02"></label></div>
<div class="controls"><label>Cap<select id="perf-cap"><option value="6">6.0%</option><option value="6.5">6.5%</option><option value="7">7.0%</option><option value="7.5">7.5%</option><option value="all">All caps</option></select></label>
<span id="perf-universe" class="muted"></span>
<label>Plot<select id="perf-view"><option value="bar">Endpoint price changes · every bond</option><option value="line">Daily change from selected start</option></select></label>
<span class="muted">Dollar price change per $100 face value</span></div>
<p class="muted">The comparison uses this report's own bond universe. Select any dates within the embedded history. The common comparison dates are the first and last dates inside the requested range with prices for all four FN/G2 benchmarks. Each bond must have actual prices on both common dates; missing endpoints are excluded, not filled. Gaps inside the daily series stay blank. Price change excludes coupon income, financing, hedging and principal cash flows. TBA prices are rolling generic series, not fixed delivery-month total returns.</p>
<div id="perf-chart"></div><p id="perf-status" class="muted"></p><div id="perf-tables" class="scroll"></div>
<p class="muted">Changes and excess changes are dollars per $100 face value: a price move from 100 to 98 is −$2.00. Positive excess = floater price change minus benchmark price change &gt; 0 (not duration-adjusted). Bond source: Bloomberg PX_LAST cache. Benchmark source: mbs_rv.xlsx / first sheet / FN 5.5, FN 6, G2 5.5 and G2 6 Price columns. Valuation-model factors remain direct Bloomberg data.</p></section>
<div id="forward-indicator"></div>
<section class="panel"><h2>Current watchlist</h2><div class="controls">
<label>Group<select id="group"><option value="all">All bonds</option><option value="portfolio">Portfolio</option><option value="analysis">Analysis universe</option></select></label>
<label>Show coupon cap<select id="cap"><option value="all">All caps</option><option value="6">6.0%</option><option value="6.5">6.5%</option><option value="7">7.0%</option><option value="7.5">7.5%</option></select></label>
<label>Watchlist valuation model<select id="watch-model"><option value="full">Full history · descriptive fit</option><option value="rolling">Prior-window · up to 120 observations</option></select></label>
</div><p class="muted">These controls apply to the watchlist table.</p>
<div class="scroll"><table><thead><tr><th>Bond</th><th>Group</th><th>Cap</th><th>MTG_COLLAT_TYP</th><th>Underlying cpn (%)</th><th>FLT_SPREAD</th><th>As of</th><th>Train n</th><th>Price</th><th>Model value</th><th>Residual (ticks)</th><th>z</th><th>Train adj. R²</th><th>2F RMSE (ticks)</th><th>4F RMSE (ticks)</th><th>RMSE improvement</th></tr></thead><tbody id="watchlist"></tbody></table></div><p class="muted">Click a bond to select it below. Negative residual = below model; positive = above model. Underlying cpn is the pass-through coupon parsed from Bloomberg MTG_COLLAT_TYP. FLT_SPREAD and collateral type are Bloomberg reference data; hover a value for its snapshot time. A dash means unavailable. Held-out RMSE uses identical dates for both models; positive improvement favors four factors.</p></section>
<section class="panel"><h2>Single-bond analysis</h2><div class="controls">
<label>Ticker<select id="ticker"></select></label>
<label>Selected bond model<select id="model"><option value="full">Full history · descriptive fit</option><option value="rolling">Prior-window · up to 120 observations</option></select></label>
<label>Chart<select id="view"><option value="price">Price and model value</option><option value="residual">Residual and ±1/2σ</option><option value="factors">Market factors</option><option value="scatter">Actual versus fitted price</option><option value="validation">Held-out errors · 2 vs 4 factors</option></select></label>
</div><div id="cards" class="cards"></div><p id="status" class="muted"></p>
<div id="chart"></div><div id="diagnostics" class="muted"></div><p class="muted">These controls apply to the selected bond's chart and metrics. Read is neutral inside ±1σ. Drag to zoom; double-click to reset. Hover shows the observation date.</p></section>
<section class="panel"><h2>Factor shocks · bonds by cap</h2><div class="controls">
<label>Coupon cap<select id="shock-cap"><option value="6">6.0%</option><option value="6.5">6.5%</option><option value="7">7.0%</option><option value="7.5">7.5%</option></select></label>
<label>Group<select id="shock-group"><option value="all">All bonds</option><option value="portfolio">Portfolio</option><option value="analysis">Analysis universe</option></select></label>
<label>CT10 change (bp)<input id="shock-rate" type="number" step="any" value="20"></label>
<label>CC basis change (bp)<input id="shock-cc" type="number" step="any" value="10"></label>
<label>Scenario valuation model<select id="scenario-model"><option value="full">Full history · descriptive fit</option><option value="rolling">Prior-window · up to 120 observations</option></select></label>
</div><p class="muted">Enter signed changes independently: positive means up, negative means down. For example, CT10 −20bp with CC basis +10bp. Every bond matching this cap and group is shown, independently of the watchlist and plot selections. 2s10s and 1y10y vol are unchanged.</p>
<div id="scenario-tables"></div><p class="muted">Contributions and total impact are in ticks (1/32 price point). Shocked price = observed price + model impact, holding today's residual fixed. Shocked model value = current model value + impact. Missing models are shown as unavailable. These are local linear estimates.</p></section>
<section class="panel"><details><summary>Method, validation and data</summary>
<p>Four-factor OLS: price = intercept + 10Y yield + agency CC basis + 2s10s + 1y10y ATM normal volatility. 2s10s is (10Y − 2Y) in bp. Fannie/Freddie use FN CC; Ginnie uses G2 CC. Yield is in percent; CC basis, curve and normal vol are in bp. Price is in points; one tick is 1/32 point.</p>
<p>Full history fits all complete observations including today's price, so its residual describes relative value within that sample. The prior-window model uses up to 120 complete observations strictly before each evaluated date, with at least 60 training observations. Its current fair value excludes today's price from training. The 120 count refers to observations, not calendar days.</p>
<p>Both models use scaled SVD least squares. Residual z is actual minus model divided by training residual standard error, using n − 5 degrees of freedom. Bands show residual scale, not a formal prediction interval. Training adjusted R² is not an out-of-sample score. A level regression can reflect common trends; short samples and collinearity can make sensitivities unstable.</p>
<p>Validation compares CT10 + CC basis against the enhanced model on the same prior-window training and held-out dates. Each evaluated date uses that day's market factors with coefficients fitted before that date; it measures contemporaneous valuation error, not a forecast of tomorrow's price. RMSE improvement = 100 × (1 − enhanced RMSE / baseline RMSE). No hyperparameters are selected from this comparison.</p>
<p>Valuation inputs: direct Bloomberg PX_LAST for bonds, USGG10YR/USGG2YR Index, .FNCC105/.G2CC105 G Index and USSNA110 FMCA Curncy (USD SOFR 1Y10Y ATM normal volatility, annualized bp). No workbook volatility or date-filling is used. Incomplete rows are excluded; every bond has its own observation and training dates.</p>
<p>All valuation histories are requested from 1 Jan 2023, where available. Raw prices remain separate from model inputs. Full-history residuals start at the first complete observation; prior-window residuals require at least 60 earlier complete observations. New bonds are not backfilled before their histories begin.</p>
<p>Volatility data treatment: the reviewed abnormal USSNA110 FMCA observations on 22 Jun 2023 (57.348) and 10 Jun 2024 (62.000) are replaced by the previous available observations (101.506 and 101.964). Raw quotes and a replacement audit are retained. Only these reviewed dates are replaced; other missing data are not filled, and no future values are used.</p>
</details></section></main>
<script type="application/json" id="comparison">__COMPARISON__</script>
<script type="application/json" id="data">__DATA__</script><script>
const bonds=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const comparison=JSON.parse(document.getElementById('comparison').textContent);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=2)=>n===null||n===undefined||!Number.isFinite(n)?'—':n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const last=a=>a?.length?a[a.length-1]:null;
function comparisonWindow(start,end){
 if(!/^\d{4}-\d{2}-\d{2}$/.test(start)||!/^\d{4}-\d{2}-\d{2}$/.test(end)||start>end)return {error:'Choose a valid start date on or before the end date.'};
 const indices=comparison.dates.map((d,i)=>i).filter(i=>comparison.dates[i]>=start&&comparison.dates[i]<=end&&comparison.benchmarks.every(b=>Number.isFinite(b.prices[i])));
 if(indices.length<2)return {error:'Fewer than two common benchmark observations in the selected range. Choose dates within the available history.'};
 const first=indices[0],final=indices.at(-1),dates=comparison.dates.slice(first,final+1);
 const calculate=b=>{const start_price=b.prices[first],end_price=b.prices[final],ok=Number.isFinite(start_price)&&Number.isFinite(end_price);return {...b,prices:b.prices.slice(first,final+1),start_price,end_price,change:ok?end_price-start_price:null,status:ok?'Available':'Missing endpoint price'};};
 const benchmarks=comparison.benchmarks.map(calculate),rows=comparison.bonds.map(calculate);
 rows.forEach(b=>b.excess=Object.fromEntries(benchmarks.map(k=>[k.label,b.change===null?null:b.change-k.change])));
 return {start:dates[0],end:dates.at(-1),dates,benchmarks,bonds:rows};
}
function forwardIndicator(){
 const indicator=comparison.indicator;
 if(!indicator?.dates.length){$('forward-indicator').innerHTML='<section class="panel"><h2>1Y1Y Bloomberg indicator</h2><p class="warning">No cached history available for SD0490FS 1Y1Y BLC Curncy.</p></section>';return;}
 const dates=indicator.dates,values=indicator.values,i=values.length-1;
 $('forward-indicator').innerHTML=`<section class="panel"><h2>${esc(indicator.ticker)}</h2><p class="muted">Requested from ${indicator.start} to latest available. Actual observations ${dates[0]} to ${dates[i]}. Bloomberg PX_LAST in original quoted units; no rescaling. Standalone market indicator, not included in the OLS valuation model.</p><div id="forward-chart"></div><p class="muted">Latest ${dates[i]}: ${fmt(values[i],4)} · change since ${dates[0]}: ${fmt(values[i]-values[0],4)}. This chart is independent of the price-comparison date controls.</p></section>`;
 Plotly.react('forward-chart',[{x:dates,y:values,type:'scatter',mode:'lines',name:indicator.ticker,line:{color:'#187f76',width:2},connectgaps:false,hovertemplate:'%{x}<br>PX_LAST %{y:.4f}<extra></extra>'}],{height:420,margin:{l:65,r:100,t:35,b:55},xaxis:{title:'Date'},yaxis:{title:'PX_LAST (Bloomberg quoted units)'},annotations:[{x:dates[i],y:values[i],text:`Latest ${fmt(values[i],4)}`,showarrow:true,arrowhead:2}]},{responsive:true,displaylogo:false});
}
function performance(){
 $('perf-universe').textContent=`${comparison.bonds.length} ${comparison.bonds.every(b=>b.group==='portfolio')?'portfolio bonds only':'analysis-universe floaters'}`;
 const window=comparisonWindow($('perf-start').value,$('perf-end').value);
 if(window.error){$('perf-status').textContent=window.error;$('perf-tables').innerHTML='';Plotly.react('perf-chart',[],{height:500,annotations:[{text:window.error,xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]},{responsive:true,displaylogo:false});return;}
 const candidates=window.bonds.filter(b=>$('perf-cap').value==='all'||b.cap===Number($('perf-cap').value));
 const rows=candidates.filter(b=>b.change!==null),benchmarks=window.benchmarks;
 const scale=1,unit='$ / $100 face',colors=['#172f50','#ab4055','#168779','#a77310'];
 const dollar=n=>n===null||n===undefined||!Number.isFinite(n)?'—':`${n<0?'−':n>0?'+':''}$${fmt(Math.abs(n),2)}`;
 const metrics=b=>`<td>${fmt(b.start_price,3)}</td><td>${fmt(b.end_price,3)}</td><td>${dollar(b.change)}</td>`;
 $('perf-tables').innerHTML='<h3>Four benchmark coupons</h3><table><thead><tr><th>Benchmark</th><th>'+window.start+' price</th><th>'+window.end+' price</th><th>Δ ('+unit+')</th><th>Coverage</th></tr></thead><tbody>'+benchmarks.map(b=>`<tr><td>${esc(b.label)}</td>${metrics(b)}<td>${esc(b.status)}</td></tr>`).join('')+'</tbody></table><h3>Every bond by cap</h3>'+[...new Set(rows.map(b=>b.cap))].sort((a,b)=>a-b).map(cap=>`<h4>Cap ${fmt(cap,1)}%</h4><table><thead><tr><th>Bond</th><th>Agency / group</th><th>${window.start} price</th><th>${window.end} price</th><th>Δ (${unit})</th>${benchmarks.map(b=>`<th>Excess vs ${esc(b.label)} (${unit})</th>`).join('')}<th>Coverage</th></tr></thead><tbody>${rows.filter(b=>b.cap===cap).map(b=>`<tr><td>${esc(b.label)}</td><td>${esc(b.agency)} / ${esc(b.group)}</td>${metrics(b)}${benchmarks.map(k=>`<td>${fmt(b.excess[k.label]===null?null:b.excess[k.label]*scale,2)}</td>`).join('')}<td>${esc(b.status)}</td></tr>`).join('')}</tbody></table>`).join('');
 const valid=rows.filter(b=>b.change!==null),traces=[],layout={height:560,margin:{l:65,r:30,t:40,b:80},hovermode:'closest',yaxis:{title:`Price change (${unit})`},xaxis:{title:'Date'},legend:{orientation:'h'},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{color:'#a6b5c0',width:1}}]};
 if($('perf-view').value==='bar'){
  const entries=[...benchmarks,...valid];layout.height=Math.max(500,entries.length*24+100);layout.margin.l=245;layout.xaxis={title:`${window.end} − ${window.start} (${unit})`};layout.yaxis={autorange:'reversed',automargin:true};layout.shapes=[];
  traces.push({type:'bar',orientation:'h',y:entries.map(b=>b.label),x:entries.map(b=>b.change===null?null:b.change*scale),text:entries.map(b=>fmt(b.change===null?null:b.change*scale,2)),textposition:'auto',marker:{color:entries.map(b=>b.cap?({6:'#478db5',6.5:'#47a596',7:'#a786bc'})[b.cap]:colors[benchmarks.indexOf(b)])},hovertemplate:`%{y}<br>${window.start} → ${window.end}: %{x:.2f} ${unit}<extra></extra>`});
 }else{
  [...valid,...benchmarks].forEach((b,i)=>{const benchmark=!b.cap;traces.push({type:'scatter',mode:'lines',x:window.dates,y:b.prices.map(p=>p===null||b.start_price===null?null:(p-b.start_price)*scale),name:b.label,connectgaps:false,line:{width:benchmark?3.5:1.5,dash:benchmark?'dash':'solid',color:benchmark?colors[benchmarks.indexOf(b)]:undefined},hovertemplate:`%{x}<br>${esc(b.label)}: %{y:.2f} ${unit}<extra></extra>`});});layout.legend={orientation:'v',x:1.02,y:1};layout.margin.r=295;
 }
 $('perf-status').textContent=`Requested ${$('perf-start').value} to ${$('perf-end').value}; actual common dates ${window.start} to ${window.end}. Showing ${rows.length}/${candidates.length} bonds with both endpoint prices. Excluded: ${candidates.filter(b=>b.change===null).map(b=>b.label).join(', ')||'none'}. Embedded history ${comparison.dates[0]} to ${comparison.dates.at(-1)}.`;
 Plotly.react('perf-chart',traces,layout,{responsive:true,displaylogo:false});
}
const selected=()=>bonds.find(b=>b.ticker===$('ticker').value), filtered=()=>bonds.filter(b=>($('group').value==='all'||b.group===$('group').value)&&($('cap').value==='all'||b.cap===Number($('cap').value)));
function selectTickers(){let prior=$('ticker').value;$('ticker').innerHTML=bonds.map(b=>`<option value="${esc(b.ticker)}">${esc(b.label)} · cap ${fmt(b.cap,1)}%${b.group==='portfolio'?' · PF':''}</option>`).join('');if(bonds.some(b=>b.ticker===prior))$('ticker').value=prior;render();}
function current(b,model=$('model').value){const m=b.models[model];if(!m)return null;const r=last(m.resid),s=last(m.sigma);return {m,r,z:s>1e-12?r/s:0,px:last(b.series.px),fv:last(m.fit)};}
function table(){ $('watchlist').innerHTML=filtered().map(b=>{const c=current(b,$('watch-model').value),v=b.validation,ref=b.reference;return `<tr data-ticker="${esc(b.ticker)}" class="${b.ticker===$('ticker').value?'selected':''}"><td>${esc(b.label)}</td><td style="text-align:left">${b.group==='portfolio'?'Portfolio':'Analysis'}</td><td>${fmt(b.cap,1)}</td><td style="text-align:left" title="${esc(ref.collateral_status||'Unavailable')}${ref.collateral_asof?' · '+esc(ref.collateral_asof):''}">${esc(ref.mtg_collat_typ||'—')}</td><td title="Coupon parsed from Bloomberg MTG_COLLAT_TYP">${ref.underlying_cpn!=null?fmt(ref.underlying_cpn,2)+'%':'—'}</td><td title="${esc(ref.status)}${ref.asof?' · '+esc(ref.asof):''}">${fmt(ref.flt_spread,1)}</td><td>${last(b.dates)||'—'}</td>${c?`<td>${c.m.n}</td><td>${fmt(c.px,3)}</td><td>${fmt(c.fv,3)}</td><td class="${c.r<0?'cheap':'rich'}">${fmt(c.r*32,1)}</td><td>${fmt(c.z)}</td><td>${fmt(c.m.adj_r2)}</td>`:'<td colspan="6">Model unavailable</td>'}<td>${fmt(v.baseline_rmse,1)}</td><td>${fmt(v.enhanced_rmse,1)}</td><td>${fmt(v.improvement,1)}${v.improvement!=null?'%':''}</td></tr>`}).join('');document.querySelectorAll('#watchlist tr').forEach(tr=>tr.onclick=()=>{$('ticker').value=tr.dataset.ticker;render()});}
function selectedScenario(){
  const rate=$('shock-rate').value.trim(),cc=$('shock-cc').value.trim();
  if(rate===''||cc===''||!Number.isFinite(Number(rate))||!Number.isFinite(Number(cc)))return null;
  const signed=v=>(v>0?'+':v<0?'−':'')+fmt(Math.abs(v),2);
  return {key:'custom',name:`CT10 ${signed(Number(rate))}bp · CC basis ${signed(Number(cc))}bp`,shocks:{CT10:Number(rate)/100,cc:Number(cc),'2s10s':0,vol:0}};
}
function scenarioImpact(b,scenario){const c=current(b,$('scenario-model').value);if(!c)return null;const parts={};let total=0;for(const [p,shock] of Object.entries(scenario.shocks)){parts[p]=c.m.b[p]*shock;total+=parts[p];}return {parts,total,price:c.px,fair:c.fv,shockedPrice:c.px+total,shockedFair:c.fv+total};}
function shocks(){
  const s=selectedScenario();
  if(!s){$('scenario-tables').innerHTML='<p class="warning">Enter valid numeric CT10 and CC basis changes.</p>';return;}
  $('scenario-tables').innerHTML=`<h3>${esc(s.name)}</h3>`+[Number($('shock-cap').value)].map(cap=>{
    const rows=bonds.filter(b=>b.cap===cap&&($('shock-group').value==='all'||b.group===$('shock-group').value)).map(b=>{const c=scenarioImpact(b,s);
      const start=`<tr data-shock-bond="${esc(b.ticker)}" data-scenario="${s.key}"><td>${esc(b.label)}</td><td>${esc(b.cc)}</td><td>${last(b.dates)||'—'}</td>`;
      if(!c)return start+'<td colspan="8">Model unavailable — insufficient or collinear history</td></tr>';
      return start+`<td>${fmt(c.price,3)}</td>`+['CT10','cc','2s10s','vol'].map(p=>`<td>${fmt(c.parts[p]*32,1)}</td>`).join('')+`<td><strong>${fmt(c.total*32,1)}</strong></td><td>${fmt(c.shockedPrice,3)}</td><td>${fmt(c.shockedFair,3)}</td></tr>`;
    }).join('');
    return `<h4>Cap ${fmt(cap,1)}%</h4><div class="scroll"><table><thead><tr><th>Bond</th><th>CC basis</th><th>As of</th><th>Observed price</th><th>CT10 Δ (ticks)</th><th>CC Δ (ticks)</th><th>2s10s Δ (ticks)</th><th>Vol Δ (ticks)</th><th>Total Δ (ticks)</th><th>Shocked price</th><th>Shocked model value</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }).join('');
}
function render(){const b=selected(),c=current(b);table();$('cards').innerHTML=c?[['Observed price',fmt(c.px,3)],['Model value',fmt(c.fv,3)],['Residual / z',`${fmt(c.r*32,1)} ticks / ${fmt(c.z)}`],['Read',Math.abs(c.z)<1?'NEUTRAL':c.r<0?'BELOW MODEL':'ABOVE MODEL']].map(([k,v])=>`<div class="card"><span class="muted">${k}</span><strong>${v}</strong></div>`).join(''):'';
$('status').textContent=c?`${b.label} · ${b.cc} · observed ${last(b.dates)} · trained ${c.m.start} to ${c.m.end} (${c.m.n} observations)${c.m.n<60?' · short history':''}${c.m.condition>30?' · correlated factors: unstable coefficients possible':''}`:b.error||'Insufficient history for the selected model.';
const traces=[],layout={template:'plotly_white',height:530,margin:{t:65,l:65,r:35,b:65},hovermode:'x unified',legend:{orientation:'h',y:1.14},title:{text:esc(b.label)},xaxis:{title:'Date'},yaxis:{title:'Price (points)'}};
const line=(y,name,color,dash='solid')=>({x:b.dates,y,name,type:'scatter',mode:'lines',line:{color,width:1.8,dash},connectgaps:false});const view=$('view').value;
if(view==='factors'){layout.height=640;layout.grid={rows:4,columns:1,pattern:'independent',ygap:.18};['CT10','cc','2s10s','vol'].forEach((p,i)=>{const ax=i?String(i+1):'';traces.push({...line(b.series[p],p,['#a6404c','#c78a26','#14896b','#7c53a5'][i]),xaxis:'x'+ax,yaxis:'y'+ax});layout['xaxis'+ax]={matches:i?'x':undefined,tickformat:'%b %Y'};layout['yaxis'+ax]={title:p+(p==='CT10'?' (%)':' (bp)')}})}
else if(view==='validation'){layout.yaxis.title='Held-out error (ticks)';traces.push(line(b.baseline?.map(v=>v===null?null:v*32)||[],'2 factors','#9faebe'));const r=b.models.rolling;if(r)traces.push(line(r.resid.map(v=>v===null?null:v*32),'4 factors','#187f76'));layout.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{color:'#a6b5c0',width:1}}]}
else if(c){if(view==='price'){traces.push({...line(b.price_history?.prices||b.series.px,'Observed','#173c58'),x:b.price_history?.dates||b.dates},line(c.m.fit,'Model value','#1b9f92'))}
else if(view==='residual'){layout.yaxis.title='Actual − model (ticks)';for(let k of [-2,-1,1,2])traces.push(line(c.m.sigma.map((v,i)=>c.m.resid[i]===null?null:v*k*32),`${k>0?'+':''}${k}σ`,'#bbc8d3','dot'));traces.push(line(c.m.resid.map(v=>v===null?null:v*32),'Residual','#173c58'));layout.shapes=[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{color:'#8398a7',width:1}}]}
else{const vals=c.m.fit.filter(v=>v!==null),ys=b.series.px;const low=Math.min(...vals,...ys),high=Math.max(...vals,...ys);traces.push({x:c.m.fit,y:ys,type:'scatter',mode:'markers',name:'Daily observations',customdata:b.dates,hovertemplate:'%{customdata}<br>Model %{x:.3f}<br>Actual %{y:.3f}<extra></extra>',marker:{color:'#187f76',size:6,opacity:.5}},{x:[low,high],y:[low,high],type:'scatter',mode:'lines',name:'Actual = model',line:{color:'#a6404c',dash:'dash'}});layout.hovermode='closest';layout.xaxis.title='Model value (points)';layout.yaxis.title='Observed price (points)'}}
// Endpoint labels use the last finite observation, preserving each trace's axes.
if(view==='scatter'&&c){
  traces.push({x:[c.fv],y:[c.px],type:'scatter',mode:'markers+text',name:'Latest observation',
    text:[`Latest ${last(b.dates)}<br>Actual ${fmt(c.px,3)} · Fitted ${fmt(c.fv,3)}`],
    textposition:'top left',textfont:{color:'#bc4252',size:12},cliponaxis:false,
    customdata:[[last(b.dates),c.r*32]],marker:{color:'#bc4252',size:13,symbol:'diamond',line:{color:'white',width:1.5}},
    hovertemplate:'Latest %{customdata[0]}<br>Actual %{y:.3f}<br>Fitted %{x:.3f}<br>Residual %{customdata[1]:.1f} ticks<extra></extra>'});
  const all=[...c.m.fit.filter(v=>v!==null),...b.series.px],lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*.15,.05);
  layout.xaxis.range=[lo-pad,hi+pad];layout.yaxis.range=[lo-pad,hi+pad];
}else{
  layout.margin.r=view==='factors'?180:215;
  const endpoints=[];layout.annotations=[];
  traces.filter(t=>!t.name.includes('σ')).forEach((t,i)=>{
    let j=t.y.length-1;while(j>=0&&(t.y[j]===null||!Number.isFinite(t.y[j])))j--;
    if(j<0)return;
    const units=view==='residual'||view==='validation'?' ticks':view==='factors'?(t.name==='CT10'?'%':' bp'):'';
    const decimals=view==='price'?3:view==='factors'?2:1;
    layout.annotations.push({x:t.x[j],y:t.y[j],xref:t.xaxis||'x',yref:t.yaxis||'y',
      text:`${esc(t.name)} ${fmt(t.y[j],decimals)}${units}${t.x[j]!==last(b.dates)?'<br>'+t.x[j]:''}`,
      showarrow:false,xanchor:'left',xshift:10,yshift:view==='factors'?0:(i%2===0?11:-11),
      font:{color:t.line.color,size:11},bgcolor:'rgba(255,255,255,.9)'});
    endpoints.push({x:[t.x[j]],y:[t.y[j]],xaxis:t.xaxis||'x',yaxis:t.yaxis||'y',
      type:'scatter',mode:'markers',name:`Latest ${t.name}`,showlegend:false,
      marker:{color:t.line.color,size:7},hovertemplate:`%{x|%Y-%m-%d}<br>${esc(t.name)} %{y:.${decimals}f}${units}<extra></extra>`});
  });traces.push(...endpoints);
}
if(!traces.length)layout.annotations=[{text:'Model unavailable for this bond / selection',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false});$('diagnostics').textContent=c?`Training adj. R² ${fmt(c.m.adj_r2,3)} · scaled condition number ${fmt(c.m.condition,1)} · validation observations ${b.validation.n||0} · four-factor RMSE improvement ${fmt(b.validation.improvement,1)}%`:'';
}
const dates=bonds.flatMap(b=>b.dates).sort();const caps=[...new Set(bonds.map(b=>b.cap))].sort((a,b)=>a-b).map(c=>c.toFixed(1)+'%').join(' / ');const nPf=bonds.filter(b=>b.group==='portfolio').length;$('asof').textContent=`${bonds.length} bonds (${nPf} portfolio + ${bonds.length-nPf} analysis) · caps ${caps} · available complete observations through ${last(dates)} · self-contained report`;
['group','cap','watch-model'].forEach(id=>$(id).addEventListener('change',table));['ticker','model','view'].forEach(id=>$(id).addEventListener('change',render));['scenario-model','shock-cap','shock-group'].forEach(id=>$(id).addEventListener('change',shocks));['shock-rate','shock-cc'].forEach(id=>$(id).addEventListener('input',shocks));selectTickers();shocks();
['perf-cap','perf-view','perf-start','perf-end'].forEach(id=>$(id).addEventListener('change',performance));performance();
forwardIndicator();
</script></body></html>'''
