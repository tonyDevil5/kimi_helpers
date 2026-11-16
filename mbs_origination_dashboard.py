"""Standalone signal dashboard following the existing coupon-report UI conventions."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

LABELS={'coupon':'Coupon','asof':'As of','orig_z':'Origination z','ready':'Ready','action':'Next-close action',
'current_simulated_notional':'Simulated position ($)','next_close_target_notional':'Next-close target ($)',
'rebalance_notional':'Rebalance ($)','blank_source_values':'Source blanks','raw_signal':'Raw signal','signal':'Latest origination signal',
'net_pnl_usd':'Net P&L ($)','gross_pnl_usd':'Gross P&L ($)','cost_usd':'Costs ($)','sharpe':'Sharpe',
'max_drawdown_usd':'Max drawdown ($)','hac_p':'HAC p','fdr_p':'FDR p','obs':'Observations',
'active_pct':'Active (%)','valid_orig_days':'Valid signal days','execution_lag':'Execution lag','cost_ticks':'Cost (ticks)'}


def table(frame, position_scale=None):
    parts=['<div class="table-scroll"><table><thead><tr>']
    for col in frame:parts.append('<th tabindex="0" role="button" aria-sort="none" title="Sort column">'+html.escape(LABELS.get(col,col))+' <span class="sort-arrow">↕</span></th>')
    parts.append('</tr></thead><tbody>')
    for _,row in frame.iterrows():
        parts.append('<tr>')
        for col,value in row.items():
            missing=pd.isna(value)
            if missing:display='—';key=''
            elif isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)):
                key=str(value);display=f'{value:,.0f}' if ('notional' in col or 'usd' in col or col in ['obs','blank_source_values','valid_orig_days','execution_lag']) else f'{value:,.3f}'
            else:key=str(value);display=str(value)
            if col=='signal' and display in ('LONG','SHORT','FLAT'):
                background={'LONG':'#15803d','SHORT':'#b91c1c','FLAT':'#475569'}[display]
                badge=(f'<span style="display:inline-block;min-width:72px;text-align:center;'
                       f'padding:7px 12px;border-radius:6px;background:{background};color:white;'
                       'font-size:13px;font-weight:800;letter-spacing:.6px">'+display+'</span>')
                parts.append('<td data-sort="'+display+'">'+badge+'</td>')
                continue
            style=''
            if position_scale and col in ('current_simulated_notional','next_close_target_notional') and not missing:
                strength=min(abs(float(value))/position_scale,1.)
                base=(220,252,231) if value>0 else (254,226,226) if value<0 else (241,245,249)
                color=tuple(round(255+(c-255)*strength) for c in base) if value else base
                ink='#166534' if value>0 else '#991b1b' if value<0 else '#64748b'
                style=f' style="background:rgb{color};color:{ink};font-weight:700" title="'+('Long' if value>0 else 'Short' if value<0 else 'Flat')+'"'
            parts.append('<td'+style+' data-sort="'+html.escape(key,quote=True)+'">'+html.escape(display)+'</td>')
        parts.append('</tr>')
    return ''.join(parts)+'</tbody></table></div>'


def render(root):
    root=Path(root)
    signals=pd.read_csv(root/'all_coupon_signals.csv');summary=pd.read_csv(root/'all_coupon_performance.csv')
    daily=pd.read_csv(root/'all_coupon_daily.csv');costs=pd.read_csv(root/'all_coupon_cost_delay.csv')
    strict=pd.read_csv(root/'strict_missing_all_coupon_performance.csv')
    payload={}
    for coupon,g in daily.groupby('coupon',sort=False):
        payload[coupon]=dict(dates=g.date.tolist(),z=[None if pd.isna(v) else float(v) for v in g.orig_z],
            position=g.position.tolist(),pnl=g.net_pnl_usd.tolist())
    config=json.loads((root/'signal_config.json').read_text()) if (root/'signal_config.json').exists() else {'notional':10000000,'cost_ticks':.5}
    encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c')
    dates=daily.date;start='2025-06-12';end=dates.max()
    signals['signal']=['Unavailable' if not ready else 'SHORT' if raw<0 else 'LONG' if raw>0 else 'FLAT' for ready,raw in zip(signals.ready,signals.raw_signal)]
    status_cols=['coupon','signal','asof','ready','orig_z','action','current_simulated_notional','next_close_target_notional','rebalance_notional']
    sections=f'''<header><h1>Coupon origination — trading signal research</h1><p>JPM hedged coupon performance · ${config['notional']/1e6:g}mm per coupon · Latest source {end}</p><nav><a href="#signals">Positions & signals</a><a href="#performance">Performance</a><a href="#history">Coupon history</a><a href="#sensitivity">Sensitivity</a><a href="#method">Method & data</a></nav></header>
    <section><p>Check each coupon's latest origination signal, simulated position and next-close rebalance below. Each coupon is evaluated independently at ${config['notional']/1e6:g}mm base notional. Evaluation period: {start} → {end}.</p></section>
    <section id="signals"><h2>Current positions and latest signals</h2><p>Latest source {end}. LONG = low origination (z ≤ −1); SHORT = high origination (z ≥ +1); FLAT = no new entry. The next-close target also reflects the existing five-day sleeves. Positions are simulated.</p>{table(signals[status_cols])}</section>
    <section id="performance"><h2>Performance by coupon</h2><p>Per ${config['notional']/1e6:g}mm per coupon · Click column headers to sort. Dollar amounts are P&L, not returns on capital.</p>{table(summary)}</section>
    <section id="history"><h2>Coupon history</h2><div class="toolbar"><label>Coupon <select id="coupon"></select></label><label>From <input type="date" id="from" value="{start}"></label><label>To <input type="date" id="to" value="{end}"></label><button id="all-history">All history</button><button id="evaluation">Evaluation period</button><button id="export">Export selected history CSV</button></div><p id="range-note"></p><div id="input-chart"></div><div id="pnl-chart"></div><div id="position-chart"></div></section>
    <section id="sensitivity"><h2>Costs and execution delay</h2><label>Coupon filter <select id="cost-filter"><option value="">All coupons</option></select></label>{table(costs)}</section>
    <section id="method"><details><summary>Rules, sizing and data assumptions</summary><p>20-observation origination sum standardized against the preceding 250 observations. Short above +1; long below −1; otherwise flat. Five equal sleeves hold each signal for five observations. Signal at close t executes at t+1 close and first earns t+2 performance. Full 250-observation warm-up. No per-coupon parameter optimization.</p><p>Shared SimpleBacktester P&L through PerformancePositionBacktester; 1 tick = notional / 100 / 32 = ${config['notional']/100/32:,.2f}. Costs: {config['cost_ticks']} tick per full-notional one-way turnover. JPM hedge is implicit in Perf; hedge contract sizing, funding, carry and roll are not reconstructed. No forced end-of-sample liquidation.</p><p>Tracker blanks are treated as zero, matching the original monitor. This unverified assumption materially affects results. Strict-missing-data results appear below. Historical release times and revisions are unknown. Evaluation is retrospective, not a new untouched holdout. HAC uses 20 lags; FDR is computed over the valid p-values of this fixed-rule coupon family.</p></details><details><summary>Source coverage and strict missing-data sensitivity</summary>{table(signals[['coupon','ready','blank_source_values']])}{table(strict)}</details></section>'''
    css='''body{font:14px Arial;background:#eef2f6;color:#172e40;margin:20px}main{max-width:1500px;margin:auto}header,section{background:white;padding:22px;margin:16px 0;border-radius:8px;border:1px solid #d8e2ea}h1{margin:0 0 12px}h2{font-size:20px}p{line-height:1.65;color:#485b69}nav{display:flex;gap:22px;flex-wrap:wrap}nav a{color:#17658d}.table-scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{background:#173e56;color:white;padding:10px;cursor:pointer;position:sticky;top:0;text-align:right}th:first-child,td:first-child{text-align:left}td{padding:9px;border:1px solid #ddd;text-align:right}tr:nth-child(even){background:#f3f6f9}tr:hover{background:#e7f2fa}select,input,button{padding:8px;border:1px solid #b3c6d2;border-radius:5px;background:white}button,summary{cursor:pointer}summary{font-weight:bold}details{margin-top:18px}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.stats{display:flex;gap:50px;flex-wrap:wrap}.stats small{display:block;color:#526878}.stats strong{display:block;font-size:24px;margin:10px 0}.sort-arrow{font-size:11px}@media(max-width:650px){body{margin:8px}section,header{padding:14px}}'''
    js='''const D=JSON.parse(document.getElementById('payload').textContent),get=id=>document.getElementById(id),s=get('coupon');Object.keys(D).forEach(k=>{for(const id of ['coupon','cost-filter']){let o=document.createElement('option');o.value=k;o.textContent=k;get(id).append(o)}});
    function render(){const d=D[s.value],rows=d.dates.map((date,i)=>({date,z:d.z[i],position:d.position[i],pnl:d.pnl[i]})).filter(r=>r.date>=get('from').value&&r.date<=get('to').value);let running=0;rows.forEach(r=>{running+=r.pnl;r.cumulative=running});get('range-note').textContent=rows.length+' observations. P&L starts at zero at the selected range start; the default evaluation period matches the performance table.';const x=rows.map(r=>r.date),config={responsive:true,scrollZoom:true,displaylogo:false};
    function plot(id,traces,title,axis,height){Plotly.react(id,traces,{title:{text:s.value+' — '+title},height,hovermode:'x unified',dragmode:'zoom',margin:{l:85,r:25,t:50,b:50},xaxis:{type:'date',rangeslider:{visible:id==='pnl-chart'},rangeselector:{buttons:[{count:3,label:'3m',step:'month',stepmode:'backward'},{count:1,label:'1y',step:'year',stepmode:'backward'},{step:'all',label:'All'}]}},yaxis:{title:{text:axis}},shapes:id==='input-chart'?[-1,1].map(y=>({type:'line',xref:'paper',x0:0,x1:1,y0:y,y1:y,line:{dash:'dash',color:'#999'}})):[],annotations:rows.length?[]:[{text:'No observations in selected range',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]},config)}
    plot('input-chart',[{x,y:rows.map(r=>r.z),name:'Origination z',mode:'lines',line:{color:'#17658d'},hovertemplate:'%{x|%Y-%m-%d}<br>z: %{y:.3f}<extra></extra>'}],'origination signal (entry ±1)','z-score',360);
    plot('pnl-chart',[{x,y:rows.map(r=>r.cumulative),name:'Net cumulative P&L',mode:'lines',line:{color:'#16856d'},hovertemplate:'%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>'}],'net cumulative P&L','USD',420);
    plot('position-chart',[{x,y:rows.map(r=>r.position),name:'Executed position',mode:'lines',line:{shape:'hv',color:'#8063a6'}}],'position after close (fraction of base notional)','−1 short / 0 flat / +1 long',300);}
    s.onchange=render;get('from').onchange=render;get('to').onchange=render;get('all-history').onclick=()=>{get('from').value=D[s.value].dates[0];get('to').value=D[s.value].dates.at(-1);render()};get('evaluation').onclick=()=>{get('from').value='2025-06-12';get('to').value=D[s.value].dates.at(-1);render()};render();
    get('cost-filter').onchange=()=>{document.querySelectorAll('#sensitivity tbody tr').forEach(r=>r.hidden=!!get('cost-filter').value&&r.cells[0].textContent!==get('cost-filter').value)};
    document.querySelectorAll('table').forEach(t=>t.querySelectorAll('th').forEach((h,i)=>{function sort(){const asc=h.getAttribute('aria-sort')!=='ascending';t.querySelectorAll('th').forEach(th=>{th.setAttribute('aria-sort','none');th.querySelector('.sort-arrow').textContent='↕'});h.setAttribute('aria-sort',asc?'ascending':'descending');h.querySelector('.sort-arrow').textContent=asc?'▲':'▼';[...t.tBodies[0].rows].sort((a,b)=>{let x=a.cells[i].dataset.sort,y=b.cells[i].dataset.sort;if(x===''||y==='')return x===y?0:x===''?1:-1;let v=Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y,undefined,{numeric:true});return asc?v:-v}).forEach(r=>t.tBodies[0].append(r))}h.onclick=sort;h.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();sort()}}}));
    get('export').onclick=()=>{const d=D[s.value],keys=['date','origination_z','executed_fraction','net_pnl_usd'],lines=[keys.join(',')];d.dates.forEach((date,i)=>{if(date>=get('from').value&&date<=get('to').value)lines.push([date,d.z[i]??'',d.position[i],d.pnl[i]].join(','))});let url=URL.createObjectURL(new Blob([lines.join('\\r\\n')],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=s.value.replace(' ','_')+'_history.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};'''
    output=root/'All_Coupon_Origination_Signals.html'
    output.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coupon origination signals</title><style>'+css+'</style><script>'+get_plotlyjs()+'</script></head><body><main>'+sections+'<script type="application/json" id="payload">'+encoded+'</script><script>'+js+'</script></main></body></html>',encoding='utf-8')
    return output

if __name__=='__main__':print(render(Path(__file__).parent/'mbs_origination_signal_output'))
