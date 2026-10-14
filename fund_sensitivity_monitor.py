"""Weekly USD total-return sensitivities for the fund-flow workbook universe."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import html

import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fund_flows_monitor import Bloomberg, read_universe, WORKBOOK

OUT=Path(__file__).parent/'fund_sensitivity_output'
MBS_WORKBOOK=Path(r'C:\Users\tonyy\Nutstore\1\python\notebook\data\mbs\mbs_rv.xlsx')
TRI='TOT_RETURN_INDEX_GROSS_DVDS'
FACTORS=['CT10','FN basis','IG OAS','2s10s','1y10y vol']


def download(universe,start,end):
    refs, returns=[],[]
    with Bloomberg() as b:
        for i in range(0,len(universe),10):
            tickers=universe.ticker.iloc[i:i+10].tolist()
            refs.append(b.request(tickers,['NAME','CRNCY','NAV_CRNCY','ID_BB_GLOBAL','FUND_CLASS_ASSETS']))
            returns.append(b.request(tickers,[TRI],start,end))
            print(f'Fund total returns: {min(i+10,len(universe))}/{len(universe)}',flush=True)
        ref=pd.concat(refs,ignore_index=True)
        raw=pd.concat(returns,ignore_index=True)
        fx=[]
        for ccy in sorted(set(ref.CRNCY.dropna())-{'USD'}):
            rates=b.request([f'{ccy}USD Curncy'],['PX_LAST'],start,end)
            if len(rates):fx.append(rates.assign(currency=ccy))
        fn=b.request(['.FNCC105 G Index'],['PX_LAST'],start,end)
        ig=b.request(['LUACTRUU Index'],['OAS'],start,end)
        if 'OAS' in ig and ig.OAS.notna().any():
            ig_series=ig.set_index('date').OAS
            ig_source='LUACTRUU Index OAS, bps'
        else:
            ig=b.request(['LUACOAS Index'],['PX_LAST'],start,end)
            if 'PX_LAST' not in ig or ig.PX_LAST.dropna().empty:raise ValueError('IG OAS unavailable')
            ig_series=ig.set_index('date').PX_LAST*100
            ig_source='LUACOAS Index PX_LAST, percent ×100 to bps (OAS fallback)'
        if 'PX_LAST' not in fn or fn.PX_LAST.dropna().empty:raise ValueError('FN basis unavailable')
        factors=pd.concat({'FN basis':fn.set_index('date').PX_LAST*100,'IG OAS':ig_series},axis=1)
        factors.index=pd.to_datetime(factors.index)
        errors=b.errors
    return ref,raw,pd.concat(fx,ignore_index=True) if fx else pd.DataFrame(columns=['date','currency','PX_LAST']),factors,ig_source,errors


def workbook_factors(path):
    d=pd.read_excel(path,sheet_name='Data',index_col=0)
    v=pd.read_excel(path,sheet_name='Data1',index_col=0)
    d.index=pd.to_datetime(d.index);v.index=pd.to_datetime(v.index)
    if d.index.has_duplicates or v.index.has_duplicates:raise ValueError('Duplicate workbook factor dates')
    return pd.concat({'CT10':d.CT10*100,'2s10s':(d.CT10-d.CT2)*100,
                      '1y10y vol':v['1y10y_atm']},axis=1).sort_index()


def weekly_endpoints(series, end):
    """Use the actual final observation in a Friday-ending week; reject stale endpoints."""
    s=series.dropna().sort_index()
    if s.index.has_duplicates:raise ValueError('Duplicate history dates')
    if s.empty:return pd.Series(dtype=float)
    dates=pd.Series(s.index,index=s.index)
    levels=s.resample('W-FRI').last()
    observed=dates.resample('W-FRI').last()
    age=(observed.index.to_series()-observed).dt.days
    return levels.where(age<=3).loc[:end]


def ytd_total_return(series, end):
    """Exact prior year-end to current completed week; no stale start/end proxy."""
    end=pd.Timestamp(end).normalize()
    baseline=pd.Timestamp(year=end.year-1,month=12,day=31)
    s=series.dropna().sort_index()
    start=s.loc[:baseline];finish=s.loc[:end]
    result=dict(ytd_return_pct=np.nan,ytd_start_observation='',ytd_end_observation='')
    if start.empty or finish.empty:return result
    if (baseline-start.index[-1]).days>3 or (end-finish.index[-1]).days>3:return result
    if start.iloc[-1]<=0:return result
    return dict(ytd_return_pct=float((finish.iloc[-1]/start.iloc[-1]-1)*100),
                ytd_start_observation=str(start.index[-1].date()),
                ytd_end_observation=str(finish.index[-1].date()))


def make_weekly(universe,ref,raw,fx,factors,end):
    meta=universe.merge(ref.rename(columns={'NAME':'bloomberg_name'}),on='ticker',how='left',validate='one_to_one')
    ids=meta.ID_BB_GLOBAL.dropna()
    if ids.duplicated().any():raise ValueError('Duplicate share-class FIGIs')
    raw=raw.copy();raw['date']=pd.to_datetime(raw.date)
    fx=fx.copy();fx['date']=pd.to_datetime(fx.date)
    levels={}; coverage=[]
    for r in meta.itertuples():
        source=raw[raw.ticker==r.ticker].set_index('date')
        tri=source[TRI].dropna() if TRI in source else pd.Series(dtype=float)
        if r.CRNCY=='USD':usd=tri
        elif pd.notna(r.CRNCY):
            rates=fx[fx.currency==r.CRNCY].set_index('date').PX_LAST.sort_index()
            aligned=rates.reindex(tri.index,method='ffill',tolerance=pd.Timedelta(days=3)) if len(rates) else pd.Series(np.nan,index=tri.index)
            usd=tri*aligned.where(aligned>0)
        else:usd=tri*np.nan
        usd=usd.where(usd>0)
        levels[r.ticker]=weekly_endpoints(usd,end)
        coverage.append(dict(ticker=r.ticker,return_currency=r.CRNCY,raw_observations=len(tri),
                             first_return_observation=tri.index.min() if len(tri) else pd.NaT,
                             last_return_observation=tri.index.max() if len(tri) else pd.NaT,
                             **ytd_total_return(usd,end)))
    fund_levels=pd.DataFrame(levels)
    # No pct_change default fill: a missing endpoint invalidates both adjacent weekly returns.
    returns=fund_levels.pct_change(fill_method=None)*10000
    factor_levels=pd.DataFrame({c:weekly_endpoints(factors[c],end) for c in FACTORS})
    changes=factor_levels.diff()
    changes.index.name='week_ending';returns.index.name='week_ending'
    return meta.merge(pd.DataFrame(coverage),on='ticker'),returns,changes


def fit_model(y,x,min_obs=40):
    d=pd.concat([y.rename('return_bps'),x],axis=1).dropna()
    if len(d)<min_obs:return None,d
    design=sm.add_constant(d[FACTORS],has_constant='add')
    if np.linalg.matrix_rank(design.to_numpy())<len(FACTORS)+1:return None,d
    model=sm.OLS(d.return_bps,design).fit(cov_type='HAC',cov_kwds={'maxlags':4})
    return model,d


def analyze(meta,returns,changes):
    summary=[]; coefficients=[];rolling={};fitted={}
    for r in meta.itertuples():
        model,d=fit_model(returns[r.ticker],changes)
        row=dict(ticker=r.ticker,group=r.group,name=r.name,ytd_return_pct=r.ytd_return_pct,observations=len(d),
                 start=str(d.index.min().date()) if len(d) else '',end=str(d.index.max().date()) if len(d) else '',
                 status='Available' if model is not None else 'Insufficient observations or singular factors')
        if model is not None:
            row.update(r2=model.rsquared,adjusted_r2=model.rsquared_adj,
                       weekly_vol_bps=d.return_bps.std(),annualized_return_vol_pct=d.return_bps.std()*np.sqrt(52)/100)
            for factor in FACTORS:
                ci=model.conf_int().loc[factor]
                row['beta_'+factor]=model.params[factor]
                row['corr_'+factor]=d.return_bps.corr(d[factor])
                coefficients.append(dict(ticker=r.ticker,factor=factor,beta_bps_per_bp=model.params[factor],
                                         hac_standard_error=model.bse[factor],hac_pvalue=model.pvalues[factor],
                                         ci95_low=ci.iloc[0],ci95_high=ci.iloc[1]))
            fit=pd.DataFrame({'actual_return_bps':d.return_bps,'fitted_return_bps':model.fittedvalues,'residual_bps':model.resid})
            fitted[r.ticker]=fit
        summary.append(row)
        records=[]
        grid=pd.concat([returns[r.ticker].rename('return_bps'),changes],axis=1)
        for t in range(51,len(grid)):
            m,train=fit_model(grid.return_bps.iloc[t-51:t+1],grid[FACTORS].iloc[t-51:t+1])
            if m is None:continue
            records.append(dict(week_ending=grid.index[t],observations=len(train),r2=m.rsquared,
                                **{'beta_'+c:m.params[c] for c in FACTORS}))
        rolling[r.ticker]=pd.DataFrame(records)
    return pd.DataFrame(summary),pd.DataFrame(coefficients),rolling,fitted


def group_returns(meta,returns):
    """Fixed-membership equal-weight baskets, available only with all constituents present."""
    baskets={}
    for group in ['All','Active','Passive']:
        names=meta.ticker if group=='All' else meta[meta.group==group].ticker
        values=returns[names.tolist()]
        baskets[group+' equal weight']=values.mean(axis=1).where(values.notna().all(axis=1))
    return pd.DataFrame(baskets)


def build_report(meta,summary,coefficients,rolling,fitted,returns,changes,baskets,group_summary,manifest):
    labels=dict(ticker='Bloomberg ticker',group='Group',name='Fund',observations='Weeks',start='Start',end='End',
                r2='R²',basket='Basket',constituents='Included funds',listed_funds='Listed funds',status='Status',
                ytd_return_pct='YTD USD return (%)',ytd_funds='Funds with YTD data')
    labels.update({'beta_'+f:f+' β (bp/bp)' for f in FACTORS})
    labels.update({'corr_'+f:f+' corr' for f in FACTORS})
    def table(d):return d.rename(columns=labels).to_html(index=False,border=0,na_rep='—',float_format=lambda v:f'{v:,.3f}')
    valid=summary[summary.status=='Available']
    fig=go.Figure()
    for group in ['Active','Passive']:
        d=valid[valid.group==group]
        fig.add_trace(go.Heatmap(z=d[['beta_'+f for f in FACTORS]].to_numpy(),x=FACTORS,
                                 y=d.ticker,zmid=0,colorscale='RdBu',reversescale=True,visible=group=='Active',
                                 hovertemplate='%{y}<br>%{x}<br>Return bps per +1bp: %{z:.3f}<extra></extra>'))
    fig.update_layout(template='plotly_white',height=860,margin=dict(l=200),title='Regression sensitivities — return bps per +1bp factor change',
                      updatemenus=[dict(buttons=[dict(label=g,method='update',args=[{'visible':[g=='Active',g=='Passive']}]) for g in ['Active','Passive']])])
    heat=fig.to_html(full_html=False,include_plotlyjs=True)
    options=''.join(f'<option value="{html.escape(r.ticker)}">{html.escape(r.group+": "+r.ticker+" — "+r.name)}</option>' for r in valid.itertuples())
    payload={}
    for r in valid.itertuples():
        fit=fitted[r.ticker];roll=rolling[r.ticker]
        payload[r.ticker]=dict(dates=[str(t.date()) for t in fit.index],actual=fit.actual_return_bps.tolist(),
                              fitted=fit.fitted_return_bps.tolist(),factors={f:changes.loc[fit.index,f].tolist() for f in FACTORS},
                              roll_dates=[str(t.date()) for t in roll.week_ending] if len(roll) else [],
                              rolling={f:roll['beta_'+f].tolist() if len(roll) else [] for f in FACTORS})
    safe=json.dumps(payload,allow_nan=False).replace('</','<\\/')
    views=['group','ticker','name','ytd_return_pct','observations','start','end','r2',*['beta_'+f for f in FACTORS],*['corr_'+f for f in FACTORS]]
    present=[c for c in views if c in summary]
    sort_options=''.join(f'<option value="{i}"'+(' selected' if c=='ytd_return_pct' else '')+f'>{html.escape(labels.get(c,c))}</option>' for i,c in enumerate(present))
    report=f'''<!doctype html><html><head><meta charset="utf-8"><title>Fund return sensitivities</title><style>
body{{font:14px Arial;background:#eef2f6;color:#17202a;margin:22px}}main{{max-width:1500px;margin:auto}}section{{background:white;padding:22px;margin:16px 0;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th{{background:#173e56;color:white;padding:9px;cursor:pointer}}th::after{{content:' ↕';color:#bfd0df}}th[aria-sort="ascending"]::after{{content:' ▲';color:white}}th[aria-sort="descending"]::after{{content:' ▼';color:white}}td{{padding:8px;border:1px solid #ddd;text-align:right}}tr:nth-child(even){{background:#f3f6f9}}select,input,button{{padding:9px;margin:10px 8px 10px 0}}button{{cursor:pointer}}summary{{cursor:pointer;font-weight:bold}}#fund-table{{max-height:650px;overflow:auto;border:1px solid #ddd}}#fund-table th{{position:sticky;top:0;z-index:1}}.table-controls{{display:flex;align-items:center;flex-wrap:wrap;gap:4px}}#row-count{{color:#52606d}}</style></head><body><main>
<section><h1>Fixed-income fund return sensitivities</h1><p><a href="../fund_flows_output/fund_flows_monitor.html">Weekly fund flows</a></p><p>Weekly USD total returns versus changes in CT10 yield, FN current-coupon basis, IG OAS, 2s10s and annualized 1y10y normal volatility. {len(valid)}/{len(meta)} funds have estimable models. Bloomberg refreshed {manifest['fetched_at']}.</p><p><b>Interpretation:</b> beta −5 for CT10 means a +10bp yield move is associated with approximately −50bp (−0.50%) fund return, holding the other factors constant. Negative basis/OAS beta means widening hurts returns. Simple correlations and multivariate betas can differ because the factors move together.</p>
<p>All five inputs are changes in basis points. Betas are fund return bps per +1bp factor change. Fits use available matched weekly observations in the last two years, with at least 40 observations. Rolling models use a 52-calendar-week window with at least 40 matched observations. These are historical associations, not causal forecasts or a trading backtest.</p></section>
<section><h2>Active and passive group baskets</h2>{table(group_summary)}<p>These fixed-membership, equal-weight USD baskets include only funds with estimable full-period models ({len(valid)} of {len(meta)}). YTD is the average of the included funds' year-to-date USD total returns, equivalent to equal initial allocations with no rebalancing; it is unavailable unless all included funds have YTD data. Regression baskets average returns with equal weights each week. YTD runs from the final available close of the previous calendar year through the latest completed week. There is no weighting by the workbook's whole-fund AUM.</p></section>
<section><h2>Fund sensitivities and correlations</h2><div class="table-controls"><input id="search" placeholder="Search fund or ticker" aria-label="Search funds"><select id="fund-group" aria-label="Fund group"><option>All</option><option>Active</option><option>Passive</option></select><label for="sort-column">Sort by</label><select id="sort-column">{sort_options}</select><select id="sort-order" aria-label="Sort direction"><option value="desc">High to low / Z to A</option><option value="asc">Low to high / A to Z</option></select><button id="reset-table">Reset</button><span id="row-count"></span></div><p>Click any column heading to sort; click again to reverse. The arrow shows the current order. Missing values stay at the bottom.</p><div id="fund-table">{table(summary[present])}</div>{heat}</section>
<section><h2>Individual fund detail</h2><select id="fund">{options}</select><select id="factor">{''.join('<option>'+f+'</option>' for f in FACTORS)}</select>
<h3>Actual versus fitted weekly return</h3><p>Compare the fund's realized USD total return with the return explained by all five factors together. Each point is one non-overlapping week; returns and factor changes cover the same interval. The gap is the residual: return the model does not explain. A closer fit means the factors explain more of the historical variation. This is an in-sample fit, not a forecast. The factor selector does not change this chart.</p><div id="fit-chart"></div>
<h3>Return versus the selected factor</h3><p>Each dot pairs one week's fund return with that same week's change in the selected factor. An upward pattern indicates positive association; a downward pattern indicates negative association. This shows the simple relationship without controlling for the other factors, so it can differ from the regression beta. Both axes are in basis points. Hover over a dot for its date and values.</p><div id="scatter-chart"></div>
<h3>Rolling sensitivity to the selected factor</h3><p>Each point is the selected factor's beta from a regression using the latest 52 calendar weeks, including that week, with at least 40 matched observations. It controls for the other four factors. A beta of −5 means a +1bp factor change is associated with −5bp of fund return, holding the others constant. Changes in the line suggest the fund's estimated exposure has changed; correlated factors and limited data can also make beta unstable. Successive windows overlap by 51 weeks.</p><div id="rolling-chart"></div></section>
<section><details><summary>Uncertainty and data coverage</summary><h3>Regression coefficient uncertainty</h3>{table(coefficients)}<p>95% intervals and p-values use Newey-West/HAC standard errors with four matched-observation lags (<a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html">method documentation</a>). Several funds and factors are tested; p-values are not adjusted for multiple testing. Correlated factors can produce unstable coefficients.</p><h3>Coverage</h3>{table(meta[['group','ticker','return_currency','raw_observations','first_return_observation','last_return_observation']])}</details></section>
<section><details><summary>Sources and calculation</summary><p>Fund universe: Active/Passive sheets in funds_info.xlsx. Bloomberg {TRI}, in security CRNCY; reinvested distributions are included as defined by Bloomberg. Non-USD indices are converted with currencyUSD Curncy PX_LAST using past FX quotes no more than three days old.</p><p>CT10 and CT2: mbs_rv.xlsx Data sheet, percent ×100 to bps. 2s10s = CT10 − CT2. 1y10y: Data1 1y10y_atm, annualized normal volatility in bps. FN basis: .FNCC105 G Index PX_LAST, percent ×100. IG: {html.escape(manifest['ig_source'])}.</p><p>Friday-ending levels use the final actual observation in each week, at most three calendar days old. No price/return filling across missing weeks. Weekly return = (USD total-return index / prior week's index − 1) ×10,000. Factor input = weekly level change. Model: return = intercept + sum(beta × factor change) + residual. Full-sample fitted values are in-sample; rolling fits include the displayed week's return. Fees, hedges and currency exposures are embedded in the historical fund return series; no extra adjustment is inferred.</p></details></section>
<script type="application/json" id="payload">{safe}</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);const factors={json.dumps(FACTORS)};
function render(){{const fund=document.getElementById('fund').value,factor=document.getElementById('factor').value,d=data[fund];if(!d)return;
Plotly.react('fit-chart',[{{x:d.dates,y:d.actual,name:'Actual weekly return'}},{{x:d.dates,y:d.fitted,name:'Fitted weekly return'}}],{{template:'plotly_white',title:fund+' — actual vs fitted weekly return',yaxis:{{title:'Return bps'}},height:430,hovermode:'x unified'}});
Plotly.react('scatter-chart',[{{x:d.factors[factor],y:d.actual,text:d.dates,mode:'markers',name:fund,hovertemplate:'%{{text}}<br>Factor Δ: %{{x:.2f}}bp<br>Return: %{{y:.2f}}bp<extra></extra>'}}],{{template:'plotly_white',title:'Simple relationship: '+factor,xaxis:{{title:'Factor change (bps)'}},yaxis:{{title:'USD return (bps)'}},height:420}});
Plotly.react('rolling-chart',[{{x:d.roll_dates,y:d.rolling[factor],name:factor}}],{{template:'plotly_white',title:'Rolling 52-week multivariate beta — '+factor,yaxis:{{title:'Return bps per +1bp'}},height:430}});}}
document.getElementById('fund').onchange=render;document.getElementById('factor').onchange=render;render();
const fundTable=document.getElementById('fund-table').querySelector('table');const originalRows=[...fundTable.tBodies[0].rows];
function sortTable(t,col,asc){{const heads=[...t.querySelectorAll('th')];heads.forEach(h=>h.setAttribute('aria-sort','none'));heads[col].setAttribute('aria-sort',asc?'ascending':'descending');
[...t.tBodies[0].rows].sort((a,b)=>{{let x=a.cells[col].textContent.trim().replace(/,/g,''),y=b.cells[col].textContent.trim().replace(/,/g,'');const empty=v=>v===''||v==='—'||v==='NaN';if(empty(x)||empty(y))return empty(x)===empty(y)?0:empty(x)?1:-1;let n=Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y);return asc?n:-n}}).forEach(r=>t.tBodies[0].appendChild(r));
if(t===fundTable){{document.getElementById('sort-column').value=String(col);document.getElementById('sort-order').value=asc?'asc':'desc';}}}}
function filterTable(){{const q=document.getElementById('search').value.trim().toLowerCase(),group=document.getElementById('fund-group').value;let count=0;originalRows.forEach(r=>{{const show=r.textContent.toLowerCase().includes(q)&&(group==='All'||r.cells[0].textContent.trim()===group);r.style.display=show?'':'none';if(show)count++;}});document.getElementById('row-count').textContent=count+' of '+originalRows.length+' funds';}}
document.querySelectorAll('table').forEach(t=>t.querySelectorAll('th').forEach((h,i)=>{{h.setAttribute('tabindex','0');h.setAttribute('aria-sort','none');h.onclick=()=>sortTable(t,i,h.getAttribute('aria-sort')!=='ascending');h.onkeydown=e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();h.onclick();}}}};}}));
const applySort=()=>sortTable(fundTable,Number(document.getElementById('sort-column').value),document.getElementById('sort-order').value==='asc');document.getElementById('sort-column').onchange=applySort;document.getElementById('sort-order').onchange=applySort;document.getElementById('search').oninput=filterTable;document.getElementById('fund-group').onchange=filterTable;
document.getElementById('reset-table').onclick=()=>{{document.getElementById('search').value='';document.getElementById('fund-group').value='All';originalRows.forEach(r=>fundTable.tBodies[0].appendChild(r));fundTable.querySelectorAll('th').forEach(h=>h.setAttribute('aria-sort','none'));filterTable();}};applySort();filterTable();</script></main></body></html>'''
    path=OUT/'fund_sensitivity_monitor.html';path.write_text(report,encoding='utf-8');return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--update',action='store_true');p.add_argument('--workbook',type=Path,default=WORKBOOK);p.add_argument('--mbs-workbook',type=Path,default=MBS_WORKBOOK);a=p.parse_args()
    OUT.mkdir(exist_ok=True);u=read_universe(a.workbook)
    end=pd.Timestamp(datetime.now(ZoneInfo('Asia/Shanghai')).date());start=end-pd.DateOffset(years=2)-pd.Timedelta(days=10)
    paths=[OUT/n for n in ['reference.csv','fund_total_return_indices.csv','fx.csv','bbg_factors.csv']]
    refresh=a.update or not all(p.exists() for p in paths)
    if not refresh:refresh=set(pd.read_csv(paths[0]).ticker)!=set(u.ticker)
    if refresh:
        ref,raw,fx,factors,ig_source,errors=download(u,start,end)
        for d,pth in zip([ref,raw,fx],paths[:3]):d.to_csv(pth,index=False)
        factors.to_csv(paths[3],index_label='date')
        manifest=dict(fetched_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),start=str(start.date()),end=str(end.date()),ig_source=ig_source)
        (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (OUT/'bloomberg_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    else:
        ref,raw,fx=[pd.read_csv(pth) for pth in paths[:3]];factors=pd.read_csv(paths[3],index_col='date',parse_dates=True)
        manifest=json.loads((OUT/'source_manifest.json').read_text())
    factors=factors.join(workbook_factors(a.mbs_workbook),how='outer').sort_index()
    last_friday=end-pd.Timedelta(days=(end.weekday()-4)%7)
    meta,returns,changes=make_weekly(u,ref,raw,fx,factors,last_friday)
    cutoff=end-pd.DateOffset(years=2)
    returns=returns.loc[cutoff:];changes=changes.loc[cutoff:]
    summary,coefficients,rolling,fitted=analyze(meta,returns,changes)
    covered=meta[meta.ticker.isin(summary.loc[summary.status=='Available','ticker'])]
    baskets=group_returns(covered,returns)
    baskets.columns=baskets.columns.str.replace(' equal weight',' covered equal weight',regex=False)
    group_rows=[]
    for name in baskets:
        m,d=fit_model(baskets[name],changes)
        group=name.split(' ')[0]
        constituents=len(covered) if group=='All' else int((covered.group==group).sum())
        listed=len(meta) if group=='All' else int((meta.group==group).sum())
        row=dict(basket=name,constituents=constituents,listed_funds=listed,observations=len(d),status='Available' if m is not None else 'Insufficient common observations')
        members=covered if group=='All' else covered[covered.group==group]
        values=members.ytd_return_pct
        row['ytd_funds']=int(values.notna().sum())
        row['ytd_return_pct']=float(values.mean()) if len(values) and values.notna().all() else np.nan
        if m is not None:row.update(r2=m.rsquared,**{'beta_'+f:m.params[f] for f in FACTORS},**{'corr_'+f:d.return_bps.corr(d[f]) for f in FACTORS})
        group_rows.append(row)
    groups=pd.DataFrame(group_rows)
    for frame,name in [(meta,'coverage'),(summary,'fund_sensitivities'),(coefficients,'coefficient_uncertainty'),(groups,'group_sensitivities')]:frame.to_csv(OUT/f'{name}.csv',index=False)
    for frame,name in [(returns,'weekly_fund_returns_bps'),(changes,'weekly_factor_changes_bps'),(baskets,'weekly_group_returns_bps'),(factors,'factor_levels')]:frame.to_csv(OUT/f'{name}.csv',index_label='date')
    for i,t in enumerate(meta.ticker):rolling[t].to_csv(OUT/f'rolling_{i:02d}.csv',index=False)
    path=build_report(meta,summary,coefficients,rolling,fitted,returns,changes,baskets,groups,manifest)
    print('Estimable funds',len(summary[summary.status=='Available']), '/',len(meta));print(groups.to_string(index=False));print('REPORT',path)
    return path


if __name__=='__main__':main()
