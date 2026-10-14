"""Do observed fund flows add information about the following market week?

Depends on fund_flows_monitor.py and fund_sensitivity_monitor.py cached inputs.
Monday-close-to-Monday-close targets; expanding, chronological out-of-sample tests.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import html

import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from fund_sensitivity_monitor import FACTORS

ROOT=Path(__file__).parent
OUT=ROOT/'fund_flow_forward_output'
FLOW=['Active flow ($bn)','Passive flow ($bn)']
CONTROLS=['Prior Δ '+f for f in FACTORS]


def cohort_and_flows(universe,daily,weekly,qualification=12):
    """Choose a fixed reporting cohort using only the first qualification weeks."""
    cutoff=weekly.index[qualification-1]
    records=[]
    for r in universe.itertuples():
        history=daily[(daily.ticker==r.ticker)&(daily.date<=cutoff)].date.sort_values()
        gap=history.diff().dt.days.dropna().median() if len(history)>1 else np.nan
        coverage=weekly[r.ticker].iloc[:qualification].notna().mean()
        included=len(history)>=8 and pd.notna(gap) and gap<=7 and coverage>=.8
        records.append(dict(ticker=r.ticker,group=r.group,name=r.name,included=included,
                            qualification_observations=len(history),median_gap_days=gap,
                            qualification_week_coverage=coverage))
    cohort=pd.DataFrame(records)
    flow=pd.DataFrame(index=weekly.index)
    counts={}
    for group in ['Active','Passive']:
        tickers=cohort[(cohort.group==group)&cohort.included].ticker.tolist()
        counts[group]=len(tickers)
        if not tickers:raise ValueError(f'No eligible {group} funds in the initial qualification sample')
        # Never turn incomplete cohort coverage into a smaller observed subtotal.
        flow[group+' flow ($bn)']=weekly[tickers].sum(axis=1,min_count=len(tickers))/1000
    flow['All flow ($bn)']=flow[FLOW].sum(axis=1,min_count=2)
    return cohort,flow.loc[cutoff+pd.Timedelta(days=1):],counts,cutoff


def forward_targets(levels,weeks):
    """Monday close or the first complete market close after a Monday holiday."""
    levels=levels[FACTORS].dropna().sort_index()
    records=[]
    for week in weeks:
        scheduled_start=week+pd.Timedelta(days=3)
        scheduled_end=week+pd.Timedelta(days=10)
        def endpoint(day):
            s=levels.loc[day:day+pd.Timedelta(days=2)]
            return s.iloc[0] if len(s) else None,s.index[0] if len(s) else pd.NaT
        start,sd=endpoint(scheduled_start);finish,ed=endpoint(scheduled_end)
        row=dict(week_ending=week,scheduled_start=scheduled_start,scheduled_end=scheduled_end,
                 actual_start=sd,actual_end=ed)
        for factor in FACTORS:row[factor]=float(finish[factor]-start[factor]) if start is not None and finish is not None else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index('week_ending')


def available_cohort_flows(cohort,weekly,week):
    """Refresh reporting eligibility using only the previous twelve completed flow weeks.

    Historical aggregate features are recalculated for each forecast's fixed cohort.
    This is causal feature selection at the forecast date, not performance selection.
    """
    history=weekly.loc[:week-pd.Timedelta(days=7)].tail(12)
    qualified=cohort[cohort.included].copy()
    coverage=history.notna().mean()
    qualified['recent_prior_week_coverage']=qualified.ticker.map(coverage)
    included=qualified[qualified.recent_prior_week_coverage>=.8]
    flow=pd.DataFrame(index=weekly.index)
    counts={}
    for group in ['Active','Passive']:
        names=included[included.group==group].ticker.tolist()
        counts[group]=len(names)
        flow[group+' flow ($bn)']=weekly[names].sum(axis=1,min_count=len(names))/1000 if names else np.nan
    flow['All flow ($bn)']=flow[FLOW].sum(axis=1,min_count=2)
    return flow,counts,included


def regress(y,x):
    d=pd.concat([y.rename('target'),x],axis=1).dropna()
    design=sm.add_constant(d[x.columns],has_constant='add')
    if len(d)<max(20,len(x.columns)+3) or np.linalg.matrix_rank(design)<design.shape[1]:return None,d
    return sm.OLS(d.target,design).fit(cov_type='HAC',cov_kwds={'maxlags':4}),d


def walkforward(flow,controls,targets,extra_lag=0,min_train=52,weekly=None,cohort=None):
    shifted=flow[FLOW].shift(extra_lag)
    features=pd.concat([controls,shifted],axis=1)
    logs=[];forecasts=[]
    for factor in FACTORS:
        for week in features.index:
            if weekly is not None:
                local,counts,members=available_cohort_flows(cohort,weekly,week)
                features=pd.concat([controls,local[FLOW].shift(extra_lag).reindex(controls.index)],axis=1)
            x=features.loc[week]
            if x.isna().any():continue
            decision=targets.loc[week,'scheduled_start']
            # Strict release-time cutoff: even previous target ending on this Monday
            # is excluded until a later signal, preserving a conservative timing rule.
            available=(targets.actual_end<decision)&(targets.index<week)
            train=pd.concat([targets.loc[available,factor].rename('target'),features.loc[available]],axis=1).dropna()
            if len(train)<min_train:continue
            aug,_=regress(train.target,train[features.columns])
            base,_=regress(train.target,train[CONTROLS])
            if aug is None or base is None:continue
            def prediction(model,cols):
                design=pd.DataFrame([[1.,*x[cols].tolist()]],columns=['const',*cols])
                return float(model.predict(design).iloc[0])
            pred=prediction(aug,features.columns.tolist());benchmark=prediction(base,CONTROLS)
            row=dict(factor=factor,extra_flow_lag_weeks=extra_lag,flow_week=week,
                     forecast_start=decision,forecast_end=targets.loc[week,'scheduled_end'],
                     training_obs=len(train),training_last_target_end=train.index.map(targets.actual_end).max(),
                     predicted_bps=pred,baseline_bps=benchmark,actual_bps=targets.loc[week,factor])
            if weekly is not None:
                row.update(active_funds=counts['Active'],passive_funds=counts['Passive'],
                           included_tickers=';'.join(members.ticker.tolist()))
            if pd.notna(row['actual_bps']):logs.append(row)
            else:forecasts.append(row)
    return pd.DataFrame(logs),pd.DataFrame(forecasts)


def score(predictions):
    rows=[]
    for (factor,lag),d in predictions.groupby(['factor','extra_flow_lag_weeks']):
        y=d.actual_bps;aug=d.predicted_bps;base=d.baseline_bps
        a=(y-aug).pow(2).mean();b=(y-base).pow(2).mean();zero=y.pow(2).mean()
        nonzero=y!=0
        rows.append(dict(factor=factor,extra_flow_lag_weeks=lag,test_weeks=len(d),
                         flow_model_rmse_bps=np.sqrt(a),baseline_rmse_bps=np.sqrt(b),
                         improvement_vs_baseline_pct=(1-a/b)*100 if b else np.nan,
                         oos_r2_vs_zero=1-a/zero if zero else np.nan,
                         predicted_actual_corr=y.corr(aug),
                         directional_hit_pct=((np.sign(y[nonzero])==np.sign(aug[nonzero])).mean()*100),
                         baseline_directional_hit_pct=((np.sign(y[nonzero])==np.sign(base[nonzero])).mean()*100)))
    return pd.DataFrame(rows)


def report(cohort,flow,counts,cutoff,targets,correlations,coefficients,metrics,logs,forecasts,sources):
    def table(d):return d.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:,.3f}')
    fig=go.Figure()
    for group in ['Active','Passive','All']:
        field=group+' flow ($bn)';d=correlations[correlations.flow_group==group]
        fig.add_trace(go.Bar(x=d.factor,y=d.correlation,name=group))
    fig.update_layout(template='plotly_white',title='Historical flow / forward market-change correlations',height=430,yaxis_title='Correlation',barmode='group')
    chart=fig.to_html(full_html=False,include_plotlyjs=True)
    payload={}
    for factor in FACTORS:
        d=logs[(logs.factor==factor)&(logs.extra_flow_lag_weeks==0)]
        pair=pd.concat([flow['All flow ($bn)'],targets[factor]],axis=1).dropna()
        payload[factor]=dict(dates=d.flow_week.dt.strftime('%Y-%m-%d').tolist(),actual=d.actual_bps.tolist(),predicted=d.predicted_bps.tolist(),baseline=d.baseline_bps.tolist(),
                             flow=pair.iloc[:,0].tolist(),forward=pair.iloc[:,1].tolist(),scatter_dates=pair.index.strftime('%Y-%m-%d').tolist())
    safe=json.dumps(payload,allow_nan=False).replace('</','<\\/')
    primary=metrics[metrics.extra_flow_lag_weeks==0]
    strongest=primary.sort_values('improvement_vs_baseline_pct',ascending=False)
    best=strongest.iloc[0] if len(strongest) else None
    verdict=('No target has enough out-of-sample observations.' if best is None else
             f"Largest measured improvement: {best.factor}, {best.improvement_vs_baseline_pct:+.1f}% versus the market-only model across {int(best.test_weeks)} test weeks. "
             'This is an exploratory comparison across five targets, not evidence that a selected target will remain predictable.')
    latest=flow.tail(1).reset_index()
    doc=f'''<!doctype html><html><head><meta charset="utf-8"><title>Fund flows and forward market moves</title><style>
body{{font:14px Arial;background:#eef2f6;color:#17202a;margin:22px}}main{{max-width:1500px;margin:auto}}section{{background:white;padding:22px;margin:16px 0;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th{{background:#173e56;color:white;padding:9px;cursor:pointer}}td{{padding:8px;border:1px solid #ddd;text-align:right}}tr:nth-child(even){{background:#f3f6f9}}select{{padding:9px}}summary{{cursor:pointer;font-weight:bold}}</style></head><body><main>
<section><h1>Fund flows and the following market week</h1><p><a href="../fund_flows_output/fund_flows_monitor.html">Fund flow monitor</a> · <a href="../fund_sensitivity_output/fund_sensitivity_monitor.html">Return sensitivities</a></p><p>Question: do Friday-ending fund flows add information about the market move from the following Monday close to the next Monday close? Positive CT10 means yields rise; positive FN basis/IG OAS means spreads widen; positive 2s10s means the curve steepens.</p><p><b>{html.escape(verdict)}</b></p><p>Latest research cohort: {counts['Active']} active and {counts['Passive']} passive share classes. Initial qualification uses reporting frequency and coverage in the first 12 weeks only, through {cutoff.date()}. At each forecast, retain qualified funds with ≥80% coverage in the previous twelve weeks; no later return or target is used. Historical features are rebuilt for that forecast's cohort. This differs from the flow monitor's current 39-fund subset. Any incomplete included-cohort week is unavailable, not zero.</p></section>
<section><h2>Out-of-sample results — do flows improve the forecast?</h2>{table(primary.drop(columns='extra_flow_lag_weeks'))}<p>Market-only baseline uses the preceding Friday-to-Friday changes in all five factors. The flow model adds active and passive net flows ($bn). Each prediction is fitted on earlier completed targets only, with at least 52 training weeks. Positive improvement means lower squared forecast error; negative means adding flows made forecasts worse. Directional accuracy excludes zero actual moves. Out-of-sample R² compares with a zero-change forecast and can be negative.</p></section>
<section><h2>Historical lead relationships</h2>{table(correlations)}{chart}<p>These full-sample correlations use the latest research cohort retrospectively, are descriptive and do not control for market moves. The out-of-sample comparison uses each historical forecast's own reporting cohort and tests incremental information. Neither establishes that flows cause market moves.</p></section>
<section><h2>Prediction and scatter detail</h2><select id="target">{''.join('<option>'+f+'</option>' for f in FACTORS)}</select><div id="prediction-chart"></div><div id="scatter-chart"></div><p>Prediction chart: forecasts were produced before the plotted target week completed. Scatter: reported total cohort flow versus the future market change. Hover for the flow-week date.</p></section>
<section><h2>Latest complete cohort flows and forward estimates</h2>{table(latest)}{table(forecasts[forecasts.extra_flow_lag_weeks==0]) if len(forecasts) else '<p>No pending estimates.</p>'}<p>Point estimates are exploratory model outputs, not trade recommendations. An estimate can be formed before Monday close; its target starts at that close. Complete data for the target week are needed before it can be scored.</p></section>
<section><details><summary>Extra reporting-lag sensitivity</summary>{table(metrics[metrics.extra_flow_lag_weeks==1])}<p>This repeats the test with fund flows one additional week old, using the same Monday-based targets. It checks whether the result depends on aggressive assumptions about flow availability.</p></details></section>
<section><details><summary>Regression coefficients, cohort and timing audit</summary>{table(coefficients)}{table(cohort)}{table(logs[['factor','flow_week','forecast_start','forecast_end','training_obs','training_last_target_end']])}<p>Coefficient units: future factor-change bps per $1bn group flow, conditional on preceding market changes. HAC uncertainty uses four matched-observation lags. Multiple targets are explored without significance-based strategy selection.</p></details></section>
<section><details><summary>Data limitations and methodology</summary><p>Bloomberg historical FUND_FLOW values are current revised snapshots, not publication-time vintages. US ETF flows use Bloomberg process dates; mutual-fund observations can arrive with different lags. Assuming a Friday-ending week's complete flows are known by the following Monday close is a working assumption, not verified for each observation. The extra-week-lag test is a conservative sensitivity check. The current workbook universe is not a historical point-in-time universe.</p><p>First 12 weeks choose frequent reporters: ≥8 observations, median spacing ≤7 days and ≥80% weekly coverage. Monthly/sparse reporters are excluded. Returns and AUM are not used to select the cohort. Group flows are USD net creations/redemptions summed without future imputation. Main targets use the first common factor close on or within two calendar days after each scheduled Monday, accommodating holidays; dates are audited in targets.csv. No Friday-to-Monday move before the forecast start is included. Training targets ending on the decision Monday are conservatively excluded.</p><p>Flow snapshot: {html.escape(sources['flows']['fetched_at'])}. Factor snapshot: {html.escape(sources['factors']['fetched_at'])}. Factor units and sources follow the fund sensitivity monitor. Effective out-of-sample samples are short; weak correlations, negative forecast R² or no error improvement should be treated as no useful predictive evidence in this test.</p></details></section>
<script type="application/json" id="payload">{safe}</script><script>const data=JSON.parse(document.getElementById('payload').textContent);function render(){{let f=document.getElementById('target').value,d=data[f];Plotly.react('prediction-chart',[{{x:d.dates,y:d.actual,name:'Actual future move'}},{{x:d.dates,y:d.predicted,name:'Flow + market forecast'}},{{x:d.dates,y:d.baseline,name:'Market-only forecast'}}],{{template:'plotly_white',title:f+' — out-of-sample forward changes',yaxis:{{title:'bps'}},height:440,hovermode:'x unified'}});Plotly.react('scatter-chart',[{{x:d.flow,y:d.forward,text:d.scatter_dates,mode:'markers',hovertemplate:'Flow week %{{text}}<br>Flow %{{x:.2f}}bn<br>Future change %{{y:.2f}}bp<extra></extra>'}}],{{template:'plotly_white',title:f+' — cohort flow versus subsequent change',xaxis:{{title:'Net fund flow ($bn)'}},yaxis:{{title:'Future change (bps)'}},height:420}})}}document.getElementById('target').onchange=render;render();document.querySelectorAll('table').forEach(t=>t.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{{let asc=h.dataset.asc!=='true';h.dataset.asc=asc;[...t.tBodies[0].rows].sort((a,b)=>{{let x=a.cells[i].textContent.replace(/,/g,''),y=b.cells[i].textContent.replace(/,/g,'');if(x==='—'||y==='—')return x===y?0:x==='—'?1:-1;let v=Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y);return asc?v:-v}}).forEach(r=>t.tBodies[0].appendChild(r))}}));</script></main></body></html>'''
    path=OUT/'fund_flow_forward_analysis.html';path.write_text(doc,encoding='utf-8');return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--update',action='store_true');a=p.parse_args()
    if a.update:
        for script in ['fund_flows_monitor.py','fund_sensitivity_monitor.py']:
            subprocess.run([sys.executable,str(ROOT/script),'--update'],check=True)
    OUT.mkdir(exist_ok=True)
    flow_dir=ROOT/'fund_flows_output';factor_dir=ROOT/'fund_sensitivity_output'
    u=pd.read_csv(flow_dir/'universe.csv');daily=pd.read_csv(flow_dir/'daily_usd_flows.csv',parse_dates=['date'])
    weekly=pd.read_csv(flow_dir/'weekly_fund_flows.csv',index_col='week_ending',parse_dates=True)
    levels=pd.read_csv(factor_dir/'factor_levels.csv',index_col='date',parse_dates=True)
    changes=pd.read_csv(factor_dir/'weekly_factor_changes_bps.csv',index_col='date',parse_dates=True)
    cohort,flow,counts,cutoff=cohort_and_flows(u,daily,weekly)
    targets=forward_targets(levels,flow.index)
    controls=changes.reindex(flow.index).rename(columns=dict(zip(FACTORS,CONTROLS)))
    logs=[];forecasts=[]
    for lag in [0,1]:
        log,forecast=walkforward(flow,controls,targets,extra_lag=lag,weekly=weekly,cohort=cohort)
        if len(log):logs.append(log)
        if len(forecast):forecasts.append(forecast)
    if not logs:raise ValueError('Insufficient history for out-of-sample analysis')
    logs=pd.concat(logs,ignore_index=True)
    forecasts=pd.concat(forecasts,ignore_index=True) if forecasts else pd.DataFrame()
    metrics=score(logs)
    current_flow,counts,current_cohort=available_cohort_flows(cohort,weekly,flow.index[-1])
    flow=current_flow.reindex(flow.index)
    cohort['included_latest']=cohort.ticker.isin(current_cohort.ticker)
    correlations=[];coefficients=[]
    for factor in FACTORS:
        for group in ['All','Active','Passive']:
            pair=pd.concat([flow[group+' flow ($bn)'],targets[factor]],axis=1).dropna()
            correlations.append(dict(factor=factor,flow_group=group,paired_weeks=len(pair),correlation=pair.iloc[:,0].corr(pair.iloc[:,1])))
        model,d=regress(targets[factor],pd.concat([controls,flow[FLOW]],axis=1))
        if model is not None:
            for f in FLOW:
                coefficients.append(dict(factor=factor,flow_group=f,beta_bps_per_bn=model.params[f],hac_pvalue=model.pvalues[f],observations=len(d)))
    correlations=pd.DataFrame(correlations);coefficients=pd.DataFrame(coefficients)
    sources=dict(flows=json.loads((flow_dir/'source_manifest.json').read_text()),factors=json.loads((factor_dir/'source_manifest.json').read_text()))
    for frame,name in [(cohort,'cohort'),(correlations,'lead_correlations'),(coefficients,'flow_coefficients'),(logs,'out_of_sample_predictions'),(metrics,'out_of_sample_metrics'),(forecasts,'latest_estimates')]:frame.to_csv(OUT/f'{name}.csv',index=False)
    for frame,name in [(flow,'cohort_weekly_flows'),(targets,'targets'),(controls,'market_controls')]:frame.to_csv(OUT/f'{name}.csv',index_label='week_ending')
    path=report(cohort,flow,counts,cutoff,targets,correlations,coefficients,metrics,logs,forecasts,sources)
    print('Cohort',counts,'qualified through',cutoff.date());print(metrics.to_string(index=False));print('REPORT',path)
    return path


if __name__=='__main__':main()
