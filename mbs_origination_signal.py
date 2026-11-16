"""Walk-forward origination signals versus future JPM coupon performance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import mbs_origination_analysis as source

OUT=Path(__file__).parent/'mbs_origination_signal_output'
MIN_TRAIN=250


def adjusted_p(values):
    result=pd.Series(np.nan,index=values.index)
    valid=values.notna() & np.isfinite(values)
    result.loc[valid]=multipletests(values.loc[valid],method='fdr_bh')[1]
    return result


def robustness():
    rows=[]
    for coupon in ('FN 6.5','G2 6.5'):
        d=pd.read_csv(OUT/(coupon.replace(' ','_')+'_predictions.csv'))
        mask=d.train_mean.notna().shift(1,fill_value=False)
        for strategy in ('always_short','always_long','high_only','orig_rule'):
            target=(pd.Series(-1. if strategy=='always_short' else 1.,index=d.index)
                    if strategy.startswith('always') else d[strategy+'_target'])
            for lag in (1,2):
                pos=target.where(d.train_mean.notna(),0).fillna(0).shift(lag).rolling(5,min_periods=5).mean().fillna(0)
                pnl=(pos*d.perf)[mask];turnover=pos.diff().abs().fillna(0)[mask]
                fit=sm.OLS(pnl,np.ones((len(pnl),1))).fit(cov_type='HAC',cov_kwds={'maxlags':20})
                rows.append(dict(coupon=coupon,strategy=strategy,lag=lag,gross_ticks=pnl.sum(),
                    sharpe=np.sqrt(252)*pnl.mean()/pnl.std(),net_05=(pnl-.5*turnover).sum(),hac_p=fit.pvalues.iloc[0]))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'benchmark_delay_checks.csv',index=False)
    return frame


def features(orig, perf, window=20, horizon=5):
    # Same observation calendar for both series; only historical information in X.
    d=pd.concat([orig.rename('orig'),perf.rename('perf')],axis=1).sort_index()
    d=d.loc[d.perf.notna()]
    flow=d.orig.rolling(window,min_periods=window).sum()
    prior=flow.shift(1)
    d['orig_z']=(flow-prior.rolling(250,min_periods=120).mean())/prior.rolling(250,min_periods=120).std()
    d['past_perf']=d.perf.rolling(window).sum()
    d['future']=pd.concat([d.perf.shift(-k) for k in range(1,horizon+1)],axis=1).sum(axis=1,min_count=horizon)
    return d.replace([np.inf,-np.inf],np.nan)


def simulate(d,horizon):
    cols={'orig_model':['orig_z'],'perf_baseline':['past_perf'],'combined':['orig_z','past_perf']}
    for name in cols:d[name]=np.nan
    d['train_mean']=np.nan
    models={};first=None
    for i in range(len(d)):
        if d.iloc[i][['orig_z','past_perf']].isna().any():continue
        # Label at j ends j+H; require j+H < i, including a one-observation embargo.
        train=d.iloc[:max(0,i-horizon)].dropna(subset=['orig_z','past_perf','future'])
        if len(train)<MIN_TRAIN:continue
        if first is None:first=i
        if not models or d.index[i].to_period('M')!=d.index[i-1].to_period('M'):
            models={name:sm.OLS(train.future,sm.add_constant(train[c],has_constant='add')).fit() for name,c in cols.items()}
            scale=train.future.std();mean=train.future.mean()
        d.loc[d.index[i],'train_mean']=mean
        for name,c in cols.items():
            prediction=float(models[name].predict([ [1.]+d.iloc[i][c].tolist() ])[0])
            d.loc[d.index[i],name]=prediction
            d.loc[d.index[i],name+'_target']=np.sign(prediction) if abs(prediction)>.2*scale else 0.
    if first is None:return None,None,None
    # Fixed hypotheses, no fitted threshold: high flow short; low flow long.
    d['orig_rule_target']=np.where(d.orig_z>=1,-1,np.where(d.orig_z<=-1,1,0)).astype(float)
    d['high_only_target']=np.where(d.orig_z>=1,-1,0).astype(float)
    d['always_short_target']=-1.
    d['always_long_target']=1.
    rows=[];daily=[]
    for name in list(cols)+['orig_rule','high_only','always_short','always_long']:
        target=d[name+'_target'].where(d.train_mean.notna(),0).fillna(0)
        # H equal staggered sleeves; signal at t enters for t+1 performance.
        position=target.shift(1).rolling(horizon,min_periods=horizon).mean().fillna(0)
        pnl=position*d.perf;turnover=position.diff().abs().fillna(position.abs())
        period=d.train_mean.notna().shift(1,fill_value=False)
        p=pnl.loc[period];t=turnover.loc[period]
        if not len(p):continue
        fit=sm.OLS(p.values,np.ones((len(p),1))).fit(cov_type='HAC',cov_kwds={'maxlags':max(20,horizon)})
        ci=fit.conf_int()[0]*252
        net={str(cost):float((p-cost*t).sum()) for cost in (0,.25,.5,1)}
        row=dict(strategy=name,obs=len(p),start=str(p.index[0].date()),end=str(p.index[-1].date()),gross_ticks=p.sum(),
            sharpe=np.sqrt(252)*p.mean()/p.std() if p.std()>0 else 0.,hac_p=float(fit.pvalues[0]),
            annual_mean_ci_low=ci[0],annual_mean_ci_high=ci[1],turnover=t.sum(),break_even_cost=p.sum()/t.sum() if t.sum() else np.nan,
            net_025=net['0.25'],net_05=net['0.5'],net_1=net['1'],active_fraction=(position.loc[period]!=0).mean(),
            max_drawdown=float((p.cumsum()-p.cumsum().cummax().clip(lower=0)).min()),
            latest_orig_z=d.orig_z.iloc[-1],latest_target=target.iloc[-1])
        if name in cols:
            scored=d.dropna(subset=[name,'future','train_mean'])
            denom=((scored.future-scored.train_mean)**2).sum()
            row.update(oos_r2=1-((scored.future-scored[name])**2).sum()/denom if denom else np.nan,
                forecast_ic=scored[name].corr(scored.future))
        rows.append(row)
        daily.extend(dict(date=str(day.date()),strategy=name,pnl_ticks=float(p.loc[day]),turnover=float(t.loc[day]),position=float(position.loc[day])) for day in p.index)
    return pd.DataFrame(rows),pd.DataFrame(daily),d


def main():
    OUT.mkdir(exist_ok=True)
    orig=source.load_origination();perf=source.load_perf()
    summaries=[];histories=[];associations=[];sens=[];currents=[]
    for name in orig.columns.intersection(perf.columns):
        print('Analyzing',name,flush=True)
        d=features(orig[name],perf[name]);summary,daily,full=simulate(d,5)
        if summary is None:continue
        summary['coupon']=name;daily['coupon']=name;summaries.append(summary);histories.append(daily)
        full.to_csv(OUT/(name.replace(' ','_')+'_predictions.csv'),index_label='date')
        a=d.dropna(subset=['orig_z','past_perf','future'])
        fit=sm.OLS(a.future,sm.add_constant(a[['orig_z','past_perf']])).fit(cov_type='HAC',cov_kwds={'maxlags':20})
        associations.append(dict(coupon=name,n=len(a),orig_beta=fit.params.orig_z,hac_p=fit.pvalues.orig_z))
        # Current targets are descriptive research signals, not execution instructions.
        currents.append(dict(coupon=name,asof=str(d.index[-1].date()),orig_z=full.orig_z.iloc[-1],
            rule_target=full.orig_rule_target.iloc[-1],model_target=full.combined_target.iloc[-1]))
        if name in ('FN 6.5','G2 6.5'):
            for w in (10,20,30):
                for h in (1,5,20):
                    s,_,_=simulate(features(orig[name],perf[name],w,h),h)
                    if s is not None:s['coupon']=name;s['window']=w;s['horizon']=h;sens.append(s)
    s=pd.concat(summaries,ignore_index=True);daily=pd.concat(histories,ignore_index=True);a=pd.DataFrame(associations)
    a['fdr_p']=adjusted_p(a.hac_p)
    s['fdr_p']=adjusted_p(s.hac_p)
    s.to_csv(OUT/'strategy_summary.csv',index=False);a.to_csv(OUT/'predictive_association.csv',index=False)
    daily.to_csv(OUT/'daily_backtests.csv',index=False);pd.concat(sens).to_csv(OUT/'sensitivity_65.csv',index=False)
    pd.DataFrame(currents).to_csv(OUT/'current_signals.csv',index=False)
    primary=s[s.coupon.isin(['FN 6.5','G2 6.5'])]
    figs=[]
    for coupon in ('FN 6.5','G2 6.5'):
        fig=go.Figure()
        for strategy,g in daily[daily.coupon==coupon].groupby('strategy'):
            fig.add_scatter(x=g.date,y=g.pnl_ticks.cumsum(),name=strategy,mode='lines')
        fig.update_layout(title=coupon+' — out-of-sample cumulative performance (ticks)',template='plotly_white',height=450,yaxis_title='32nds (ticks), unit exposure')
        figs.append(fig.to_html(full_html=False,include_plotlyjs=False))
    def table(frame):return frame.to_html(index=False,float_format=lambda v:f'{v:.3f}',border=0)
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Origination signal research</title><style>body{font:14px Arial;margin:30px;color:#193244}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #ddd}section{overflow:auto;margin:25px 0}p{max-width:1100px;line-height:1.6}</style><script>'''+get_plotlyjs()+'''</script></head><body><h1>Coupon origination: predictive signal research</h1>
    <p>Primary specification fixed before testing: 20-observation origination sum, standardized against the preceding 250 observations (minimum 120); predict the next 5 observations of JPM coupon performance. Positive performance benefits a long coupon exposure. Model features observed through signal close; first P&amp;L is next observation. Source release times are unavailable: the test assumes saved daily origination was available by that close. Missing origination values are treated as zero by the source loader, which may bias results if they represent unavailable data.</p>
    <p>Expanding training, minimum 250 fully realized labels, monthly OLS refits, one-observation label embargo. Compare origination-only, past-performance-only and combined models. Model trades only when |forecast| exceeds 0.2 training target standard deviations. Fixed rule: short when origination z ≥ 1, long when z ≤ −1; high-only rule shorts high origination. Five staggered equal sleeves hold signals for five observations; gross exposure ≤ 1. No parameter selected by test performance. Costs are ticks per unit one-way turnover; net columns use 0.25, 0.5 and 1 tick, not calibrated execution estimates. No funding, carry or roll. This is coupon performance exposure, not a fully specified hedgeable trade.</p>
    <p>Sharpe uses daily calendar-aligned performance including flat days. HAC mean tests use 20 lags; FDR adjusts the primary coupon/strategy family. Same-window regressions can reflect reverse causality. Full-sample predictive coefficients below are descriptive; the walk-forward test is the effectiveness check. Overlapping forward labels affect forecast R²/IC; strategy P&amp;L uses realized daily observations. Coupon selection and sensitivity results are exploratory; independent future validation remains necessary.</p>
    <h2>6.5 coupon primary out-of-sample results</h2><section>'''+table(primary)+ '</section>'+''.join(figs)+'<h2>Predictive origination coefficient controlling for past performance</h2><section>'+table(a)+'</section><h2>All coupon strategy results</h2><section>'+table(s)+'</section><h2>6.5 sensitivity (exploratory)</h2><section>'+table(pd.concat(sens))+'</section></body></html>'
    (OUT/'Origination_Signal_Research.html').write_text(html,encoding='utf-8')
    checks=robustness()
    path=OUT/'Origination_Signal_Research.html'
    path.write_text(html.replace('<h2>6.5 coupon primary','<h2>Direction and signal-delay checks</h2>'+table(checks)+'<h2>6.5 coupon primary'),encoding='utf-8')
    print(primary.to_string(index=False));print('REPORT',OUT/'Origination_Signal_Research.html')

if __name__=='__main__':
    from plotly.offline import get_plotlyjs
    main()
