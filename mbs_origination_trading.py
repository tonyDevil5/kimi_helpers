"""Daily all-coupon origination positions on JPM performance, per $10mm."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go
import mbs_origination_analysis as source
from mbs_origination_signal import adjusted_p
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'BackTestingTool'))
from performance_position_backtester import PerformancePositionBacktester
OUT=ROOT/'mbs_origination_signal_output'


def prepare(orig,perf):
    flow=orig.reindex(perf.index).rolling(20,min_periods=20).sum()
    prior=flow.shift(1)
    z=(flow-prior.rolling(250,min_periods=250).mean())/prior.rolling(250,min_periods=250).std().replace(0,np.nan)
    z=z.replace([np.inf,-np.inf],np.nan)
    raw=pd.Series(np.where(z>=1,-1.,np.where(z<=-1,1.,0.)),index=perf.index)
    # Five equal daily sleeves, with up to one full $10mm exposure per coupon.
    target=raw.rolling(5,min_periods=5).mean().fillna(0)
    target=target.where(z.notna(),0)
    return z,raw,target


def metrics(p):
    sd=p.std();cum=p.cumsum();fit=sm.OLS(p,np.ones((len(p),1))).fit(cov_type='HAC',cov_kwds={'maxlags':20})
    return dict(obs=len(p),net_pnl_usd=p.sum(),sharpe=np.sqrt(252)*p.mean()/sd if sd>0 else np.nan,
        max_drawdown_usd=(cum-cum.cummax().clip(lower=0)).min(),hac_p=fit.pvalues.iloc[0])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--notional',type=float,default=10_000_000)
    parser.add_argument('--cost-ticks',type=float,default=.5);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    raw=pd.read_excel(source.rv.CITI_FNAME,sheet_name='MBS_ORIG',header=1,index_col=0)
    raw.index=pd.to_datetime(raw.index);raw=raw.sort_index()
    if raw.index.has_duplicates:raise ValueError('Duplicate origination dates')
    orig={}
    for col in raw:
        m=source._ORIG_RE.search(str(col))
        if m:orig[f'{source._AGENCY_MAP[m[1]]} {float(m[2]+"."+m[3]):g}']=pd.to_numeric(raw[col],errors='coerce').where(lambda x:x>=0)
    orig=pd.DataFrame(orig)
    missing_counts=orig.isna().sum().to_dict()
    strict_orig=orig.copy()
    # Match the documented Citi tracker convention in the existing monitor.
    orig=orig.fillna(0.)
    perf=source.load_perf()
    if perf.index.has_duplicates or perf.isna().any().any():raise ValueError('Unresolved JPM performance gaps')
    summaries=[];signals=[];daily=[];payload={};sensitivity=[];strict_rows=[]
    for name in perf:
        o=orig[name] if name in orig else pd.Series(np.nan,index=perf.index)
        z,rule,target=prepare(o,perf[name])
        frame=pd.DataFrame(dict(daily_ticks=perf[name],target=target))
        bt=PerformancePositionBacktester(frame,args.notional,1,args.cost_ticks);d=bt.run()
        # Same retrospective comparison period as prior research, never optimized here.
        scored=d.loc[d.index>=pd.Timestamp('2025-06-12')]
        row=dict(coupon=name,**metrics(scored.net_pnl_usd),gross_pnl_usd=scored.gross_pnl_usd.sum(),cost_usd=scored.cost_usd.sum(),
            active_pct=100*scored.held_position.ne(0).mean(),valid_orig_days=z.notna().sum())
        summaries.append(row)
        current=d.position.iloc[-1];next_target=target.iloc[-1];delta=next_target-current
        signals.append(dict(coupon=name,asof=str(perf.index[-1].date()),orig_z=z.iloc[-1],raw_signal=rule.iloc[-1],
            ready=bool(pd.notna(z.iloc[-1])),blank_source_values=int(missing_counts.get(name,0)),current_simulated_notional=current*args.notional,
            next_close_target_notional=next_target*args.notional,rebalance_notional=delta*args.notional,
            action='BLOCKED: missing/insufficient origination' if pd.isna(z.iloc[-1]) else 'BUY' if delta>1e-9 else 'SELL' if delta< -1e-9 else 'HOLD',
            hedge='JPM Treasury hedge implicit in Perf; hedge contracts require desk mapping'))
        d['coupon']=name;d['orig_z']=z;d['raw_signal']=rule;daily.append(d)
        d.to_csv(OUT/(name.replace(' ','_')+'_tradable_daily.csv'),index_label='date')
        rebalances=bt.rebalances.copy();rebalances['notional_change']=rebalances.rebalance*args.notional
        rebalances.to_csv(OUT/(name.replace(' ','_')+'_rebalance_log.csv'),index_label='date')
        payload[name]=dict(dates=d.index.strftime('%Y-%m-%d').tolist(),z=[None if pd.isna(v) else float(v) for v in z],
            position=d.position.tolist(),pnl=d.cumulative_pnl_usd.tolist())
        for lag,cost in [(1,0),(1,.25),(1,.5),(1,1),(2,.5)]:
            test=PerformancePositionBacktester(frame,args.notional,lag,cost).run().loc[scored.index]
            sensitivity.append(dict(coupon=name,execution_lag=lag,cost_ticks=cost,**metrics(test.net_pnl_usd)))
    for name in perf:
        strict_o=strict_orig[name] if name in strict_orig else pd.Series(np.nan,index=perf.index)
        strict_z,_,strict_target=prepare(strict_o,perf[name])
        strict_frame=pd.DataFrame(dict(daily_ticks=perf[name],target=strict_target))
        strict_d=PerformancePositionBacktester(strict_frame,args.notional,1,args.cost_ticks).run()
        strict_rows.append(dict(coupon=name,valid_orig_days=int(strict_z.notna().sum()),**metrics(strict_d.loc[strict_d.index>=pd.Timestamp('2025-06-12'),'net_pnl_usd'])))
    pd.DataFrame(strict_rows).to_csv(OUT/'strict_missing_all_coupon_performance.csv',index=False)
    summary=pd.DataFrame(summaries);summary['fdr_p']=adjusted_p(summary.hac_p)
    signals=pd.DataFrame(signals);history=pd.concat(daily).reset_index(names='date')
    summary.to_csv(OUT/'all_coupon_performance.csv',index=False);signals.to_csv(OUT/'all_coupon_signals.csv',index=False)
    history.to_csv(OUT/'all_coupon_daily.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'all_coupon_cost_delay.csv',index=False)
    print(summary[['coupon','net_pnl_usd','sharpe','fdr_p']].to_string(index=False))
    (OUT/'signal_config.json').write_text(json.dumps(dict(notional=args.notional,cost_ticks=args.cost_ticks,window=20,z_window=250,entry=1,holding_sleeves=5,execution_lag=1,evaluation_start='2025-06-12'),indent=2))
    from mbs_origination_dashboard import render
    print('REPORT',render(OUT))

if __name__=='__main__':main()
