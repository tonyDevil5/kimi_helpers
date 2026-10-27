"""Bloomberg GT07 Govt: overlapping two-month yield-change distributions since 2009."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'adhoc_output'/'ust7y_2m_change'
TICKER='GT07 Govt'


def changes(yields, window=42):
    s=yields.dropna().sort_index()
    if s.index.has_duplicates:raise ValueError('Duplicate yield dates')
    if len(s)<=window:raise ValueError('Insufficient yield history')
    result=pd.DataFrame({'end_yield_pct':s,'end_date':s.index},index=s.index)
    result['start_date_42obs']=pd.Series(s.index,index=s.index).shift(window)
    result['start_yield_42obs_pct']=s.shift(window)
    result['change_42obs_bp']=(s-s.shift(window))*100
    cutoffs=s.index-pd.DateOffset(months=2)
    positions=s.index.searchsorted(cutoffs,side='right')-1
    result['calendar_cutoff']=cutoffs
    result['calendar_start_date']=[s.index[i] if i>=0 else pd.NaT for i in positions]
    result['calendar_start_yield_pct']=[s.iloc[i] if i>=0 else np.nan for i in positions]
    result['change_calendar2m_bp']=(s-result.calendar_start_yield_pct)*100
    # Latest-anchored, non-overlapping 42-observation intervals as a sensitivity check.
    ends=np.arange(len(s)-1,window-1,-window)[::-1]
    nonoverlap=result.iloc[ends].copy()
    return result,nonoverlap


def describe(label, values):
    a=values.dropna()
    latest=a.iloc[-1]
    row={'Definition':label,'Observations':len(a),'Latest (bp)':latest,'Mean (bp)':a.mean(),
        'Median (bp)':a.median(),'Std (bp)':a.std(),'Min (bp)':a.min(),'Max (bp)':a.max(),
        'Latest percentile (%)':a.le(latest).mean()*100,'Positive share (%)':a.gt(0).mean()*100,
        'Latest z-score':(latest-a.mean())/a.std()}
    for q in [.01,.05,.1,.25,.75,.9,.95,.99]:row[f'P{int(q*100):02d} (bp)']=a.quantile(q)
    return row


def main(from_cache=False):
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'gt07_bloomberg_yields.csv'
    pulled=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    if not from_cache:
        from xbbg import blp
        from ust_technical_monitor import _normalize_bdh
        print(f'Downloading {TICKER} PX_LAST since 2009-01-01...',flush=True)
        raw=blp.bdh(tickers=[TICKER],flds=['PX_LAST'],start_date='2009-01-01',
            end_date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d'))
        data=_normalize_bdh(raw,[TICKER])
        if isinstance(data.columns,pd.MultiIndex):data.columns=data.columns.get_level_values(0)
        s=pd.to_numeric(data[TICKER],errors='coerce').dropna().sort_index()
        s=s.loc[(s.index>='2009-01-01')&(s.index.dayofweek<5)].rename('yield_pct')
        if s.empty:raise ValueError('Bloomberg returned no valid 7Y yields')
        s.to_csv(cache,index_label='date')
        (OUT/'source_metadata.json').write_text(json.dumps(dict(ticker=TICKER,field='PX_LAST',
            requested_start='2009-01-01',pulled_china_time=pulled),indent=2),encoding='utf-8')
    else:
        s=pd.read_csv(cache,index_col='date',parse_dates=True).yield_pct
        pulled=json.loads((OUT/'source_metadata.json').read_text(encoding='utf-8'))['pulled_china_time']
    d,nonoverlap=changes(s)
    a=d.change_42obs_bp.dropna();b=d.change_calendar2m_bp.dropna()
    summary=pd.DataFrame([describe('42 observations (main; overlapping)',a),
        describe('2 calendar months (overlapping)',b),
        describe('42 observations (non-overlapping)',nonoverlap.change_42obs_bp)])
    d.to_csv(OUT/'rolling_2m_changes.csv',index_label='date')
    nonoverlap.to_csv(OUT/'nonoverlapping_42obs_changes.csv',index_label='date')
    summary.to_csv(OUT/'distribution_summary.csv',index=False)
    # Independent alignment checks on the first, middle and final main observations.
    for i in [42,len(s)//2,len(s)-1]:
        assert np.isclose(d.change_42obs_bp.iloc[i],100*(s.iloc[i]-s.iloc[i-42]))
        assert d.start_date_42obs.iloc[i]==s.index[i-42]
    latest=d.iloc[-1]
    current=a.iloc[-1];percentile=a.le(current).mean()*100
    p05,p95=a.quantile([.05,.95])
    bins=np.linspace(np.floor(min(a.min(),b.min())/10)*10,np.ceil(max(a.max(),b.max())/10)*10,50)
    fig=go.Figure()
    for label,values,opacity,visible in [('42 observations',a,.7,True),('2 calendar months',b,.35,'legendonly'),
        ('42 observations — non-overlapping',nonoverlap.change_42obs_bp,.45,'legendonly')]:
        fig.add_trace(go.Histogram(x=values,name=label,histnorm='percent',opacity=opacity,
            xbins=dict(start=bins[0],end=bins[-1],size=bins[1]-bins[0]),visible=visible))
    fig.add_vline(x=current,line_color='#dc2626',line_width=2,annotation_text=f'Latest {current:+.1f} bp ({percentile:.1f}th percentile)')
    fig.add_vline(x=p05,line_dash='dash',line_color='#64748b',annotation_text='P5')
    fig.add_vline(x=p95,line_dash='dash',line_color='#64748b',annotation_text='P95')
    fig.update_layout(template='plotly_white',title='Distribution of 2-month 7Y Treasury yield changes since 2009',
        xaxis_title='Yield change (bp): negative = rally, positive = selloff',yaxis_title='Share of historical observations (%)',
        barmode='overlay',height=550,legend=dict(orientation='h',y=-.2))
    histogram=fig.to_html(full_html=False,include_plotlyjs=False,config=dict(responsive=True,displaylogo=False))
    ecdf=np.sort(a.to_numpy())
    fig=go.Figure(go.Scatter(x=ecdf,y=np.arange(1,len(ecdf)+1)/len(ecdf)*100,name='Historical cumulative share'))
    fig.add_trace(go.Scatter(x=[current],y=[percentile],mode='markers',name='Latest',marker=dict(color='#dc2626',size=11)))
    fig.update_layout(template='plotly_white',title='Empirical cumulative distribution — 42-observation changes',
        xaxis_title='Yield change (bp)',yaxis_title='Historical share at or below change (%)',height=470)
    cumulative=fig.to_html(full_html=False,include_plotlyjs=False)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=('7Y Treasury yield (%)','Rolling 42-observation change (bp)'))
    fig.add_trace(go.Scatter(x=s.index,y=s,name='7Y yield'),row=1,col=1)
    fig.add_trace(go.Scatter(x=a.index,y=a,name='42-observation change'),row=2,col=1)
    for y,color in [(0,'black'),(p05,'#64748b'),(p95,'#64748b')]:fig.add_hline(y=y,line_dash='dash',line_color=color,row=2,col=1)
    fig.update_layout(template='plotly_white',height=740,hovermode='x unified',title='Yield and 2-month changes through history')
    fig.update_xaxes(rangeslider_visible=True,row=2,col=1)
    history=fig.to_html(full_html=False,include_plotlyjs=False)
    extremes=d.dropna(subset=['change_42obs_bp'])[['start_date_42obs','end_date','start_yield_42obs_pct','end_yield_pct','change_42obs_bp']]
    extremes=pd.concat([extremes.nsmallest(10,'change_42obs_bp').assign(Direction='Largest rallies'),
        extremes.nlargest(10,'change_42obs_bp').assign(Direction='Largest selloffs')])
    extremes.to_csv(OUT/'largest_2m_moves.csv',index=False)
    table=summary.round(2).to_html(index=False,border=0)
    extreme_table=extremes.round(3).to_html(index=False,border=0)
    title='UST 7Y — Two-Month Yield-Change Distribution'
    page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#eef3f7;color:#183247;font:14px Arial,sans-serif}}main{{max-width:1450px;margin:auto;padding:20px}}section{{background:white;padding:22px;border-radius:9px;margin-bottom:18px}}h1{{font-size:25px}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px;width:100%}}th{{background:#173e56;color:white;padding:9px}}td{{padding:8px;border-bottom:1px solid #dbe3ea;text-align:right}}.table-wrap{{overflow:auto}}.note{{font-size:12px;color:#536b7d}}a{{color:#155b86}}</style><script>{get_plotlyjs()}</script></head><body><main>
<section><h1>{title}</h1><p>Bloomberg {TICKER} · PX_LAST · {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d} · {len(s):,} yield observations.</p>
<p><strong>Latest 42-observation change: {current:+.2f} bp</strong> — from {latest.start_yield_42obs_pct:.3f}% on {latest.start_date_42obs:%Y-%m-%d} to {latest.end_yield_pct:.3f}% on {s.index[-1]:%Y-%m-%d}; historical percentile {percentile:.1f}%.</p>
<p>The historical central 90% range is {p05:+.2f} to {p95:+.2f} bp (5th–95th percentiles).</p>
<p>Calendar-month check: {latest.change_calendar2m_bp:+.2f} bp, using {latest.calendar_start_date:%Y-%m-%d}, the latest observed yield on or before {latest.calendar_cutoff:%Y-%m-%d}.</p>
<p><a href="#distribution">Distribution</a> · <a href="#stats">Statistics</a> · <a href="#ecdf">Cumulative distribution</a> · <a href="#history">History</a> · <a href="#extremes">Largest moves</a></p>
<p class="note">Main definition: 2 months ≈ 42 available weekday Bloomberg yield observations. Change = 100 × (ending yield in % − starting yield in %). No yield filling or interpolation. Daily overlapping windows are dependent observations, not independent events or a forecast. A latest-anchored non-overlapping 42-observation check is included; its sampling phase is one possible choice. Exact calendar months use the most recent quote on or before the cutoff. Generic Treasury yields are benchmark series rather than a constant-security holding return. Latest Bloomberg observations can be intraday. Pulled {html.escape(pulled)}.</p></section>
<section id="distribution">{histogram}</section><section id="stats"><h2>Distribution statistics</h2><div class="table-wrap">{table}</div></section>
<section id="ecdf">{cumulative}</section><section id="history">{history}</section><section id="extremes"><h2>Largest overlapping 2-month moves</h2><p class="note">Neighbouring endpoints can belong to the same market episode.</p><div class="table-wrap">{extreme_table}</div></section></main></body></html>'''
    report=OUT/'UST7Y_TwoMonth_Change_Analysis.html';report.write_text(page,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axes[0].hist(a,bins=bins,weights=np.ones(len(a))*100/len(a),color='#1d638d',edgecolor='white',linewidth=.4)
    axes[0].axvline(current,color='#dc2626',linewidth=2,label=f'Latest {current:+.1f} bp')
    for q in [p05,p95]:axes[0].axvline(q,color='#64748b',linestyle='--')
    axes[0].set(xlabel='42-observation yield change (bp)',ylabel='Historical observations (%)',title='Two-month change distribution')
    axes[0].legend(frameon=False)
    axes[1].plot(ecdf,np.arange(1,len(ecdf)+1)/len(ecdf)*100,color='#1d638d')
    axes[1].scatter([current],[percentile],color='#dc2626',zorder=3,label=f'Latest percentile: {percentile:.1f}%')
    axes[1].set(xlabel='Yield change (bp)',ylabel='Cumulative share (%)',title='Empirical cumulative distribution')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    figure.suptitle(f'UST 7Y yield: 2-month changes | {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d}',fontsize=14)
    figure.savefig(OUT/'UST7Y_TwoMonth_Change_Distribution.png',dpi=170)
    plt.close(figure)
    print(summary.to_string(index=False));print('REPORT',report)
    print('LATEST',latest[['start_date_42obs','start_yield_42obs_pct','end_yield_pct','change_42obs_bp','calendar_start_date','change_calendar2m_bp']].to_string())
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache',action='store_true')
    main(parser.parse_args().from_cache)
