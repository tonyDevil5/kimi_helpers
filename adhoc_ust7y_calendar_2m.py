"""Non-overlapping 7Y yield moves: first quote of month to first quote two months later."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from adhoc_ust7y_2m_change import OUT, TICKER, describe


def month_starts(yields):
    s=yields.dropna().sort_index()
    if s.index.has_duplicates:raise ValueError('Duplicate yield dates')
    first=s.groupby(s.index.to_period('M')).apply(lambda group:pd.Series(
        {'date':group.index[0],'yield_pct':group.iloc[0]})).unstack()
    # Incomplete initial history (e.g. Bloomberg 7Y begins Feb 26, 2009)
    # must not be treated as a beginning-of-month quote. Apply to gaps too.
    return first.loc[pd.to_datetime(first.date).dt.day<=7].copy()


def build_periods(yields, phase=1):
    """phase 1: Jan→Mar, Mar→May ...; phase 0: Feb→Apr, Apr→Jun ..."""
    if phase not in [0,1]:raise ValueError('Phase must be 0 or 1')
    starts=month_starts(yields)
    rows=[]
    for month,row in starts.iterrows():
        if month.month%2!=phase:continue
        target=month+2
        if target not in starts.index:continue
        end=starts.loc[target]
        rows.append(dict(start_month=str(month),end_month=str(target),start_date=row.date,end_date=end.date,
            start_yield_pct=float(row.yield_pct),end_yield_pct=float(end.yield_pct),
            change_bp=100*(float(end.yield_pct)-float(row.yield_pct))))
    result=pd.DataFrame(rows)
    if len(result)>1:
        assert (pd.to_datetime(result.end_date.iloc[:-1]).to_numpy()<=pd.to_datetime(result.start_date.iloc[1:]).to_numpy()).all()
    return result


def main():
    s=pd.read_csv(OUT/'gt07_bloomberg_yields.csv',index_col='date',parse_dates=True).yield_pct
    metadata=json.loads((OUT/'source_metadata.json').read_text(encoding='utf-8'))
    primary=build_periods(s,1);alternate=build_periods(s,0)
    for data,name in [(primary,'jan_mar'),(alternate,'feb_apr')]:
        data.to_csv(OUT/f'calendar_nonoverlap_{name}_periods.csv',index=False)
    summaries=[]
    for label,data in [('Jan-Mar / Mar-May / ... (main)',primary),('Feb-Apr / Apr-Jun / ... (alternative)',alternate)]:
        row=describe(label,data.change_bp)
        row['Latest start']=str(pd.Timestamp(data.iloc[-1].start_date).date())
        row['Latest end']=str(pd.Timestamp(data.iloc[-1].end_date).date())
        summaries.append(row)
    summary=pd.DataFrame(summaries).rename(columns={'Observations':'Completed periods','Latest (bp)':'Latest completed (bp)'})
    summary.to_csv(OUT/'calendar_nonoverlap_summary.csv',index=False)
    moves=primary.change_bp
    current=moves.iloc[-1];pct=moves.le(current).mean()*100
    p5,p95=moves.quantile([.05,.95])
    bins=np.arange(np.floor(min(moves.min(),alternate.change_bp.min())/10)*10,
        np.ceil(max(moves.max(),alternate.change_bp.max())/10)*10+10,10)
    fig=go.Figure()
    for title,data,visible in [('Jan-start schedule',primary,True),('Feb-start schedule',alternate,'legendonly')]:
        fig.add_trace(go.Histogram(x=data.change_bp,histnorm='percent',name=title,opacity=.65,visible=visible,
            xbins=dict(start=bins[0],end=bins[-1],size=10)))
    fig.add_vline(x=current,line_color='#dc2626',annotation_text=f'Latest completed {current:+.1f} bp')
    for value,label in [(p5,'P5'),(p95,'P95')]:fig.add_vline(x=value,line_dash='dash',line_color='#64748b',annotation_text=label)
    fig.update_layout(template='plotly_white',height=550,barmode='overlay',
        title='Non-overlapping beginning-of-month 2-month yield changes',
        xaxis_title='7Y yield change (bp): negative = rally; positive = selloff',yaxis_title='Historical periods (%)')
    histogram=fig.to_html(full_html=False,include_plotlyjs=False,config=dict(responsive=True,displaylogo=False))
    fig=go.Figure(go.Bar(x=pd.to_datetime(primary.end_date),y=moves,
        customdata=primary[['start_date','end_date']].astype(str).to_numpy(),
        marker_color=np.where(moves>=0,'#c36c27','#216a97'),
        hovertemplate='%{customdata[0]} → %{customdata[1]}<br>%{y:+.2f} bp<extra></extra>'))
    fig.update_layout(template='plotly_white',height=500,title='Completed non-overlapping periods — Jan-start schedule',yaxis_title='Yield change (bp)')
    history=fig.to_html(full_html=False,include_plotlyjs=False)
    first,last=primary.iloc[0],primary.iloc[-1]
    starts=month_starts(s)
    unfinished=[]
    for month,row in starts.iterrows():
        if month.month%2==1 and month+2>s.index[-1].to_period('M'):
            unfinished.append(f'{pd.Timestamp(row.date):%Y-%m-%d} → {month+2} first quote pending; '
                f'move through {s.index[-1]:%Y-%m-%d}: {(s.iloc[-1]-float(row.yield_pct))*100:+.2f} bp. '
                'This incomplete interval is excluded from the distribution.')
    format_table=lambda frame:'<div class="table-wrap">'+frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.2f}')+'</div>'
    page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UST 7Y — non-overlapping calendar two-month changes</title><style>
body{{background:#eef3f7;color:#183247;font:14px Arial,sans-serif;margin:0}}main{{max-width:1450px;margin:auto;padding:20px}}section{{background:white;padding:22px;margin-bottom:18px;border-radius:9px}}h1{{font-size:25px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}th{{background:#173e56;color:white;padding:9px}}td{{border-bottom:1px solid #dbe3ea;padding:8px;text-align:right}}.note{{color:#546b7c;font-size:12px}}
</style><script>{get_plotlyjs()}</script></head><body><main><section><h1>UST 7Y — Non-overlapping calendar two-month changes</h1>
<p>Bloomberg {TICKER} PX_LAST; cached data through {s.index[-1]:%Y-%m-%d}. First available source date {s.index[0]:%Y-%m-%d}.</p>
<p><strong>Main schedule: Jan→Mar, Mar→May, May→Jul, Jul→Sep, Sep→Nov, Nov→Jan.</strong> Each endpoint is the first available quoted weekday of that month. Change = (end yield % − start yield %) × 100. Consecutive intervals share a boundary quote, with no overlapping holding period.</p>
<p><strong>{len(primary)} completed periods</strong>, from {pd.Timestamp(first.start_date):%Y-%m-%d}→{pd.Timestamp(first.end_date):%Y-%m-%d} through {pd.Timestamp(last.start_date):%Y-%m-%d}→{pd.Timestamp(last.end_date):%Y-%m-%d}.</p>
<p>Latest completed: {last.start_yield_pct:.3f}% → {last.end_yield_pct:.3f}% = <strong>{current:+.2f} bp</strong> ({pct:.1f}th historical percentile). Historical middle 90%: {p5:+.2f} to {p95:+.2f} bp.</p>
<p class="note">{html.escape(' '.join(unfinished))}</p>
<p class="note">February 2009 begins only on February 26 in this source, so it cannot supply a month-start quote. Month quotes first appearing after day 7 are excluded as incomplete boundaries. No filling or interpolation. The February-start schedule is a separate sensitivity check; the two schedules overlap with each other and are never pooled into one non-overlapping sample. Removing overlap does not guarantee statistical independence. Bloomberg download: {html.escape(metadata['pulled_china_time'])}.</p>
</section><section>{histogram}</section><section><h2>Statistics — schedules kept separate</h2>{format_table(summary)}</section>
<section>{history}</section><section><h2>All completed periods — main schedule</h2>{format_table(primary)}</section>
<section><details><summary>All completed periods — February-start schedule</summary>{format_table(alternate)}</details></section></main></body></html>'''
    report=OUT/'UST7Y_Nonoverlapping_Calendar_2Month.html';report.write_text(page,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    ax.hist(moves,bins=bins,weights=np.ones(len(moves))*100/len(moves),color='#216a97',edgecolor='white')
    ax.axvline(current,color='#dc2626',linewidth=2,label=f'Latest completed {current:+.1f} bp')
    for value,label in [(p5,'5th percentile'),(p95,'95th percentile')]:ax.axvline(value,color='#64748b',linestyle='--',label=f'{label}: {value:+.1f} bp')
    ax.set(title=f'UST 7Y: non-overlapping calendar 2-month changes\nJan–Mar / Mar–May / … | {len(primary)} completed periods',
        xlabel='Yield change (bp)',ylabel='Historical periods (%)')
    ax.legend(frameon=False);ax.grid(alpha=.2);ax.set_axisbelow(True)
    figure.savefig(OUT/'UST7Y_Nonoverlapping_Calendar_2Month.png',dpi=170);plt.close(figure)
    print(summary.to_string(index=False));print('LATEST MAIN',last.to_string());print('REPORT',report)


if __name__=='__main__':main()
