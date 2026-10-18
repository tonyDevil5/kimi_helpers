"""Notification-day TBA drop and holding monitor for FN/G2 5.5, 6 and 6.5."""
from pathlib import Path
import argparse,html,json
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from plotly.offline import get_plotlyjs
import mbs_rv_report as mbs

OUT=Path(__file__).parent/'tba_drop_realization_output'
CALENDAR=Path(mbs.FNAME).parent/'tba_settle_dates.xlsx'
COUPONS=[f'{agency} {coupon:g}' for agency in ['FN','G2'] for coupon in [5.5,6,6.5]]
BANK_DAY=CustomBusinessDay(calendar=USFederalHolidayCalendar())
SIFMA_URL='https://www.sifma.org/resources/general/mbs-notification-and-settlement-dates'
FED_URL='https://www.newyorkfed.org/markets/ambs-treasury-faq-05012024'


def load_calendar(path):
    frame=pd.read_excel(path)
    names={str(c).replace(' ','').lower():c for c in frame}
    rows=[]
    for agency,kind in [('FN','a'),('G2','c')]:
        if 'class'+kind not in names:raise ValueError(f'Missing Class {kind.upper()} calendar')
        dates=pd.to_datetime(frame[names['class'+kind]],errors='coerce').dropna().sort_values()
        if dates.duplicated().any() or dates.dt.to_period('M').duplicated().any():raise ValueError('Duplicate settlement month')
        if len(dates)>1 and not np.all(np.diff(dates.dt.to_period('M').astype('int64'))==1):raise ValueError('Missing settlement month in calendar')
        for i,settle in enumerate(dates):
            if settle.weekday()>4 or not BANK_DAY.is_on_offset(settle):raise ValueError(f'Non-bank settlement date {settle}')
            rows.append(dict(agency=agency,settlement_month=str(settle.to_period('M')),
                front_settlement=settle,back_settlement=dates.iloc[i+1] if i+1<len(dates) else pd.NaT,
                notification_date=settle-2*BANK_DAY))
    return pd.DataFrame(rows).sort_values(['agency','front_settlement']).reset_index(drop=True)


def load_quotes(path):
    d=pd.read_excel(path,sheet_name='Data',index_col=0)
    d.index=pd.to_datetime(d.index,errors='coerce')
    d=d.loc[~d.index.isna()].sort_index()
    if d.index.has_duplicates:raise ValueError('Duplicate market-data dates')
    fields=[f'{c} {field}' for c in COUPONS for field in ['Price','Drop','Perf']]
    missing=set(fields)-set(d)
    if missing:raise ValueError(f'Missing market data: {sorted(missing)}')
    d=d[fields].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return d.loc[d.notna().any(axis=1)]


def build_coupon(quotes,calendar,coupon,start,notional):
    """Trace one delivery contract through each notification-to-notification cycle."""
    if notional<=0:raise ValueError('Notional must be positive')
    agency=coupon.split()[0];tick_usd=notional/3200
    cal=calendar[calendar.agency==agency].sort_values('front_settlement')
    q=quotes.loc[quotes.index>=pd.Timestamp(start),[f'{coupon} {f}' for f in ['Price','Drop','Perf']]].copy()
    q.columns=['front_price_points','quoted_drop_ticks','jpm_perf_ticks']
    eligible=q.front_price_points.dropna()
    if eligible.empty:raise ValueError(f'No holding-start price for {coupon}')
    q=q.loc[eligible.index[0]:].copy();first=q.index[0];asof=q.index[-1]
    relevant=cal[(cal.notification_date>=first)&(cal.notification_date<=asof)]
    q=q.reindex(q.index.union(pd.DatetimeIndex(relevant.notification_date)).sort_values())
    q.index.name='date'
    dates=cal.front_settlement.tolist()
    def front_index(date):
        candidates=np.flatnonzero(np.asarray([x>=date for x in dates]))
        if not len(candidates):raise ValueError('Settlement calendar needs extension')
        return int(candidates[0])
    held_settle=dates[front_index(first)]
    entry_date=first;entry_front=q.front_price_points.iloc[0];entry_drop=0.;cost=entry_front
    realized=0.;quoted=0.;missing=0;cycles=[];rolls=[]
    records=[]
    event_map={e.notification_date:e for e in relevant.itertuples()}
    for date,row in q.iterrows():
        front_i=front_index(date);front_settle=dates[front_i]
        price=row.front_price_points;drop=row.quoted_drop_ticks
        realized_today=0.;quote_today=0.
        if date in event_map:
            e=event_map[date]
            if held_settle!=e.front_settlement:
                raise ValueError(f'Held delivery does not match outgoing front on {date}')
            pnl=(price-cost)*32
            price_component=(price-entry_front)*32
            cycles.append(dict(coupon=coupon,delivery_month=str(held_settle.to_period('M')),
                entry_date=entry_date,exit_or_mark_date=date,entry_front_price_points=entry_front,
                entry_drop_ticks=entry_drop,purchase_price_points=cost,sale_or_mark_price_points=price,
                price_component_ticks=price_component,holding_pnl_ticks=pnl,holding_pnl_usd=pnl*tick_usd,
                jpm_hedged_perf_ticks=q.loc[(q.index>entry_date)&(q.index<=date),'jpm_perf_ticks'].sum(skipna=False),
                status='Closed initial front holding' if entry_date==first and entry_drop==0 else 'Closed cycle'))
            realized_today=pnl;realized+=pnl
            back_price=price-drop/32
            valid=pd.notna(price) and pd.notna(drop) and pd.notna(e.back_settlement)
            if not valid:missing+=1
            quoted+=drop;quote_today=drop
            rolls.append(dict(coupon=coupon,roll_date=date,front_settlement=e.front_settlement,
                back_settlement=e.back_settlement,front_price_points=price,drop_ticks=drop,
                back_purchase_price_points=back_price,quoted_drop_usd=drop*tick_usd,
                both_settlement_dates_passed=pd.notna(e.back_settlement) and e.back_settlement<=asof,
                status='Modeled trade at workbook quote' if valid else 'Missing roll quote/calendar'))
            held_settle=e.back_settlement;entry_date=date;entry_front=price;entry_drop=drop;cost=back_price
        if held_settle==front_settle:
            mark=price;mark_kind='Held contract is quoted front'
        elif front_i+1<len(dates) and held_settle==dates[front_i+1]:
            mark=price-drop/32;mark_kind='Held contract is back: front minus current drop'
        else:
            mark=np.nan;mark_kind='Contract mapping unavailable'
        unrealized=(mark-cost)*32
        total=realized+unrealized
        # This independent decomposition includes the current held-back adjustment.
        adjustment=(price-mark)*32
        decomposition=(price-q.front_price_points.iloc[0])*32+quoted-adjustment
        if np.isfinite(total) and np.isfinite(decomposition):
            np.testing.assert_allclose(total,decomposition,atol=1e-8)
        records.append(dict(date=date,held_delivery_month=str(held_settle.to_period('M')) if pd.notna(held_settle) else 'Unknown',
            quoted_front_delivery_month=str(front_settle.to_period('M')),held_contract_mark_points=mark,
            mark_method=mark_kind,open_purchase_price_points=cost,roll_quote_ticks=quote_today,
            cumulative_quoted_drop_ticks=quoted,realized_today_ticks=realized_today,
            cumulative_realized_pnl_ticks=realized,open_unrealized_pnl_ticks=unrealized,
            total_holding_pnl_ticks=total,current_front_to_held_adjustment_ticks=adjustment))
    computed=pd.DataFrame(records).set_index('date');q=q.join(computed)
    perf=q.jpm_perf_ticks.copy();perf.iloc[0]=0
    q['cumulative_jpm_perf_ticks']=perf.cumsum(skipna=False)
    for field in ['cumulative_quoted_drop','realized_today','cumulative_realized_pnl','open_unrealized_pnl','total_holding_pnl','cumulative_jpm_perf']:
        q[field+'_usd']=q[field+'_ticks']*tick_usd
    last=q.iloc[-1]
    cycles.append(dict(coupon=coupon,delivery_month=last.held_delivery_month,entry_date=entry_date,
        exit_or_mark_date=asof,entry_front_price_points=entry_front,entry_drop_ticks=entry_drop,
        purchase_price_points=cost,sale_or_mark_price_points=last.held_contract_mark_points,
        price_component_ticks=(last.held_contract_mark_points-entry_front)*32,
        holding_pnl_ticks=last.open_unrealized_pnl_ticks,holding_pnl_usd=last.open_unrealized_pnl_usd,
        jpm_hedged_perf_ticks=perf.loc[(perf.index>entry_date)&(perf.index<=asof)].sum(skipna=False),
        status='Open: '+last.mark_method))
    events=pd.DataFrame(cycles)
    roll_frame=pd.DataFrame(rolls,columns=['coupon','roll_date','front_settlement','back_settlement',
        'front_price_points','drop_ticks','back_purchase_price_points','quoted_drop_usd',
        'both_settlement_dates_passed','status'])
    q.attrs['roll_trades']=roll_frame
    if np.isfinite(last.total_holding_pnl_usd):
        np.testing.assert_allclose(events.holding_pnl_usd.sum(skipna=False),last.total_holding_pnl_usd,atol=1e-6)
    future=cal[cal.notification_date>asof]
    summary=dict(coupon=coupon,start=str(first.date()),as_of=str(asof.date()),rolls=len(rolls),
        closed_holding_pnl_usd=last.cumulative_realized_pnl_usd,
        open_holding_pnl_usd=last.open_unrealized_pnl_usd,total_holding_pnl_usd=last.total_holding_pnl_usd,
        quoted_drop_total_ticks=last.cumulative_quoted_drop_ticks,
        quoted_drop_total_usd=last.cumulative_quoted_drop_usd,
        held_delivery_month=last.held_delivery_month,held_contract_mark_points=last.held_contract_mark_points,
        jpm_hedged_perf_usd=last.cumulative_jpm_perf_usd,latest_drop_ticks=last.quoted_drop_ticks,
        next_notification=str(future.notification_date.iloc[0].date()) if len(future) else 'Calendar extension needed',
        missing_roll_quotes=missing)
    year_rows=[];prior_total=0.;prior_unrealized=0.
    for year,part in q.groupby(q.index.year):
        end=part.iloc[-1]
        closed=part.realized_today_usd.sum(skipna=False)
        change_unrealized=end.open_unrealized_pnl_usd-prior_unrealized
        total=end.total_holding_pnl_usd-prior_total
        if np.isfinite(total):np.testing.assert_allclose(closed+change_unrealized,total,atol=1e-6)
        year_rows.append(dict(coupon=coupon,year=int(year),closed_holding_pnl_usd=closed,
            change_in_open_pnl_usd=change_unrealized,total_holding_pnl_usd=total,
            quoted_drop_total_usd=part.roll_quote_ticks.sum(skipna=False)*tick_usd,
            jpm_hedged_perf_usd=perf.loc[part.index].sum(skipna=False)*tick_usd))
        prior_total=end.total_holding_pnl_usd;prior_unrealized=end.open_unrealized_pnl_usd
    return q,events,summary,pd.DataFrame(year_rows)


def table(d):
    labels={'coupon':'Coupon','year':'Year','closed_holding_pnl_usd':'Closed-cycle P&L ($)',
      'open_holding_pnl_usd':'Open contract P&L ($)','total_holding_pnl_usd':'Total holding P&L ($)',
      'quoted_drop_total_usd':'Sum of roll quotes ($, not profit)',
      'quoted_drop_total_ticks':'Sum of roll quotes (ticks, not profit)',
      'jpm_hedged_perf_usd':'JPM hedged performance ($)','latest_drop_ticks':'Latest drop (ticks)',
      'entry_drop_ticks':'ENTRY-date drop (ticks)','drop_ticks':'New roll quote (ticks)',
      'purchase_price_points':'Held-contract purchase price','sale_or_mark_price_points':'Same-contract sale/mark price',
      'holding_pnl_ticks':'Holding P&L (ticks)','holding_pnl_usd':'Holding P&L ($)',
      'delivery_month':'Held delivery month','entry_date':'Purchase date','exit_or_mark_date':'Sale / mark date',
      'price_component_ticks':'Sale/mark minus entry FRONT (ticks)',
      'held_delivery_month':'Current held delivery','next_notification':'Next notification',
      'change_in_open_pnl_usd':'Change in open P&L ($)','missing_roll_quotes':'Missing roll quotes'}
    return d.rename(columns=labels).to_html(index=False,border=0,na_rep='—',float_format=lambda v:f'{v:,.2f}')


def report(summary,yearly,results,calendar,notional,source,calendar_source):
    payload={};views=[]
    for coupon,(q,cycles) in results.items():
        rolls=q.attrs['roll_trades']
        serial=lambda s:[float(v) if pd.notna(v) and np.isfinite(v) else None for v in s]
        payload[coupon]=dict(dates=q.index.strftime('%Y-%m-%d').tolist(),drop=serial(q.quoted_drop_ticks),
            quoted=serial(q.cumulative_quoted_drop_usd),closed=serial(q.cumulative_realized_pnl_usd),
            open=serial(q.open_unrealized_pnl_usd),total=serial(q.total_holding_pnl_usd),
            front=serial(q.front_price_points),held=serial(q.held_contract_mark_points),
            held_month=q.held_delivery_month.tolist(),jpm=serial(q.cumulative_jpm_perf_usd),
            roll_dates=pd.to_datetime(rolls.roll_date).dt.strftime('%Y-%m-%d').tolist(),roll_drops=serial(rolls.drop_ticks))
        def display(frame):
            d=frame.copy()
            for c in d:
                if pd.api.types.is_datetime64_any_dtype(d[c]):d[c]=d[c].dt.strftime('%Y-%m-%d').fillna('—')
            return d
        cols=['delivery_month','entry_date','exit_or_mark_date','purchase_price_points',
              'sale_or_mark_price_points','entry_drop_ticks','holding_pnl_ticks','holding_pnl_usd','status']
        views.append(f'<div class="detail" data-coupon="{coupon}"><h3>{coupon}: yearly holding P&L</h3>'+table(yearly[yearly.coupon==coupon].drop(columns='coupon'))+
            '<h3>Same-contract holding cycles</h3>'+table(display(cycles)[cols])+
            '<details><summary>Cycle decomposition: entry drop and market price move</summary>'+table(display(cycles))+'</details>'+
            '<details><summary>Roll executions assumed at quoted front/back prices</summary>'+table(display(rolls))+'</details></div>')
    options=''.join(f'<option>{c}</option>' for c in payload)
    data=json.dumps(payload,allow_nan=False).replace('</','<\\/')
    compact=['coupon','rolls','closed_holding_pnl_usd','open_holding_pnl_usd','total_holding_pnl_usd','jpm_hedged_perf_usd','missing_roll_quotes']
    body='''<!doctype html><html><head><meta charset="utf-8"><title>TBA holding and drop monitor</title><style>
body{font:14px Arial;background:#eef2f6;color:#17202a;margin:20px}main{max-width:1550px;margin:auto}section{background:white;padding:22px;margin:16px 0;border-radius:8px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{background:#173e56;color:white;padding:9px;cursor:pointer}td{padding:8px;border:1px solid #ddd;text-align:right}tr:nth-child(even){background:#f3f6f9}select{padding:9px}summary{cursor:pointer;font-weight:bold}details{margin:14px 0}.detail{overflow:auto}</style></head><body><main>'''
    body+=f'''<section><h1>TBA holding and dollar-roll monitor — same-contract accounting</h1>
<p>FN/G2 5.5/6/6.5 held from {summary.start.min()} through {summary.as_of.max()}; ${notional/1e6:g}mm face per coupon, ${notional/3200:,.0f}/tick. Class A settlement for FN, Class C for G2. Start by buying the quoted front contract, then sell front and buy back on each notification day.</p>
<p><b>Contract mapping uses your assumption:</b> Excel Price refers to the front delivery month through its settlement date (inclusive), then switches to the next delivery month. Drop is front minus back for that same pair. On a roll date, buy back at front price − drop/32. While that held contract is still the back month, mark it at current front price − current drop/32. After the previous front settles, mark the held contract directly at Excel Price.</p>
<p><b>Each completed cycle compares the same delivery contract:</b> sale price at the next notification minus purchase price at the prior notification. Equivalently: front-to-front price change plus the <b>ENTRY-date drop</b>. The drop on the sale date belongs to the next purchase, not the cycle just closed. Initial front holding is a separate partial cycle with no entry drop.</p>
<p><b>Rolling itself creates no immediate profit.</b> On the roll day the new contract is marked at its inferred purchase price. Total holding P&amp;L = cumulative closed-cycle P&amp;L + open-contract marked P&amp;L. The open contract can remain the back month until settlement. The sum of monthly drop quotes is retained only as reference and is not labeled realized income.</p>
<p>These are gross modeled execution/mark prices, not broker fills. Funding, coupon income, accrued interest, amortization, reinvestment, bid/ask and transaction costs are excluded. The front-through-settlement convention is assumed, not independently verified by a price jump. JPM hedged performance is shown separately; drops are never added to it.</p>
<p>Notification = two US bank business days before the supplied settlement, with a US federal holiday calendar. Exact-date roll quotes are required; missing quotes are exposed, not filled or shifted. Yearly P&amp;L is closed-cycle P&amp;L booked that year plus the change in open marked P&amp;L. The last year is YTD.</p>
<p>Inputs: {html.escape(str(source))}, Data Price/Drop/Perf; {html.escape(str(calendar_source))}, ClassA/ClassC. Prices are points; Drop/Perf are ticks. Method/calendar references: <a href="{SIFMA_URL}">SIFMA calendar</a>; <a href="{FED_URL}">New York Fed dollar-roll definition</a>.</p></section>
<section><h2>Continuous holding from the start of 2025</h2>{table(summary[compact])}<details><summary>Latest held contracts, upcoming roll and quote totals</summary>{table(summary)}</details></section>
<section><label>Coupon <select id="coupon">{options}</select></label><div id="holding-chart"></div><p>Holding P&amp;L from tracking the same delivery contract; total equals closed cycles plus the open marked position.</p><div id="contract-chart"></div><p>The held contract is marked from Price − Drop/32 while it is back month, then directly from Price once it becomes front. Its mark resets to the new purchase price at a roll; this contract-price reset is not a P&amp;L jump.</p><div id="drop-chart"></div><p>Daily front-minus-back drop and quotes used for purchases. A quoted drop is not automatically holding profit.</p><div id="jpm-chart"></div><p>Separate JPM Treasury-hedged performance, without adding drops.</p>{''.join(views)}</section>
<section><details><summary>Settlement and derived notification calendar</summary>{table(calendar.astype({'front_settlement':str,'back_settlement':str,'notification_date':str}))}</details></section>'''
    body+='<script>'+get_plotlyjs()+'</script><script type="application/json" id="payload">'+data+'</script>'
    body+='''<script>const data=JSON.parse(document.getElementById('payload').textContent);
function render(){const c=document.getElementById('coupon').value,d=data[c];
Plotly.react('holding-chart',[{x:d.dates,y:d.total,name:'Total holding P&L'},{x:d.dates,y:d.closed,name:'Closed-cycle P&L',line:{shape:'hv'}},{x:d.dates,y:d.open,name:'Open-contract P&L'}],{title:c+' — same-contract holding P&L',yaxis:{title:'USD'},height:430,hovermode:'x unified',legend:{orientation:'h'}});
Plotly.react('contract-chart',[{x:d.dates,y:d.front,name:'Excel generic FRONT price'},{x:d.dates,y:d.held,name:'HELD delivery-contract mark',text:d.held_month,hovertemplate:'%{x}<br>Held delivery: %{text}<br>Mark: %{y:.6f}<extra></extra>'}],{title:c+' — quoted front versus held contract',yaxis:{title:'Price points'},height:350,hovermode:'x unified'});
Plotly.react('drop-chart',[{x:d.dates,y:d.drop,name:'Daily front/back drop'},{x:d.roll_dates,y:d.roll_drops,name:'ENTRY-date drop used',mode:'markers',marker:{size:9}}],{title:c+' — quoted drop (not realized income)',yaxis:{title:'Ticks (32nds)'},height:350});
Plotly.react('jpm-chart',[{x:d.dates,y:d.jpm,name:'JPM hedged performance'}],{title:c+' — separate JPM hedged performance',yaxis:{title:'USD'},height:350});
document.querySelectorAll('.detail').forEach(e=>e.style.display=e.dataset.coupon===c?'':'none');}
document.getElementById('coupon').onchange=render;render();document.querySelectorAll('table').forEach(t=>t.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{const asc=h.dataset.asc!=='true';h.dataset.asc=asc;[...t.tBodies[0].rows].sort((a,b)=>{let x=a.cells[i].textContent.replace(/,/g,''),y=b.cells[i].textContent.replace(/,/g,'');if(x==='—'||y==='—')return x===y?0:x==='—'?1:-1;let v=Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y);return asc?v:-v}).forEach(row=>t.tBodies[0].appendChild(row))}));</script></main></body></html>'''
    path=OUT/'tba_drop_realization_monitor.html';path.write_text(body,encoding='utf-8');return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workbook',type=Path,default=Path(mbs.FNAME));p.add_argument('--calendar',type=Path,default=CALENDAR);p.add_argument('--start',default='2025-01-01');p.add_argument('--notional',type=float,default=10_000_000);a=p.parse_args()
    if a.notional<=0:p.error('Notional must be positive')
    OUT.mkdir(exist_ok=True);calendar=load_calendar(a.calendar);quotes=load_quotes(a.workbook)
    if calendar.front_settlement.max()<quotes.index.max():raise ValueError('Settlement calendar needs extension through latest quotes')
    summary=[];years=[];events=[];results={}
    for coupon in COUPONS:
        q,e,s,y=build_coupon(quotes,calendar,coupon,a.start,a.notional);results[coupon]=(q,e);summary.append(s);years.append(y);events.append(e)
        q.to_csv(OUT/(coupon.replace(' ','_')+'_daily.csv'),index_label='date')
        q.attrs['roll_trades'].to_csv(OUT/(coupon.replace(' ','_')+'_roll_trades.csv'),index=False)
    summary=pd.DataFrame(summary);yearly=pd.concat(years,ignore_index=True)
    quotes.to_csv(OUT/'source_quotes.csv',index_label='date');calendar.to_csv(OUT/'roll_calendar.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False);yearly.to_csv(OUT/'yearly.csv',index=False);pd.concat(events,ignore_index=True).to_csv(OUT/'roll_events.csv',index=False)
    path=report(summary,yearly,results,calendar,a.notional,a.workbook,a.calendar)
    print(summary.to_string(index=False));print('REPORT',path)

if __name__=='__main__':main()
