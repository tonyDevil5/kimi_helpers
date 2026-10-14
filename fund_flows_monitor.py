"""Bloomberg weekly flow monitor for the Active/Passive workbook universe.

Run with the Bloomberg-enabled mercury_env Python. --update refreshes source data.
Sparse/monthly flow reporters are kept visible but excluded from weekly aggregates.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import html
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

WORKBOOK = Path(r'C:\Users\tonyy\Nutstore\1\python\notebook\data\mbs\funds_info.xlsx')
OUT = Path(__file__).parent / 'fund_flows_output'
REF_FIELDS = ['NAME', 'NAV_CRNCY', 'FUND_FLOW_DATE', 'FUND_CLASS_ASSETS', 'ID_BB_GLOBAL']


def read_universe(path):
    sheets = pd.read_excel(path, sheet_name=['Active', 'Passive'])
    rows = []
    for group, d in sheets.items():
        for _, r in d.dropna(subset=['Ticker']).iterrows():
            rows.append(dict(ticker=str(r['Ticker']).strip(), group=group,
                             name=str(r['Fund Name']), fund_type=str(r['Fund Type']),
                             workbook_aum=str(r['AUM (USD)'])))
    u = pd.DataFrame(rows)
    if u.empty or u.ticker.duplicated().any():
        raise ValueError('Empty or duplicate workbook ticker universe')
    return u


class Bloomberg:
    """Bounded Desktop API requests; preserve security and field errors."""
    def __enter__(self):
        import blpapi
        self.api = blpapi
        self.session = blpapi.Session()
        if not self.session.start() or not self.session.openService('//blp/refdata'):
            raise RuntimeError('Bloomberg Desktop API is unavailable')
        self.service = self.session.getService('//blp/refdata')
        self.errors = []
        return self

    def __exit__(self, *args):
        self.session.stop()

    def request(self, tickers, fields, start=None, end=None):
        historical = start is not None
        req = self.service.createRequest('HistoricalDataRequest' if historical else 'ReferenceDataRequest')
        for t in tickers: req.getElement('securities').appendValue(t)
        for f in fields: req.getElement('fields').appendValue(f)
        if historical:
            req.set('startDate', pd.Timestamp(start).strftime('%Y%m%d'))
            req.set('endDate', pd.Timestamp(end).strftime('%Y%m%d'))
            req.set('periodicitySelection', 'DAILY')
            req.set('nonTradingDayFillOption', 'ACTIVE_DAYS_ONLY')
        cid = self.session.sendRequest(req)
        deadline = time.monotonic() + 90
        rows = []
        while time.monotonic() < deadline:
            event = self.session.nextEvent(1000)
            for msg in event:
                if cid not in msg.correlationIds(): continue
                if msg.hasElement('responseError'):
                    raise RuntimeError(str(msg.getElement('responseError')))
                if not msg.hasElement('securityData'): continue
                sd = msg.getElement('securityData')
                securities = [sd] if historical else [sd.getValueAsElement(i) for i in range(sd.numValues())]
                for sec in securities:
                    ticker = sec.getElementAsString('security')
                    for error in ['securityError', 'fieldExceptions']:
                        if sec.hasElement(error) and str(sec.getElement(error)).strip() not in ('fieldExceptions[] = {\n}',):
                            element = sec.getElement(error)
                            if error == 'securityError' or element.numValues():
                                self.errors.append(dict(ticker=ticker, error=str(element)))
                    if not sec.hasElement('fieldData'): continue
                    fd = sec.getElement('fieldData')
                    values = [fd.getValueAsElement(i) for i in range(fd.numValues())] if historical else [fd]
                    for val in values:
                        row = dict(ticker=ticker)
                        for i in range(val.numElements()):
                            el = val.getElement(i)
                            if el.isNull(): continue
                            value = el.getValue()
                            row[str(el.name())] = value.isoformat() if hasattr(value, 'isoformat') else value
                        rows.append(row)
            if event.eventType() == self.api.Event.RESPONSE: return pd.DataFrame(rows)
        self.session.cancel(cid)
        raise TimeoutError('Bloomberg fund-flow request exceeded 90 seconds')


def fetch(universe, start, end):
    refs, histories = [], []
    with Bloomberg() as bbg:
        for offset in range(0, len(universe), 10):
            tickers = universe.ticker.iloc[offset:offset+10].tolist()
            refs.append(bbg.request(tickers, REF_FIELDS))
            histories.append(bbg.request(tickers, ['FUND_FLOW'], start, end))
            print(f'Fetched {min(offset+10,len(universe))}/{len(universe)} funds', flush=True)
        ref = pd.concat(refs, ignore_index=True)
        raw = pd.concat(histories, ignore_index=True)
        currencies = sorted(set(ref.NAV_CRNCY.dropna()) - {'USD'})
        fx = []
        for ccy in currencies:
            rates = bbg.request([f'{ccy}USD Curncy'], ['PX_LAST'], start, end)
            if not rates.empty: fx.append(rates.assign(currency=ccy))
        errors = bbg.errors
    return ref, raw, pd.concat(fx, ignore_index=True) if fx else pd.DataFrame(columns=['date','PX_LAST','currency']), errors


def prepare(universe, ref, raw, fx, as_of):
    ref = ref.rename(columns={'NAME': 'bloomberg_name'})
    meta = universe.merge(ref, on='ticker', how='left', validate='one_to_one')
    if 'FUND_FLOW' not in raw: raw['FUND_FLOW'] = np.nan
    if 'date' not in raw: raw['date'] = pd.NaT
    d = raw[['ticker','date','FUND_FLOW']].dropna().copy()
    d['date'] = pd.to_datetime(d.date)
    d['FUND_FLOW'] = pd.to_numeric(d.FUND_FLOW, errors='raise')
    if d.duplicated(['ticker','date']).any(): raise ValueError('Duplicate Bloomberg flow observations')
    d = d.merge(meta[['ticker','group','NAV_CRNCY']], on='ticker', validate='many_to_one')
    d['fx_to_usd'] = np.nan
    d.loc[d.NAV_CRNCY == 'USD','fx_to_usd'] = 1.
    fx = fx.copy()
    fx['date'] = pd.to_datetime(fx.date)
    for ccy in d.NAV_CRNCY.dropna().unique():
        if ccy == 'USD': continue
        rates = fx[fx.currency == ccy].set_index('date').PX_LAST.sort_index()
        dates = d.loc[d.NAV_CRNCY == ccy, 'date']
        if not rates.empty:
            # Only past FX quotes, with at most seven calendar days of staleness.
            matched = rates.reindex(pd.DatetimeIndex(dates), method='ffill', tolerance=pd.Timedelta(days=7))
            d.loc[dates.index, 'fx_to_usd'] = matched.to_numpy()
    d.loc[d.fx_to_usd <= 0,'fx_to_usd'] = np.nan
    d['flow_usd_mm'] = d.FUND_FLOW * d.fx_to_usd
    d['week'] = d.date.dt.to_period('W-FRI').dt.end_time.dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    last_friday = as_of - pd.Timedelta(days=(as_of.weekday()-4)%7)
    cadence = []
    for t in meta.ticker:
        dates = d.loc[d.ticker == t,'date'].sort_values()
        recent = dates[dates >= as_of-pd.Timedelta(days=180)]
        gap = recent.diff().dt.days.dropna().median() if len(recent)>1 else np.nan
        weekly = len(recent)>=8 and pd.notna(gap) and gap<=7
        cadence.append(dict(ticker=t, weekly_reporter=weekly, recent_observations=len(recent),
                            median_gap_days=gap, last_flow_date=dates.max() if len(dates) else pd.NaT))
    meta = meta.merge(pd.DataFrame(cadence),on='ticker')
    # Same FIGI cannot contribute twice even if the spreadsheet has two aliases.
    if 'ID_BB_GLOBAL' in meta:
        ids = meta.ID_BB_GLOBAL.dropna()
        if ids.duplicated().any(): raise ValueError('Duplicate share-class FIGIs; resolve aliases before aggregation')
    periods = pd.date_range(d.week.min() if len(d) else last_friday, last_friday, freq='W-FRI')
    weekly = d.groupby(['week','ticker']).flow_usd_mm.sum(min_count=1).unstack('ticker') if len(d) else pd.DataFrame()
    if len(d):
        invalid = d.assign(invalid=d.flow_usd_mm.isna()).groupby(['week','ticker']).invalid.any().unstack('ticker')
        weekly = weekly.mask(invalid.fillna(False).astype(bool))
    weekly = weekly.reindex(index=periods,columns=meta.ticker)
    weekly.index.name = 'week_ending'
    aggregates = []
    for group in ['All','Active','Passive']:
        expected = meta if group=='All' else meta[meta.group==group]
        eligible = expected[expected.weekly_reporter]
        amounts = weekly[eligible.ticker.tolist()]
        result = pd.DataFrame(index=periods)
        result['net_usd_mm'] = amounts.sum(axis=1,min_count=1)
        result['positive_fund_net_usd_mm'] = amounts.clip(lower=0).sum(axis=1,min_count=1)
        result['negative_fund_net_usd_mm'] = amounts.clip(upper=0).sum(axis=1,min_count=1)
        result['reporting_funds'] = amounts.notna().sum(axis=1)
        result['eligible_weekly_funds'] = len(eligible)
        result['universe_funds'] = len(expected)
        result['group'] = group
        result['complete_eligible_coverage'] = (result.reporting_funds==len(eligible)) & (len(eligible)>0)
        aggregates.append(result.rename_axis('week_ending').reset_index())
    return meta, d, weekly, pd.concat(aggregates,ignore_index=True), last_friday


def report(meta, daily, weekly, aggregates, end):
    latest = meta.copy()
    latest['last_week_usd_mm'] = weekly.loc[end].reindex(latest.ticker).to_numpy()
    latest.loc[~latest.weekly_reporter,'last_week_usd_mm'] = np.nan
    last_observations = daily.sort_values('date').groupby('ticker').tail(1).set_index('ticker')
    latest['last_reported_flow_usd_mm'] = latest.ticker.map(last_observations.flow_usd_mm)
    latest['weekly_status'] = np.where(latest.weekly_reporter, 'Weekly/daily reporter', 'Sparse/monthly or unavailable; excluded from weekly totals')
    latest.to_csv(OUT/'fund_latest.csv',index=False)
    charts=[]
    monthly = []
    for group in ['All','Active','Passive']:
        subset = daily if group == 'All' else daily[daily.group == group]
        subset = subset.assign(month=subset.date.dt.to_period('M').dt.to_timestamp())
        m = subset.groupby('month').agg(net_usd_mm=('flow_usd_mm',lambda s:s.sum(min_count=1)),
                                      reporting_funds=('ticker','nunique')).reset_index()
        m['group'] = group
        monthly.append(m)
    monthly = pd.concat(monthly,ignore_index=True)
    monthly.to_csv(OUT/'monthly_reported_aggregates.csv',index=False)
    fig=go.Figure()
    for group in ['Active','Passive','All']:
        s=aggregates[aggregates.group==group]
        fig.add_trace(go.Bar(x=s.week_ending,y=s.net_usd_mm,name=group))
    fig.update_layout(template='plotly_white',barmode='group',height=480,title='Weekly observed net flows — weekly/daily reporters',yaxis_title='USD millions',hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=True,range=[end-pd.Timedelta(weeks=26),end])
    charts.append(fig.to_html(full_html=False,include_plotlyjs=True))
    fig=go.Figure()
    for group in ['All','Active','Passive']:
        s=aggregates[aggregates.group==group].set_index('week_ending')
        fig.add_trace(go.Scatter(x=s.index,y=s.net_usd_mm.rolling(4,min_periods=4).sum(),name=group))
    fig.update_layout(template='plotly_white',title='Trailing four-week observed net flows',yaxis_title='USD millions',height=430,hovermode='x unified')
    charts.append(fig.to_html(full_html=False,include_plotlyjs=False))
    month_fig = go.Figure()
    for group in ['All','Active','Passive']:
        m = monthly[monthly.group == group]
        month_fig.add_trace(go.Bar(x=m.month,y=m.net_usd_mm,name=group))
    month_fig.update_layout(template='plotly_white',height=430,barmode='group',
                            title='Monthly reported-flow totals — includes sparse reporters',
                            yaxis_title='USD millions',hovermode='x unified')
    monthly_chart = month_fig.to_html(full_html=False,include_plotlyjs=False)
    fig=go.Figure()
    for i,r in enumerate(meta.itertuples()):
        if r.weekly_reporter:
            x,y=weekly.index,weekly[r.ticker]
        else:
            s=daily[daily.ticker==r.ticker]
            x,y=s.date,s.flow_usd_mm
        fig.add_trace(go.Bar(x=x,y=y,name=r.ticker,visible=i==0))
    fig.update_layout(template='plotly_white',height=450,title=f'{meta.iloc[0].ticker} — reported flow observations',yaxis_title='USD millions',
                      updatemenus=[dict(buttons=[dict(label=f'{r.group}: {r.ticker}',method='update',args=[{'visible':[j==i for j in range(len(meta))]}, {'title':f'{r.ticker} — '+('weekly net flows' if r.weekly_reporter else 'reported observations; not weekly flows')}]) for i,r in enumerate(meta.itertuples())])])
    charts.append(fig.to_html(full_html=False,include_plotlyjs=False))
    labels = dict(group='Group',ticker='Bloomberg ticker',name='Fund',NAV_CRNCY='NAV currency',
                  net_usd_mm='Net flow ($mm)',positive_fund_net_usd_mm='Positive fund net flows ($mm)',
                  negative_fund_net_usd_mm='Negative fund net flows ($mm)',reporting_funds='Funds with data',
                  eligible_weekly_funds='Weekly-eligible funds',universe_funds='Listed funds',
                  complete_eligible_coverage='Eligible subset covered',last_week_usd_mm='Latest week ($mm)',
                  last_flow_date='Last observation',last_reported_flow_usd_mm='Last reported amount ($mm)',
                  weekly_status='Reporting status',month='Observation month')
    def table(d): return d.rename(columns=labels).to_html(index=False,border=0,na_rep='—',float_format=lambda v:f'{v:,.2f}',classes='data')
    current=aggregates[aggregates.week_ending==end].copy()
    columns=['group','net_usd_mm','positive_fund_net_usd_mm','negative_fund_net_usd_mm','reporting_funds','eligible_weekly_funds','universe_funds','complete_eligible_coverage']
    detail=latest[['group','ticker','name','NAV_CRNCY','last_week_usd_mm','last_flow_date','last_reported_flow_usd_mm','weekly_status']].copy()
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Fixed-income fund flows</title><style>
body{{font:14px Arial;background:#eef2f6;color:#17202a;margin:22px}}main{{max-width:1500px;margin:auto}}section{{background:white;padding:22px;margin:16px 0;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th{{background:#173e56;color:white;padding:9px;cursor:pointer}}td{{padding:8px;border:1px solid #ddd;text-align:right}}tr:nth-child(even){{background:#f3f6f9}}input{{padding:9px;width:350px;margin-bottom:12px}}.note{{color:#52606d}}summary{{cursor:pointer;font-weight:bold}}</style></head><body><main>
<section><h1>Fixed-income fund flow monitor</h1><p><a href="../fund_sensitivity_output/fund_sensitivity_monitor.html">Fund return sensitivities</a></p><p>Week ending {end.date()}. Generated {datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')} Shanghai time. Bloomberg data refreshed {json.loads((OUT/'source_manifest.json').read_text())['fetched_at']}.</p>
<p><b>Coverage matters:</b> weekly totals include only tickers with frequent flow observations. Sparse/monthly reporters are shown individually and excluded; their month-sized flows are never assigned to a single week. These are observed subtotals for the workbook's listed share classes, not complete flows across every share class of each fund.</p>{table(current[columns])}
<p class="note">Positive/negative columns sum positive/negative fund net flows, rather than gross subscriptions/redemptions. Missing funds are not zero. A zero is shown only when Bloomberg reports zero. Weeks end Friday using Bloomberg observation/process dates; current unfinished weeks are excluded.</p></section>
<section>{charts[0]}{charts[1]}</section><section><h2>Fund details</h2><input id="filter" placeholder="Search ticker, fund, active/passive or status">{table(detail)}</section>
<section><h2>Individual fund history</h2>{charts[2]}</section>
<section><details><summary>Monthly totals across all available fund reporters</summary><p>Includes both frequent and sparse reporters. Amounts are assigned to Bloomberg observation months; missing reporters are not zero. An observation may span an irregular reporting interval. The current month is partial.</p>{monthly_chart}{table(monthly.tail(36))}</details></section>
<section><details><summary>Data and methodology</summary><p>Universe: Active and Passive sheets of funds_info.xlsx, reread on each run. Source: Bloomberg Desktop API FUND_FLOW (FD447), millions of NAV_CRNCY. Bloomberg calculates net share creation/redemption value; mutual fund amounts can span irregular asset-reporting intervals. US ETP values use process date with a one-day reporting lag. No flow forward-fill or AUM-change proxy is used.</p><p>Weekly reporter classification: at least eight observations in the last 180 calendar days and median interval no more than seven calendar days. This is a reporting-frequency screen, not proof of complete coverage. Funds without a weekly value remain missing. Complete coverage refers only to the eligible weekly subset. Currency conversion uses same-date or prior FX within seven days, from currencyUSD Curncy PX_LAST. Without currency/FX data, USD flow is missing. Current universe is fixed throughout history; changes to the workbook change the monitored universe.</p></details></section>
<script>document.getElementById('filter').oninput=e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('#filter + table tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')}};document.querySelectorAll('table').forEach(t=>t.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{{let asc=h.dataset.asc!=='true';h.dataset.asc=asc;let rows=[...t.tBodies[0].rows];rows.sort((a,b)=>{{let x=a.cells[i].textContent.replace(/,/g,''),y=b.cells[i].textContent.replace(/,/g,'');if(x==='—'||y==='—')return x===y?0:x==='—'?1:-1;let z=Number.isFinite(Number(x))&&Number.isFinite(Number(y))?Number(x)-Number(y):x.localeCompare(y);return asc?z:-z}});rows.forEach(r=>t.tBodies[0].appendChild(r))}}));</script></main></body></html>'''
    path=OUT/'fund_flows_monitor.html'
    path.write_text(body,encoding='utf-8')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook',type=Path,default=WORKBOOK)
    p.add_argument('--update',action='store_true')
    p.add_argument('--start-date',default=None)
    a=p.parse_args()
    end=pd.Timestamp(datetime.now(ZoneInfo('Asia/Shanghai')).date())
    start=pd.Timestamp(a.start_date) if a.start_date else end-pd.DateOffset(years=2)
    OUT.mkdir(exist_ok=True)
    u=read_universe(a.workbook)
    u.to_csv(OUT/'universe.csv',index=False)
    paths=[OUT/n for n in ['reference.csv','raw_flows.csv','fx.csv']]
    refresh=a.update or not all(path.exists() for path in paths)
    if not refresh:
        old_ref=pd.read_csv(paths[0])
        refresh=set(old_ref.ticker)!=set(u.ticker)
    if refresh:
        ref,raw,fx,errors=fetch(u,start,end)
        for frame,path in zip([ref,raw,fx],paths): frame.to_csv(path,index=False)
        (OUT/'bloomberg_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        (OUT/'source_manifest.json').write_text(json.dumps(dict(fetched_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),start=str(start.date()),end=str(end.date()),field='FUND_FLOW',units='millions of NAV currency'),indent=2),encoding='utf-8')
    else:
        ref,raw,fx=[pd.read_csv(path) for path in paths]
    meta,daily,weekly,agg,last=prepare(u,ref,raw,fx,end)
    meta.to_csv(OUT/'coverage.csv',index=False)
    daily.to_csv(OUT/'daily_usd_flows.csv',index=False)
    weekly.to_csv(OUT/'weekly_fund_flows.csv')
    agg.to_csv(OUT/'weekly_aggregates.csv',index=False)
    path=report(meta,daily,weekly,agg,last)
    print(agg[agg.week_ending==last].to_string(index=False))
    print('Weekly eligible:',meta.groupby('group').weekly_reporter.sum().to_dict())
    print('REPORT',path)
    return path


if __name__=='__main__': main()
