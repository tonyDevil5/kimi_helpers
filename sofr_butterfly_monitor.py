"""Spot and forward-starting SOFR par-rate butterflies, with incremental history."""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import QuantLib as ql

from eur_forward_basis_history import atomic_csv, read_csv
from sofr_forward_curve import build_curve

TENORS = (1,2,3,4,5,6,7,8,9,10,12,15,20,30)
PILLARS = tuple(sorted(set(TENORS) | {25,40}))
STARTS = tuple(range(11))
FLIES = tuple(zip(TENORS,TENORS[1:],TENORS[2:]))
OUTPUT = Path(__file__).parent/'sofr_butterfly_output'
VERSION = 'sofr-equal-wing-fly-v1-ql'+ql.__version__


def load_quotes(start, end):
    from xbbg import blp
    tickers = [f'USOSFR{n} Curncy' for n in PILLARS]
    print(f'Bloomberg: {start} to {end}, {len(tickers)} SOFR par-rate tickers',flush=True)
    frame=blp.bdh(tickers=tickers,flds=['PX_LAST'],start_date=start,end_date=end)
    if not isinstance(frame,pd.DataFrame):
        frame=frame.to_pandas()
    if {'date','ticker','value'}.issubset(frame.columns):
        frame=frame.pivot(index='date',columns='ticker',values='value')
    elif isinstance(frame.columns,pd.MultiIndex):
        frame.columns=frame.columns.get_level_values(0)
    if frame.empty:
        raise RuntimeError('Bloomberg returned no quotes. Saved data were not modified.')
    frame.index=pd.to_datetime(frame.index)
    frame=frame.reindex(columns=tickers).apply(pd.to_numeric,errors='coerce')
    return pd.DataFrame([dict(date=day.date().isoformat(),tenor_years=n,
                              rate_pct=float(row[f'USOSFR{n} Curncy']),ticker=f'USOSFR{n} Curncy')
                         for day,row in frame.sort_index().iterrows() for n in PILLARS])


def upsert(old, fresh):
    keys=['date','tenor_years']
    if fresh.duplicated(keys).any() or (not old.empty and old.duplicated(keys).any()):
        raise ValueError('Duplicate date/tenor input')
    merged=fresh.set_index(keys) if old.empty else fresh.set_index(keys).combine_first(old.set_index(keys))
    return merged.reset_index().sort_values(keys).reset_index(drop=True)


def quote_rows(group):
    group=group.sort_values('tenor_years')
    if tuple(group.tenor_years)!=PILLARS or not group.rate_pct.between(-5,30,inclusive='neither').all():
        raise ValueError('Missing or invalid SOFR input pillar; require all 16 1Y–40Y quotes in percent')
    return [dict(tenor_years=int(r.tenor_years),lch_rate_pct=float(r.rate_pct),basis_bps=0.)
            for r in group.itertuples()]


def input_hash(rows):
    return hashlib.sha256(json.dumps([VERSION,[(r['tenor_years'],format(r['lch_rate_pct'],'.12g')) for r in rows]]).encode()).hexdigest()


def par_forward(curve, asof, start_years, tenor_years):
    ql.Settings.instance().evaluationDate=asof
    calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    spot=calendar.advance(asof,2,ql.Days)
    effective=calendar.advance(spot,ql.Period(start_years,ql.Years),ql.ModifiedFollowing)
    handle=ql.YieldTermStructureHandle(curve)
    swap=ql.MakeOIS(ql.Period(tenor_years,ql.Years),ql.Sofr(handle),0.,effectiveDate=effective,
                    paymentCalendar=calendar,endOfMonth=False,
                    paymentFrequency=ql.Annual,paymentLag=2,paymentAdjustmentConvention=ql.ModifiedFollowing,
                    fixedLegDayCount=ql.Actual360(),telescopicValueDates=True)
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    extension=max(0,max(c.date() for c in swap.fixedLeg()+swap.overnightLeg())-curve.maxDate())
    if extension>10:
        raise ValueError(f'Cash flow exceeds 40Y curve by {extension} days; cap is 10')
    return dict(start_years=start_years,tenor_years=tenor_years,rate_pct=swap.fairRate()*100,
                effective_date=effective.ISO(),maturity_date=swap.maturityDate().ISO(),tail_extension_days=extension)


def calculate_day(group):
    rows=quote_rows(group)
    digest=input_hash(rows)
    asof=ql.DateParser.parseISO(group.date.iloc[0])
    curve,calibration=build_curve(rows,'LCH',asof)
    curve.enableExtrapolation()
    rates=[par_forward(curve,asof,start,n) for start in STARTS for n in TENORS]
    indexed={(r['start_years'],r['tenor_years']):r for r in rates}
    flies=[]
    for start in STARTS:
        for short,belly,long in FLIES:
            legs=[indexed[start,n] for n in (short,belly,long)]
            flies.append(dict(start_years=start,structure=f'{short}/{belly}/{long}',
                              short_years=short,belly_years=belly,long_years=long,
                              short_rate_pct=legs[0]['rate_pct'],belly_rate_pct=legs[1]['rate_pct'],
                              long_rate_pct=legs[2]['rate_pct'],
                              fly_bps=(2*legs[1]['rate_pct']-legs[0]['rate_pct']-legs[2]['rate_pct'])*100))
    return digest,rates,flies,calibration


def update(asof, offline=False, rebuild=False):
    folder=OUTPUT/'history';folder.mkdir(parents=True,exist_ok=True)
    paths={k:folder/(k+'.csv') for k in ('quotes','forward_rates','butterflies','calibration')}
    frames={k:read_csv(p) for k,p in paths.items()}
    quotes=frames['quotes']
    if not offline:
        initial=(pd.Timestamp(asof)-pd.DateOffset(years=2)).date().isoformat()
        prior=quotes.loc[quotes.date<=asof] if not quotes.empty else quotes
        start=initial if prior.empty else (pd.Timestamp(prior.date.max())-pd.Timedelta(days=7)).date().isoformat()
        quotes=upsert(quotes,load_quotes(start,asof))
        atomic_csv(quotes,paths['quotes'])
    if quotes.empty:
        raise ValueError('No local history. Run once with Bloomberg connected.')
    if rebuild:
        for k in ('forward_rates','butterflies','calibration'):
            frames[k]=pd.DataFrame()
    done={}
    flies=frames['butterflies']
    if not flies.empty and not frames['forward_rates'].empty and not frames['calibration'].empty:
        for day,g in flies.groupby('date'):
            rate_day=frames['forward_rates'].loc[frames['forward_rates'].date==day]
            checks=frames['calibration'].loc[frames['calibration'].date==day]
            pairs=set(zip(g.start_years,g.structure))
            expected={(s,f'{a}/{b}/{c}') for s in STARTS for a,b,c in FLIES}
            if len(g)==132 and pairs==expected and len(rate_day)==154 and len(checks)==16 and g.input_sha256.nunique()==1:
                done[day]=g.input_sha256.iloc[0]
    computed=reused=0;failures=[]
    def save():
        for key in ('forward_rates','butterflies','calibration'):
            atomic_csv(frames[key],paths[key])
    for day,group in quotes.loc[quotes.date<=asof].groupby('date',sort=True):
        if not ql.UnitedStates(ql.UnitedStates.GovernmentBond).isBusinessDay(ql.DateParser.parseISO(day)):
            continue
        try:
            digest=input_hash(quote_rows(group))
            if done.get(day)==digest:
                reused+=1;continue
            for key in ('forward_rates','butterflies','calibration'):
                if not frames[key].empty:
                    frames[key]=frames[key].loc[frames[key].date!=day]
            digest,rates,flies,checks=calculate_day(group)
            for key,rows in [('forward_rates',rates),('butterflies',flies),('calibration',checks)]:
                extra=pd.DataFrame([dict(date=day,input_sha256=digest,model_version=VERSION,**r) for r in rows])
                frames[key]=pd.concat([frames[key],extra],ignore_index=True)
            computed+=1
            if computed%25==0:
                print(f'Calculated {computed} dates; through {day}',flush=True);save()
        except (RuntimeError,ValueError) as exc:
            failures.append(dict(date=day,reason=str(exc)))
    save()
    status=dict(requested_asof=asof,model_version=VERSION,calculated_dates=computed,reused_dates=reused,failures=failures)
    (folder/'update_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'{computed} calculated, {reused} reused, {len(failures)} incomplete/failed dates',flush=True)
    return quotes,frames,status


def report(quotes,frames,status):
    flies=frames['butterflies'].loc[frames['butterflies'].date<=status['requested_asof']].sort_values('date')
    if flies.empty:
        raise ValueError('No calculated fly history. Inspect history/update_status.json')
    latest=flies.date.max()
    snapshot=[];hist={}
    for (start,structure),group in flies.groupby(['start_years','structure'],sort=False):
        group=group.sort_values('date');values=group.fly_bps
        last=group.iloc[-1].to_dict();tail=values.tail(250);std=tail.std(ddof=1)
        last['z250']=float((values.iloc[-1]-tail.mean())/std) if len(tail)==250 and std>1e-12 else None
        for n in (1,5,20):
            last[f'change_{n}d']=float(values.iloc[-1]-values.iloc[-1-n]) if len(values)>n else None
        last['z_mean']=float(tail.mean());last['z_std']=float(std) if len(tail)>1 else None
        last['z_start']=group.date.iloc[-min(250,len(group))];last['z_obs']=len(tail)
        snapshot.append(last)
        hist[f'{start}:{structure}']=group[['date','fly_bps']].to_dict('records')
    q=quotes.loc[quotes.date==latest].sort_values('tenor_years').to_dict('records')
    checks=frames['calibration'].loc[frames['calibration'].date==latest].to_dict('records')
    metadata=dict(asof=latest,history_start=flies.date.min(),history_dates=flies.date.nunique(),
                  generated=datetime.now(timezone(timedelta(hours=8))).isoformat(),quantlib=ql.__version__,
                  requested_asof=status['requested_asof'],calculated_dates=status['calculated_dates'],
                  reused_dates=status['reused_dates'],skipped_dates=len(status['failures']))
    payload=dict(metadata=metadata,tenors=TENORS,starts=STARTS,structures=[f'{a}/{b}/{c}' for a,b,c in FLIES],
                 snapshot=snapshot,history=hist,quotes=q,calibration=checks)
    encoded=json.dumps(payload,allow_nan=False).replace('<','\\u003c')
    template=Path(__file__).with_name('sofr_butterfly_template.html').read_text(encoding='utf-8')
    document=template.replace('__SNAPSHOT_JSON__',encoded)
    stable=OUTPUT/'SOFR_Butterfly_Monitor.html'
    stable.write_text(document,encoding='utf-8')
    (OUTPUT/f'SOFR_Butterfly_Monitor_{latest.replace("-","")}.html').write_text(document,encoding='utf-8')
    (OUTPUT/'latest_snapshot.json').write_text(encoded,encoding='utf-8')
    atomic_csv(pd.DataFrame(snapshot),OUTPUT/'latest_flies.csv')
    print(f'REPORT {stable.resolve()} | {metadata["history_dates"]} dates | Latest {latest}')
    return stable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asof',default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    quotes,frames,status=update(args.asof,args.offline,args.rebuild)
    report(quotes,frames,status)


if __name__=='__main__':
    main()
