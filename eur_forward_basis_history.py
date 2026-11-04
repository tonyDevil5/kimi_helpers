"""Incremental Bloomberg inputs and dated QuantLib forward-rate caches."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import QuantLib as ql

MODEL_VERSION = 'eur-multicurve-v2-bounded-calendar-tail-ql' + ql.__version__


def atomic_csv(frame, path):
    temporary = path.with_suffix('.tmp.csv')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig', float_format='%.17g')
    os.replace(temporary, path)


def read_csv(path):
    try:
        return pd.read_csv(path, float_precision='round_trip') if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fetch_inputs(core, start, end):
    from xbbg import blp
    tickers = [t.format(n=n) for n in core.TENORS for t in core.TEMPLATES.values()]
    print(f'Bloomberg history request: {start} to {end} (45 tickers)', flush=True)
    frame = blp.bdh(tickers=tickers, flds=['PX_LAST'], start_date=start, end_date=end)
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    if {'ticker', 'date', 'value'}.issubset(frame.columns):
        frame = frame.pivot(index='date', columns='ticker', values='value')
    elif isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise RuntimeError('Bloomberg returned no history; local files were not modified.')
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reindex(columns=tickers).apply(pd.to_numeric, errors='coerce').sort_index()
    records = []
    for day, snapshot in frame.iterrows():
        for n in core.TENORS:
            row = dict(date=day.date().isoformat(), tenor_years=n)
            for name, template in core.TEMPLATES.items():
                row[name+'_pct'] = snapshot[template.format(n=n)]
                row[name+'_ticker'] = template.format(n=n)
            records.append(row)
    return pd.DataFrame(records)


def merge_inputs(old, fresh):
    """Upsert nonmissing values; an empty re-query cannot erase a saved quote."""
    keys = ['date', 'tenor_years']
    fresh = fresh.set_index(keys)
    if fresh.index.duplicated().any():
        raise ValueError('Duplicate input date/tenor')
    if old.empty:
        return fresh.reset_index().sort_values(keys).reset_index(drop=True)
    old = old.set_index(keys)
    if old.index.duplicated().any():
        raise ValueError('Duplicate date/tenor in saved input history')
    return fresh.combine_first(old).reset_index().sort_values(keys).reset_index(drop=True)


def fingerprint(rows):
    values = [[int(r['tenor_years'])]+[format(float(r[c+'_pct']), '.12g') for c in ('estr','eur3m','eur6m')] for r in rows]
    return hashlib.sha256(json.dumps([MODEL_VERSION, values]).encode()).hexdigest()


def compute_missing(core, quotes, forwards, calibration, rebuild=False, checkpoint=None):
    cached = {}
    if not rebuild and not forwards.empty and 'input_sha256' in forwards:
        for day, group in forwards.groupby('date'):
            pairs=set(zip(group.start_years, group.tenor_years))
            expected={(s,t) for s in core.STARTS for t in core.LENGTHS}
            if len(group)==70 and pairs==expected and group.input_sha256.nunique()==1:
                if not calibration.empty and len(calibration[calibration.date==day])==45:
                    cached[day] = group.input_sha256.iloc[0]
    if rebuild:
        forwards, calibration = pd.DataFrame(), pd.DataFrame()
    changed, reused, failures = [], [], []
    for i, (day, group) in enumerate(quotes.groupby('date', sort=True)):
        qdate = ql.DateParser.parseISO(day)
        if not ql.TARGET().isBusinessDay(qdate):
            continue
        try:
            rows = core.validate_quotes(group.sort_values('tenor_years').to_dict('records'))
            digest = fingerprint(rows)
            if cached.get(day)==digest:
                reused.append(day)
                continue
            # Remove obsolete results before a revision is recalculated.
            if not forwards.empty:
                forwards = forwards.loc[forwards.date!=day]
            if not calibration.empty:
                calibration = calibration.loc[calibration.date!=day]
            curves, fits = core.build_curves(rows, qdate)
            computed = [dict(date=day, input_sha256=digest, model_version=MODEL_VERSION,
                             **core.forward_rates(curves,qdate,s,t)) for s in core.STARTS for t in core.LENGTHS]
            forwards = pd.concat([forwards,pd.DataFrame(computed)],ignore_index=True)
            calibration = pd.concat([calibration,pd.DataFrame([dict(date=day, **r) for r in fits])],ignore_index=True)
            changed.append(day)
            if len(changed)%25==0:
                print(f'Calculated {len(changed)} dates; through {day}',flush=True)
                if checkpoint:
                    checkpoint(forwards,calibration)
        except (ValueError,RuntimeError) as exc:
            failures.append(dict(date=day, reason=str(exc)))
    if not forwards.empty:
        forwards=forwards.sort_values(['date','start_years','tenor_years']).reset_index(drop=True)
    if not calibration.empty:
        calibration=calibration.sort_values(['date','curve','tenor_years']).reset_index(drop=True)
    return forwards, calibration, dict(calculated_dates=len(changed), reused_dates=len(reused), failures=failures)


def update(core, asof, offline=False, rebuild=False):
    folder=core.OUTPUT/'history'
    folder.mkdir(parents=True,exist_ok=True)
    qp,fp,cp=[folder/name for name in ('quotes.csv','forwards.csv','calibration.csv')]
    quotes,forwards,calibration=read_csv(qp),read_csv(fp),read_csv(cp)
    initial=(pd.Timestamp(asof)-pd.DateOffset(years=2)).date().isoformat()
    if not offline:
        start=initial if quotes.empty else max(initial,(pd.Timestamp(quotes.date.max())-pd.Timedelta(days=7)).date().isoformat())
        fresh=fetch_inputs(core,start,asof)
        quotes=merge_inputs(quotes,fresh)
        atomic_csv(quotes,qp)
    if quotes.empty:
        raise ValueError('No local input history. Run once with Bloomberg connected.')
    eligible=quotes.loc[quotes.date<=asof]
    def checkpoint(f,c):
        atomic_csv(f,fp)
        atomic_csv(c,cp)
    forwards,calibration,audit=compute_missing(core,eligible,forwards,calibration,rebuild,checkpoint)
    checkpoint(forwards,calibration)
    (folder/'update_status.json').write_text(json.dumps(dict(requested_asof=asof,model_version=MODEL_VERSION,**audit),indent=2),encoding='utf-8')
    print(f'History: {audit["calculated_dates"]} calculated, {audit["reused_dates"]} reused, {len(audit["failures"])} incomplete/failed dates',flush=True)
    available=forwards.loc[forwards.date<=asof] if not forwards.empty else forwards
    if available.empty:
        raise ValueError('No valid forward calculations; inspect history/update_status.json')
    latest=available.date.max()
    rows=core.validate_quotes(quotes.loc[quotes.date==latest].to_dict('records'))
    fits=calibration.loc[calibration.date==latest].drop(columns='date').to_dict('records')
    snapshot=available.loc[available.date==latest].drop(columns=['date','input_sha256','model_version']).to_dict('records')
    history={}
    for name, group in available.groupby('forward',sort=False):
        history[name]=group[['date','eur3m_minus_estr_bps','eur6m_minus_eur3m_bps','eur6m_minus_estr_bps']].sort_values('date').to_dict('records')
    meta=dict(history_start=available.date.min(), history_end=latest, history_dates=available.date.nunique(),
              calculated_dates=audit['calculated_dates'], reused_dates=audit['reused_dates'],
              skipped_dates=len(audit['failures']), cache_folder=str(folder.resolve()))
    return rows,latest,snapshot,fits,history,meta
