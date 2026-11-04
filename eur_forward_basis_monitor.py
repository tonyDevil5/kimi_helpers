"""EUR forward par-swap-rate differences: EURIBOR 3M minus ESTR, 6M minus 3M.

Run with D:\\Anaconda\\python.exe (QuantLib). Bloomberg PX_LAST quotes are percent.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import QuantLib as ql

TENORS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)
STARTS = tuple(range(1, 11))
LENGTHS = (1, 2, 3, 5, 7, 10, 20)
TEMPLATES = {'estr': 'EESWE{n} BGN Curncy', 'eur3m': 'EUSW{n}V3 BGN Curncy', 'eur6m': 'EUSA{n} BGN Curncy'}
OUTPUT = Path(__file__).parent / 'eur_forward_basis_output'


def validate_quotes(rows):
    indexed = {}
    for row in rows:
        n = int(row['tenor_years'])
        if n in indexed:
            raise ValueError(f'Duplicate {n}Y pillar')
        row = dict(row, tenor_years=n)
        for curve in TEMPLATES:
            rate = float(row[curve + '_pct'])
            if not math.isfinite(rate) or not -5 < rate < 30:
                raise ValueError(f'Invalid {curve} {n}Y quote; expected percent, e.g. 3.1')
            row[curve + '_pct'] = rate
            row.setdefault(curve + '_ticker', TEMPLATES[curve].format(n=n))
        indexed[n] = row
    if set(indexed) != set(TENORS):
        raise ValueError(f'Require all 15 pillars: {TENORS}')
    return [indexed[n] for n in TENORS]


def load_bloomberg(asof):
    import pandas as pd
    from xbbg import blp
    tickers = [template.format(n=n) for n in TENORS for template in TEMPLATES.values()]
    start = (datetime.fromisoformat(asof) - timedelta(days=15)).date().isoformat()
    frame = blp.bdh(tickers=tickers, flds=['PX_LAST'], start_date=start, end_date=asof)
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    if {'ticker', 'date', 'value'}.issubset(frame.columns):
        frame = frame.pivot(index='date', columns='ticker', values='value')
    elif isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reindex(columns=tickers).apply(pd.to_numeric, errors='coerce').sort_index()
    complete = frame.dropna()
    if complete.empty:
        missing = frame.columns[frame.isna().all()].tolist()
        raise ValueError(f'No common complete Bloomberg snapshot in {start} to {asof}; unavailable tickers: {missing}')
    snapshot = complete.iloc[-1]
    quote_date = complete.index[-1].date().isoformat()
    rows = [dict(tenor_years=n, **{curve+'_pct': float(snapshot[template.format(n=n)])
                                  for curve, template in TEMPLATES.items()}) for n in TENORS]
    return validate_quotes(rows), quote_date


def build_curves(rows, asof):
    rows = validate_quotes(rows)
    ql.Settings.instance().evaluationDate = asof
    calendar = ql.TARGET()
    helpers = [ql.OISRateHelper(2, ql.Period(r['tenor_years'], ql.Years),
                               ql.QuoteHandle(ql.SimpleQuote(r['estr_pct']/100)), ql.Estr(),
                               ql.YieldTermStructureHandle(), True, 2,
                               ql.ModifiedFollowing, ql.Annual, calendar, endOfMonth=False)
               for r in rows]
    estr = ql.PiecewiseLogLinearDiscount(asof, helpers, ql.Actual365Fixed())
    estr.discount(estr.maxDate())
    curves = {'estr': estr}
    helper_sets = {'estr': helpers}
    discount = ql.YieldTermStructureHandle(estr)
    for name, index in [('eur3m', ql.Euribor3M()), ('eur6m', ql.Euribor6M())]:
        helpers = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(r[name+'_pct']/100)),
                                    ql.Period(r['tenor_years'], ql.Years), calendar,
                                    ql.Annual, ql.ModifiedFollowing, ql.Thirty360(ql.Thirty360.BondBasis),
                                    index, ql.QuoteHandle(), ql.Period(0, ql.Days), discount,
                                    2, ql.Pillar.LastRelevantDate, ql.Date(), False) for r in rows]
        curve = ql.PiecewiseLogLinearDiscount(asof, helpers, ql.Actual365Fixed())
        curve.discount(curve.maxDate())
        curves[name] = curve
        helper_sets[name] = helpers
    # Calendar adjustment of 10Y + 20Y can extend a few days beyond spot + 30Y.
    # Forward pricing below limits this log-linear tail to 10 calendar days.
    for curve in curves.values():
        curve.enableExtrapolation()
    calibration = []
    for name, helpers in helper_sets.items():
        for row, helper in zip(rows, helpers):
            implied = helper.impliedQuote()
            error = (implied - row[name+'_pct']/100)*10000
            if abs(error) > 1e-5:
                raise ValueError(f'{name} {row["tenor_years"]}Y repricing error {error} bp')
            calibration.append(dict(curve=name, tenor_years=row['tenor_years'], input_pct=row[name+'_pct'],
                                    repriced_pct=implied*100, error_bps=error))
    return curves, calibration


def forward_rates(curves, asof, start_years, length_years):
    ql.Settings.instance().evaluationDate = asof
    calendar = ql.TARGET()
    spot = calendar.advance(asof, 2, ql.Days)
    start = calendar.advance(spot, ql.Period(start_years, ql.Years), ql.ModifiedFollowing)
    discount = ql.YieldTermStructureHandle(curves['estr'])
    ois = ql.MakeOIS(ql.Period(length_years, ql.Years), ql.Estr(discount), 0.0,
                     effectiveDate=start, paymentCalendar=calendar, endOfMonth=False,
                     paymentFrequency=ql.Annual, paymentLag=2,
                     paymentAdjustmentConvention=ql.ModifiedFollowing,
                     fixedLegDayCount=ql.Actual360(), telescopicValueDates=True)
    ois.setPricingEngine(ql.DiscountingSwapEngine(discount))
    def check_horizon(curve, horizon):
        extension = max(0, horizon - curve.maxDate())
        if extension > 10:
            raise ValueError(f'Cash flow extends {extension} days past the curve; maximum permitted is 10')
        return extension
    extension = check_horizon(curves['estr'], max(cf.date() for cf in ois.fixedLeg()+ois.overnightLeg()))
    rates = {'estr_pct': ois.fairRate()*100}
    for name, index_type in [('eur3m', ql.Euribor3M), ('eur6m', ql.Euribor6M)]:
        index = index_type(ql.YieldTermStructureHandle(curves[name]))
        swap = ql.MakeVanillaSwap(ql.Period(length_years, ql.Years), index, 0.0, ql.Period(0, ql.Days),
                                  effectiveDate=start, fixedLegTenor=ql.Period(1, ql.Years),
                                  fixedLegDayCount=ql.Thirty360(ql.Thirty360.BondBasis),
                                  fixedLegCalendar=calendar, floatingLegCalendar=calendar,
                                  fixedLegConvention=ql.ModifiedFollowing,
                                  floatingLegConvention=ql.ModifiedFollowing,
                                  fixedLegTerminationDateConvention=ql.ModifiedFollowing,
                                  floatingLegTerminationDateConvention=ql.ModifiedFollowing,
                                  fixedLegEndOfMonth=False, floatingLegEndOfMonth=False)
        swap.setPricingEngine(ql.DiscountingSwapEngine(discount))
        horizon = max(cf.date() for cf in swap.fixedLeg()+swap.floatingLeg())
        extension = max(extension, check_horizon(curves['estr'],horizon),check_horizon(curves[name],horizon))
        rates[name+'_pct'] = swap.fairRate()*100
    return dict(forward=f'{start_years}Y{length_years}Y', start_years=start_years, tenor_years=length_years,
                effective_date=start.ISO(), maturity_date=ois.maturityDate().ISO(), **rates,
                tail_extension_days=extension,
                eur3m_minus_estr_bps=(rates['eur3m_pct']-rates['estr_pct'])*100,
                eur6m_minus_eur3m_bps=(rates['eur6m_pct']-rates['eur3m_pct'])*100,
                eur6m_minus_estr_bps=((rates['eur3m_pct']-rates['estr_pct'])*100
                                      + (rates['eur6m_pct']-rates['eur3m_pct'])*100))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_zscores(history, asof, window=250):
    """Level z-score including the endpoint, using sample standard deviation."""
    result = {}
    metrics = ('eur3m_minus_estr_bps','eur6m_minus_eur3m_bps','eur6m_minus_estr_bps')
    for name, rows in history.items():
        ordered = sorted((r for r in rows if r['date'] <= asof), key=lambda r:r['date'])
        result[name] = {}
        for metric in metrics:
            valid = [r for r in ordered if r.get(metric) is not None and math.isfinite(r[metric])][-window:]
            stat = dict(z=None, obs=len(valid), mean=None, std=None)
            if len(valid)==window and valid[-1]['date']==asof:
                values=[r[metric] for r in valid]
                mean=math.fsum(values)/window
                std=math.sqrt(math.fsum((v-mean)**2 for v in values)/(window-1))
                stat.update(mean=mean,std=std,start=valid[0]['date'],end=valid[-1]['date'],
                            z=(values[-1]-mean)/std if std>1e-12 else None)
            result[name][metric]=stat
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asof', default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument('--quotes-csv', type=Path, help='Columns tenor_years, estr_pct, eur3m_pct, eur6m_pct')
    parser.add_argument('--offline', action='store_true', help='Use saved history without Bloomberg')
    parser.add_argument('--rebuild', action='store_true', help='Recalculate all stored historical forward rates')
    args = parser.parse_args()
    history, history_meta = {}, {}
    if args.quotes_csv:
        with args.quotes_csv.open(encoding='utf-8-sig') as f:
            quotes = validate_quotes(list(csv.DictReader(f)))
        asof = args.asof
        source = f'Local supplied quotes: {args.quotes_csv.name}'
        qdate = ql.DateParser.parseISO(asof)
        curves, calibration = build_curves(quotes, qdate)
        forwards = [forward_rates(curves, qdate, start, tenor) for start in STARTS for tenor in LENGTHS]
    else:
        from eur_forward_basis_history import update
        quotes, asof, forwards, calibration, history, history_meta = update(
            sys.modules[__name__], args.asof, offline=args.offline, rebuild=args.rebuild)
        source = 'Bloomberg PX_LAST; persistent daily history; no forward filling'
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT/f'EUR_Quotes_{asof.replace("-", "")}.csv', quotes)
    metadata = dict(asof=asof, requested_asof=args.asof, source=source,
                    generated=datetime.now(timezone(timedelta(hours=8))).isoformat(), quantlib=ql.__version__, **history_meta)
    zscores = latest_zscores(history,asof)
    payload = dict(metadata=metadata, quotes=quotes, calibration=calibration, forwards=forwards, history=history, zscores=zscores)
    write_csv(OUTPUT/f'EUR_Forward_Basis_{asof.replace("-", "")}.csv', forwards)
    write_csv(OUTPUT/f'EUR_Quotes_{asof.replace("-", "")}.csv', quotes)
    write_csv(OUTPUT/f'EUR_Calibration_{asof.replace("-", "")}.csv', calibration)
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    document = Path(__file__).with_name('eur_forward_basis_template.html').read_text(encoding='utf-8').replace('__SNAPSHOT_JSON__', encoded)
    path = OUTPUT/f'EUR_Forward_Basis_Monitor_{asof.replace("-", "")}.html'
    path.write_text(document, encoding='utf-8')
    (OUTPUT/'EUR_Forward_Basis_Monitor.html').write_text(document, encoding='utf-8')
    (OUTPUT/'latest_snapshot.json').write_text(encoded, encoding='utf-8')
    print(f'As of {asof}: {len(forwards)} forwards, 3 matrices, max repricing error {max(abs(r["error_bps"]) for r in calibration):.8f} bp')
    print('REPORT', path)


if __name__ == '__main__':
    main()
