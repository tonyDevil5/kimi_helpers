"""Persistent exact-date Bloomberg quotes and calibrated SOFR forward history."""
import hashlib
import json

import pandas as pd
import QuantLib as ql

from eur_forward_basis_history import atomic_csv, read_csv, merge_inputs


def fetch_inputs(core, args, start, end):
    from xbbg import blp
    tickers = [template.format(n=n) for n in core.TENORS
               for template in (args.lch_template, args.basis_template)]
    print(f'Bloomberg SOFR history: {start} to {end} ({len(tickers)} tickers)', flush=True)
    frame = blp.bdh(tickers=tickers, flds=[args.field], start_date=start, end_date=end)
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    if {'ticker', 'date', 'field', 'value'}.issubset(frame.columns):
        frame = frame.loc[frame.field.str.lower() == args.field.lower()]
        frame = frame.pivot(index='date', columns='ticker', values='value')
    elif isinstance(frame.columns, pd.MultiIndex):
        frame = frame.loc[:, frame.columns.get_level_values(-1).str.lower() == args.field.lower()]
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise RuntimeError('Bloomberg returned no historical quotes; cache unchanged.')
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reindex(columns=tickers).apply(pd.to_numeric, errors='coerce')
    return pd.DataFrame([
        dict(date=day.date().isoformat(), tenor_years=n,
             lch_rate_pct=row[args.lch_template.format(n=n)],
             basis_bps=row[args.basis_template.format(n=n)],
             lch_ticker=args.lch_template.format(n=n), basis_ticker=args.basis_template.format(n=n))
        for day, row in frame.sort_index().iterrows() for n in core.TENORS
    ])


def calculate(core, quotes, saved, args):
    """Reuse complete dates only when both quotes and pricing conventions match."""
    settings = dict(version=1, quantlib=ql.__version__, basis_sign=args.basis_sign,
                    settlement_days=args.settlement_days, payment_lag=args.payment_lag,
                    forwards=core.FORWARDS)
    results, failures, reused = [], [], 0
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    for day, group in quotes.groupby('date', sort=True):
        asof = ql.DateParser.parseISO(day)
        if not calendar.isBusinessDay(asof):
            continue
        try:
            rows = core.validate_quotes(group.to_dict('records'))
            digest = hashlib.sha256(json.dumps([settings, [[r['tenor_years'], r['lch_rate_pct'], r['basis_bps']]
                for r in rows]], sort_keys=True).encode()).hexdigest()
            cached = saved.loc[saved.date == day] if not saved.empty else pd.DataFrame()
            if (not args.rebuild_history and len(cached) == len(core.FORWARDS)
                    and set(cached.forward) == {f'{s}y{t}y' for s, t in core.FORWARDS}
                    and cached.input_sha256.eq(digest).all()):
                results.extend(cached.to_dict('records'))
                reused += 1
                continue
            curves, checks = {}, []
            for venue in ('LCH', 'CME'):
                curves[venue], fit = core.build_curve(rows, venue, asof, args.basis_sign,
                    args.settlement_days, args.payment_lag)
                checks.extend(fit)
            dated = []
            for start, length in core.FORWARDS:
                lch, effective, maturity = core.forward_rate(curves['LCH'], asof, start, length,
                    args.settlement_days, args.payment_lag)
                cme, _, _ = core.forward_rate(curves['CME'], asof, start, length,
                    args.settlement_days, args.payment_lag)
                dated.append(dict(date=day, forward=f'{start}y{length}y', effective_date=effective,
                    maturity_date=maturity, lch_rate_pct=lch*100, cme_rate_pct=cme*100,
                    cme_minus_lch_bps=(cme-lch)*10000, input_sha256=digest,
                    max_calibration_error_bps=max(abs(c['error_bps']) for c in checks)))
            results.extend(dated)
        except Exception as exc:
            failures.append(dict(date=day, reason=str(exc)))
        if (len(results)//len(core.FORWARDS)) % 50 == 0:
            print(f'SOFR history through {day}', flush=True)
    return pd.DataFrame(results), pd.DataFrame(failures, columns=['date', 'reason']), reused


def update(core, args):
    root = args.output_dir / 'history'
    root.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ('lch_template', 'basis_template', 'field')}
    config_path = root / 'source.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('History source configuration changed; use a separate --output-dir.')
    old = read_csv(root / 'input_quotes.csv')
    end = pd.Timestamp(args.asof)
    start = end - pd.DateOffset(years=args.history_years)
    # Requery a short overlap for corrections, and backfill if the requested window grows.
    if not old.empty and set(old.tenor_years.astype(int)) == set(core.TENORS) and pd.Timestamp(old.date.min()) <= start + pd.Timedelta(days=7):
        start = max(start, pd.Timestamp(old.date.max()) - pd.Timedelta(days=10))
    fresh = fetch_inputs(core, args, start.date().isoformat(), args.asof)
    quotes = merge_inputs(old, fresh)
    atomic_csv(quotes, root / 'input_quotes.csv')
    config_path.write_text(json.dumps(config, indent=2))
    selected = quotes.loc[(quotes.date <= args.asof) &
        (quotes.date >= (end-pd.DateOffset(years=args.history_years)).date().isoformat())]
    forwards, failures, reused = calculate(core, selected, read_csv(root / 'forward_rates.csv'), args)
    if forwards.empty:
        raise RuntimeError('No complete historical dates could be priced; see source quotes.')
    atomic_csv(forwards, root / 'forward_rates.csv')
    atomic_csv(failures, root / 'failed_dates.csv')
    (root / 'metadata.json').write_text(json.dumps(dict(updated_at=pd.Timestamp.now(tz='UTC').isoformat(),
        source=config, history_years=args.history_years, basis_sign=args.basis_sign,
        settlement_days=args.settlement_days, payment_lag=args.payment_lag, quantlib=ql.__version__,
        missing_policy='Skip incomplete dates; never forward-fill quotes',
        dates=forwards.date.nunique(), reused_dates=reused, failed_dates=len(failures)), indent=2))
    print(f'SOFR history: {forwards.date.nunique()} dates, {reused} reused, {len(failures)} failed', flush=True)
    return forwards.to_dict('records'), failures.to_dict('records')
