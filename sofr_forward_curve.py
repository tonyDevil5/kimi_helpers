"""Bootstrap LCH/CME SOFR quote-implied curves and price forward OIS swaps.

Run with D:\\Anaconda\\python.exe; see SOFR_FORWARD_README.md for conventions.
Bloomberg par rates are percent; CCP basis quotes are basis points.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import QuantLib as ql


TENORS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)
FORWARD_STARTS = tuple(range(3, 11))
SWAP_TENORS = (1, 2, 3, 4, 5, 10)
FORWARDS = [(start, tenor) for start in FORWARD_STARTS for tenor in SWAP_TENORS]


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_quotes(rows):
    indexed = {}
    for row in rows:
        n = int(row["tenor_years"])
        if n in indexed:
            raise ValueError("Duplicate tenor: {}Y".format(n))
        rate = float(row["lch_rate_pct"])
        basis = float(row["basis_bps"])
        if not all(math.isfinite(x) for x in (rate, basis)):
            raise ValueError("Non-finite quote at {}Y".format(n))
        if not -5 < rate < 30 or abs(basis) > 500:
            raise ValueError("Implausible quote at {}Y; check percent/bp units".format(n))
        indexed[n] = dict(row, tenor_years=n, lch_rate_pct=rate, basis_bps=basis)
    if set(indexed) != set(TENORS):
        raise ValueError("Supply exactly one quote for each supported tenor: 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30")
    return [indexed[n] for n in TENORS]


def load_bloomberg(args):
    """Exact-date historical snapshot; missing pillars never silently filled."""
    from xbbg import blp
    import pandas as pd

    pairs = [(n, args.lch_template.format(n=n), args.basis_template.format(n=n))
             for n in TENORS]
    tickers = [t for _, lch, basis in pairs for t in (lch, basis)]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ticker templates must generate distinct securities")
    if args.latest:
        frame = blp.bdp(tickers=tickers, flds=[args.field])
    else:
        frame = blp.bdh(tickers=tickers, flds=[args.field],
                        start_date=args.asof, end_date=args.asof)
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    values = {}
    if args.latest:
        if {"ticker", "field", "value"}.issubset(frame.columns):
            for row in frame.to_dict("records"):
                if str(row["field"]).lower() == args.field.lower():
                    values[row["ticker"]] = float(row["value"])
        else:
            columns = [c for c in frame.columns if str(c).lower() == args.field.lower()]
            if columns:
                values = {t: float(frame.loc[t, columns[0]]) for t in tickers if t in frame.index}
    elif {"ticker", "date", "field", "value"}.issubset(frame.columns):
        for row in frame.to_dict("records"):
            if (str(row["field"]).lower() == args.field.lower()
                    and str(pd.Timestamp(row["date"]).date()) == args.asof):
                values[row["ticker"]] = float(row["value"])
    else:
        frame = frame.loc[pd.to_datetime(frame.index).strftime("%Y-%m-%d") == args.asof]
        for ticker in tickers:
            if isinstance(frame.columns, pd.MultiIndex):
                cols = [c for c in frame.columns if c[0] == ticker
                        and str(c[-1]).lower() == args.field.lower()]
                if cols and not frame.empty:
                    values[ticker] = float(frame[cols[0]].iloc[-1])
            elif ticker in frame.columns and not frame.empty:
                values[ticker] = float(frame[ticker].iloc[-1])
    missing = [t for t in tickers if t not in values or not math.isfinite(values[t])]
    if missing:
        raise ValueError("Missing quotes on {} (check source/date/ticker): {}".format(
            args.asof, ", ".join(missing)))
    return [dict(tenor_years=n, lch_rate_pct=values[lch], basis_bps=values[basis],
                 lch_ticker=lch, basis_ticker=basis,
                 lch_status="Quoted", basis_status="Quoted") for n, lch, basis in pairs]


def build_curve(rows, venue, asof, basis_sign=1, settlement_days=2, payment_lag=2):
    ql.Settings.instance().evaluationDate = asof
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    index = ql.Sofr()
    helpers, rates = [], []
    for row in rows:
        rate = row["lch_rate_pct"] / 100
        if venue == "CME":
            rate += basis_sign * row["basis_bps"] / 10000
        rates.append(rate)
        helpers.append(ql.OISRateHelper(
            settlement_days, ql.Period(row["tenor_years"], ql.Years),
            ql.QuoteHandle(ql.SimpleQuote(rate)), index,
            ql.YieldTermStructureHandle(), True, payment_lag,
            ql.ModifiedFollowing, ql.Annual, calendar, endOfMonth=False))
    curve = ql.PiecewiseLogLinearDiscount(asof, helpers, ql.Actual365Fixed())
    curve.discount(curve.maxDate())  # Force the lazy bootstrap; no extrapolation.
    calibration = []
    for row, helper, rate in zip(rows, helpers, rates):
        implied = helper.impliedQuote()
        error = (implied - rate) * 10000
        if abs(error) > 1e-5:
            raise ValueError("{} {}Y calibration error: {} bp".format(
                venue, row["tenor_years"], error))
        calibration.append(dict(venue=venue, tenor_years=row["tenor_years"],
                                input_rate_pct=rate * 100,
                                repriced_rate_pct=implied * 100, error_bps=error,
                                pillar_date=helper.latestDate().ISO()))
    return curve, calibration


def forward_rate(curve, asof, start_years, length_years, settlement_days=2, payment_lag=2):
    ql.Settings.instance().evaluationDate = asof
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    spot = calendar.advance(asof, settlement_days, ql.Days)
    start = calendar.advance(spot, ql.Period(start_years, ql.Years), ql.ModifiedFollowing)
    handle = ql.YieldTermStructureHandle(curve)
    swap = ql.MakeOIS(
        ql.Period(length_years, ql.Years), ql.Sofr(handle), 0.0,
        effectiveDate=start,
        paymentCalendar=calendar, endOfMonth=False,
        paymentFrequency=ql.Annual, paymentLag=payment_lag,
        paymentAdjustmentConvention=ql.ModifiedFollowing,
        fixedLegDayCount=ql.Actual360(), telescopicValueDates=True)
    swap.setPricingEngine(ql.DiscountingSwapEngine(handle))
    if max(cf.date() for cf in swap.overnightLeg()) > curve.maxDate():
        raise ValueError("Forward payments extend beyond the curve")
    return swap.fairRate(), start.ISO(), swap.maturityDate().ISO()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--quotes-csv", type=Path)
    source.add_argument("--demo", action="store_true", help="Synthetic data; no Bloomberg")
    parser.add_argument("--asof", default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument("--lch-template", default="USOSFR{n} Curncy")
    parser.add_argument("--basis-template", default="USLCSS{n} Curncy")
    parser.add_argument("--latest", action="store_true", help="Use latest available Bloomberg BDP quotes; individual quotes may be stale")
    parser.add_argument("--field", default="PX_LAST")
    parser.add_argument("--basis-sign", type=int, choices=[-1, 1], default=1,
                        help="CME = LCH + sign * basis; default +1 (CME minus LCH)")
    parser.add_argument("--settlement-days", type=int, default=2)
    parser.add_argument("--payment-lag", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "sofr_forward_output")
    args = parser.parse_args()
    if args.latest and (args.demo or args.quotes_csv):
        parser.error("--latest is only valid for Bloomberg input")
    if args.latest and args.asof != datetime.now(timezone(timedelta(hours=8))).date().isoformat():
        parser.error("--latest requires today's valuation date; use BDH for historical dates")
    if args.settlement_days < 0 or args.payment_lag < 0:
        parser.error("Settlement days and payment lag must be nonnegative")
    asof = ql.DateParser.parseISO(args.asof)
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    if not calendar.isBusinessDay(asof):
        parser.error("--asof must be a US Government Bond business day")
    if args.demo:
        rows = [dict(tenor_years=n, lch_rate_pct=3.3 + 0.025*n,
                     basis_bps=0.4 + 0.035*n) for n in TENORS]
    elif args.quotes_csv:
        with args.quotes_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = load_bloomberg(args)
    rows = validate_quotes(rows)
    curves, checks = {}, []
    for venue in ("LCH", "CME"):
        curves[venue], calibration = build_curve(
            rows, venue, asof, args.basis_sign, args.settlement_days, args.payment_lag)
        checks.extend(calibration)
    forwards = []
    for start, length in FORWARDS:
        lch, effective, maturity = forward_rate(
            curves["LCH"], asof, start, length, args.settlement_days, args.payment_lag)
        cme, _, _ = forward_rate(
            curves["CME"], asof, start, length, args.settlement_days, args.payment_lag)
        forwards.append(dict(forward="{}y{}y".format(start, length),
                             effective_date=effective, maturity_date=maturity,
                             lch_rate_pct=lch*100, cme_rate_pct=cme*100,
                             cme_minus_lch_bps=(cme-lch)*10000))
    output = args.output_dir / (args.asof + ("_demo" if args.demo else ""))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "input_quotes.csv", rows)
    write_csv(output / "calibration.csv", checks)
    write_csv(output / "forward_rates.csv", forwards)
    write_csv(output / "curve_nodes.csv", [
        dict(venue=venue, date=d.ISO(), discount_factor=df)
        for venue, curve in curves.items() for d, df in curve.nodes()])
    metadata = dict(asof=args.asof, synthetic=args.demo, quantlib_version=ql.__version__,
                    source=str(args.quotes_csv) if args.quotes_csv else ("demo" if args.demo else ("Bloomberg BDP latest available (individual quotes may be stale)" if args.latest else "Bloomberg BDH")),
                    retrieved_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    missing_quote_policy="Reject missing quotes; use only the 14 supported market maturities",
                    pillar_tenors=list(TENORS),
                    lch_template=args.lch_template, basis_template=args.basis_template,
                    field=args.field, basis_sign=args.basis_sign,
                    cme_formula="LCH decimal rate + basis_sign * basis_bps / 10000",
                    settlement_days=args.settlement_days, payment_lag=args.payment_lag,
                    calendar="US Government Bond", day_count="Actual/360",
                    frequency="Annual", business_day="Modified Following",
                    interpolation="Log-linear discount factors", first_pillar="2Y",
                    model="Separate single-curve quote-implied venue curves")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from sofr_forward_monitor import write_monitor
    write_monitor(output / "monitor.html", forwards, rows, checks, metadata)
    print("SYNTHETIC DEMO" if args.demo else "Market quote-implied forwards: " + args.asof)
    print("Assumption: CME = LCH {:+d} * basis (bp)".format(args.basis_sign))
    print("{:<8} {:>12} {:>12} {:>17}".format("Forward", "LCH (%)", "CME (%)", "CME-LCH (bp)"))
    for row in forwards:
        print("{forward:<8} {lch_rate_pct:>12.6f} {cme_rate_pct:>12.6f} {cme_minus_lch_bps:>17.6f}".format(**row))
    print("Saved to {}".format(output))
    print("HTML monitor: {}".format(output / "monitor.html"))


if __name__ == "__main__":
    main()
