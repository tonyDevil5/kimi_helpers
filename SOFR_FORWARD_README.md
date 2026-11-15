# SOFR forward monitor

## Historical analysis

Run `python sofr_forward_curve.py --history` for three years of Bloomberg
exact-date closes (configure with `--history-years`). Nonbusiness valuation dates
roll back to the preceding US Government Bond business day. The combined swap
report uses this mode automatically.

Persistent files in `sofr_forward_output/history/`:

- `input_quotes.csv`: dated LCH par rates (%) and CME-minus-LCH basis (bp).
- `forward_rates.csv`: 66 structures per complete date, both venue rates,
  forward basis, effective/maturity dates and maximum calibration error.
- `failed_dates.csv`: incomplete or failed dates excluded from the result.
- `metadata.json` and `source.json`: pricing conventions and data provenance.

Each update rechecks ten calendar days of overlap for revisions and downloads
new dates. Earlier inputs are retained. Complete cached results are reused only
when input quotes, QuantLib version and pricing settings match; use
`--rebuild-history` to recalculate everything. An expanded history window triggers
backfill. Missing pillar quotes are never forward-filled. No-data downloads fail
without replacing the cache.

The Historical section lets you select any forward structure and a date range,
view LCH/CME rates and forward basis, hover for dated values, and export the
selected history. Structures maintain constant start/tenor from each day's spot
date; they do not track a fixed-maturity trade through time.

Run in PowerShell:

```powershell
& D:\Anaconda\python.exe .\sofr_forward_curve.py --latest
```

This downloads latest available Bloomberg PX_LAST quotes and generates
`sofr_forward_output/<valuation-date>/monitor.html`, input quotes, calibration,
curve nodes, forward rates CSVs, and metadata. Latest available quotes may
have different update times. The HTML is an offline snapshot; rerun to refresh.

## Market pillars

Both curves use matching quoted maturities only:
**1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30 years**.

LCH uses `USOSFR{n} Curncy`. CME par quotes equal LCH par quotes plus
`USLCSS{n} Curncy` in basis points. No intermediate basis or par-rate quotes
are manufactured. Any missing quote at a configured pillar fails the run.
The former `--interpolate-missing` option has been removed.
QuantLib interpolates log discount factors between calibrated market pillars.

Bloomberg verified the basis security names as CME minus LCH. Use
`--basis-sign -1` for inputs defined with the opposite sign. Ticker templates
are configurable with `--lch-template` and `--basis-template`.

## Other inputs

```powershell
# Exact-date historical closes, no stale-date filling
& D:\Anaconda\python.exe .\sofr_forward_curve.py --asof 2026-10-30
# Synthetic demonstration, stored separately with a _demo suffix
& D:\Anaconda\python.exe .\sofr_forward_curve.py --demo --asof 2026-11-02
# Local snapshot for the supplied valuation date
& D:\Anaconda\python.exe .\sofr_forward_curve.py --quotes-csv quotes.csv --asof 2026-10-30
```

CSV requires exactly the 15 market pillars, with columns `tenor_years`,
`lch_rate_pct`, `basis_bps`. For example, 4.50 means 4.50 percent and 1.50
means 1.50 bp. CSV dates must match the supplied valuation date.

## Pricing

Outputs include all 66 combinations of forward starts {0,1,2,3,4,5,6,7,8,9,10}Y
and swap tenors {1,2,3,4,5,10}Y. The first number is years from spot and
the second is swap length. The HTML includes an 11 × 6 basis-only heatmap in bp, a detailed table, and filters for all six swap tenors. Each venue uses its own quote-implied
SOFR forecasting/discounting curve. This is a practical representation of
venue-specific par quotes, rather than a structural model of CCP margin costs.
Forward rates use dated OIS cash flows and QuantLib `fairRate()`.

Conventions: annual payments, compounded SOFR, Actual/360 coupons,
T+2 spot, US Government Bond calendar, Modified Following payments,
two-business-day payment lag, no end-of-month rule. QuantLib MakeOIS schedule
conventions apply. Settlement days/payment lag are configurable. Curve time
uses Actual/365 Fixed, with log-linear discount factors and no extrapolation.
Before the 1Y first pillar, curve shape depends on the interpolation assumption.
Confirm conventions against Bloomberg settings for terminal comparisons.

Each curve must reproduce its 15 input quotes within 0.00001 bp.
The monitor shows forward rates, model-implied forward CME–LCH basis, input
spot basis quotes, filters and CSV export. Forward basis generally differs
from spot-maturity basis quotes.

## Validation

```powershell
& D:\Anaconda\python.exe -m unittest test_sofr_forward_curve -v
```

Verified with QuantLib 1.33. Tests cover independent spot-swap repricing,
zero-basis forward equality, sign direction, and invalid quote rejection.

0Y starts are spot-starting (T+2) swaps. Adding pillars triggers a full-window input backfill and recalibration.
