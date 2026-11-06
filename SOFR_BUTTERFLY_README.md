# SOFR curve butterfly monitor

```powershell
& D:\Anaconda\python.exe D:\kimi_code\mkt_monitors\sofr_butterfly_monitor.py
```

Open `sofr_butterfly_monitor.html` in the project root, or
`sofr_butterfly_output/SOFR_Butterfly_Monitor.html`. The source template is
`sofr_butterfly_template.html`; open the generated report rather than the template.
Requires QuantLib, xbbg and pandas, available in the base Anaconda environment.

## Structures

Consecutive triples from 1/2/3/4/5/6/7/8/9/10/12/15/20/30:
1/2/3, 2/3/4, 3/4/5, 4/5/6, 5/6/7, 6/7/8, 7/8/9, 8/9/10,
9/10/12, 10/12/15, 12/15/20, 15/20/30.

Fly bp = 100 × (2 × belly percentage rate − short-wing percentage rate −
long-wing percentage rate). Equal weights apply even for uneven tenor spacing.
These rate flies have no DV01/notional sizing and are not strategy P&L.

Rows: spot plus forward starts 1–10Y. Columns: 12 flies, giving 132 structures.
For example, 5Y forward 10/12/15 = 2 × 5Y12Y − 5Y10Y − 5Y15Y.
Two adjacent colour matrices show current fly levels and each fly's own trailing
250-valid-observation level z-score, including the endpoint, using sample standard
deviation (ddof=1). Require 250 points and positive volatility. Spot summary shows
leg rates, 1/5/20-observation changes and z-score. Click any cell to select its
historical chart; filter dates, hover points, or download CSV. History follows
constant forward starts/tenors, not a single ageing swap.

## Source and valuation

Bloomberg `USOSFR{n} Curncy`, PX_LAST in percent, same family as the existing
LCH SOFR forward monitor. Quote pillars: requested tenors plus 25Y and 40Y.
40Y is needed for the 10Y-forward 30Y leg. QuantLib bootstraps one SOFR curve
per date via the existing `sofr_forward_curve.build_curve` implementation.
All 16 inputs reprice within 0.00001 bp. Dated OIS swaps compute each forward.

T+2, US Government Bond calendar, Modified Following, annual Actual/360 fixed
coupons, compounded SOFR, 2-business-day payment lag, no EOM. Curve time is
Actual/365 Fixed with log-linear discount-factor interpolation. Allow up to ten
calendar days of flat terminal-forward extrapolation for calendar-adjusted 10Y+30Y
cash flows; reject longer extensions. Saved forward rows record the extension.

## Local history

First run backfills two calendar years through the requested Asia/Shanghai date.
Later runs query the preceding week from the last saved date, upsert nonmissing
quotes and calculate only changed/new dates. Complete curves are required; missing
inputs/valuation failures are recorded, never silently filled. Raw quotes are kept.
No scheduled task is installed; refresh occurs when you run the script.

Files under `sofr_butterfly_output/history/`:

- `quotes.csv`: date × 16 Bloomberg spot quotes.
- `forward_rates.csv`: date × 154 component forward par rates.
- `butterflies.csv`: date × 132 calculated flies.
- `calibration.csv`: quote repricing audit.
- `update_status.json`: changed/reused counts and failure reasons.

CSV files use atomic replacement, with checkpoints every 25 calculated dates.
Fingerprints include model version and quotes, allowing resume/reuse and revisions.
Generated market data and reports are gitignored.

```powershell
& D:\Anaconda\python.exe sofr_butterfly_monitor.py --offline
& D:\Anaconda\python.exe sofr_butterfly_monitor.py --offline --rebuild
& D:\Anaconda\python.exe sofr_butterfly_monitor.py --asof YYYY-MM-DD
& D:\Anaconda\python.exe -m unittest test_sofr_butterfly_monitor -v
```
