# EUR forward basis monitor

Run the live interactive monitor:

```powershell
& D:\Anaconda\python.exe eur_forward_basis_monitor.py
```

Open `eur_forward_basis_monitor.html` in the project root to go to the latest
populated report, or use `eur_forward_basis_output/EUR_Forward_Basis_Monitor.html`.
The source template is `eur_forward_basis_template.html`; it is not a report.

Requires QuantLib, pandas and xbbg (available in the base Anaconda environment).
The default requested date is today's Asia/Shanghai date. First run fetches two
calendar years of Bloomberg history and computes the 70 forwards for each complete
TARGET business date. Subsequent runs fetch from seven calendar days before the
last stored input date, upsert revised/new nonmissing quotes, and calculate only
uncached or changed dates. Existing older observations are retained. Missing quotes
are never forward-filled or substituted with the example table; failed/incomplete
dates are listed in `history/update_status.json`.

Persistent files under `eur_forward_basis_output/history/`:

- `quotes.csv`: date × 15 spot pillars, each with all three percentage rates/tickers.
- `forwards.csv`: date × 70 forward structures, component rates and all three bases.
- `calibration.csv`: dated repricing checks for the 45 input quotes.
- `update_status.json`: calculated/reused dates, model version and failure reasons.

Writes use temporary files followed by atomic replacement; the backfill checkpoints
every 25 calculated dates. Quote fingerprints/model version identify revised dates;
repeated runs do not duplicate rows. The same historical raw input reproduces its
forward calculations. Recent re-query NaNs cannot erase an existing saved quote.

The monitor adds a selectable historical chart with date filters, point tooltips,
CSV export, 1/5/20-observation changes and a 120-observation level z-score (sample
standard deviation). These statistics use the selected date range. Historical
series track constant forward-start/tenor structures, rather than one ageing swap.

```powershell
& D:\Anaconda\python.exe eur_forward_basis_monitor.py --offline
& D:\Anaconda\python.exe eur_forward_basis_monitor.py --offline --rebuild
```

`--offline` regenerates from local inputs/cache. `--rebuild` forces recalculation
of the stored historical forward rates. Updating is performed when the script runs;
no scheduler is installed. Keep copies of the local data folder for backup.

Output: `eur_forward_basis_output/EUR_Forward_Basis_Monitor_YYYYMMDD.html`,
forward-rate/basis CSV, quoted inputs CSV, calibration CSV and snapshot JSON.
The standalone HTML works offline, with clickable/keyboard-selectable cells,
component rates, dated forwards and CSV download. All three matrices are 10 × 7:
forward starts 1–10Y; swap lengths 1, 2, 3, 5, 7, 10 and 20Y.

## Tickers and signs

Pillars: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30Y.

- ESTR: `EESWE{n} BGN Curncy`
- EURIBOR 3M: `EUSW{n}V3 BGN Curncy`
- EURIBOR 6M: `EUSA{n} BGN Curncy`

Bloomberg `PX_LAST` rates are percentages. Forward basis = 100 times the
difference between percentage rates: **3M − ESTR**, **6M − 3M**, in bp.
The third matrix is **6M − ESTR**, the sum of the first two before rounding.
Each basis matrix has a companion 250-observation level z-score matrix. Use the
latest 250 valid saved basis levels, including the endpoint, with sample standard
deviation (ddof=1). Positive means historically wide, negative historically tight.
Require 250 observations and positive volatility. The 6M−ESTR z-score is calculated
from its own summed basis history; component z-scores are not added. Hover a cell
for its mean, standard deviation and window dates. Chart date filters do not alter
these latest-snapshot z-scores.

## Curves

ESTR is bootstrapped from OIS quotes and discounts all swaps. The 3M and 6M
projection curves are independently bootstrapped with that discount curve.
QuantLib PiecewiseLogLinearDiscount; Actual/365 Fixed curve time. The longest
10Y-forward 20Y swap may extend a few days beyond the spot 30Y pillar due to TARGET
calendar adjustment. Permit at most 10 calendar days of flat terminal-forward-rate
extrapolation and record `tail_extension_days` for each forward; reject any larger
extension. This prevents one calendar edge case from discarding an entire date.
All 45 helpers must reprice within 0.00001 bp. Before the first 1Y pillar,
interpolation defines the short-end shape.

TARGET, T+2, Modified Following, no EOM. OIS uses annual Actual/360 fixed
payments, compounded ESTR and 2-business-day payment lag. Euribor swaps use
annual 30/360 Bond Basis fixed coupons, quarterly/semiannual Actual/360
floating coupons, and no payment lag. Forward fair rates use dated swap cash
flows. Bloomberg terminal comparisons require matching these conventions.

These are differences of forward **par fixed swap rates**, not directly quoted
floating-versus-floating basis-swap margins. OIS/Euribor day counts and payment
dates differ. The raw par quotes alone do not uniquely identify an exact
market-convention float-float basis without additional convention choices.

Replay an exported quote snapshot (explicit quote date required):

```powershell
& D:\Anaconda\python.exe eur_forward_basis_monitor.py --quotes-csv PATH --asof YYYY-MM-DD
& D:\Anaconda\python.exe -m unittest test_eur_forward_basis_monitor -v
& D:\Anaconda\python.exe -m unittest test_eur_forward_basis_history -v
```
