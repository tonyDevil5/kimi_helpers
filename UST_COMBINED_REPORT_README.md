# Combined Treasury report

Open `ust_combined_output/UST_Combined_Report.html`. The links at the top switch
between UST fair value, technical analysis with DM government-bond comparisons, and Treasury CTA momentum / 10Y
EWMA-KAMA walk-forward, plus UST swaption ATM volatility and skew.
Each analysis keeps its interactive controls. Bookmarks can point to
`#fair-value`, `#technical`, `#momentum`, or `#swaptions`.

Technical analysis includes DM government-bond performance/carry, country spreads,
yield curves, quantiles and the 1-month heatmap under its **DM comparison** button.
See `UST_TECHNICAL_DM_README.md` for coverage and calculation conventions.

Refresh the analyses and produce the combined report in mercury_env:

```powershell
python ust_combined_report.py
```

To combine the latest existing reports without fetching new data:

```powershell
python ust_combined_report.py --from-existing
```

The backtesting repository defaults to the sibling `../BackTestingTool`. Set
`--backtest-root PATH` to use a different checkout, or `--output PATH` to select
another output file. A refresh stops if any underlying report fails; it does not
silently package a stale report as refreshed data.

The swaption section reads the saved `Data1` quotes in `mbs_rv.xlsx` and does not
refresh Excel's upstream data feed. It shows its own source date. Use
`python swaption_skew_report.py --format interactive --source PATH` to select a
different workbook. The default tries the original D: location then the Nutstore
data directory. No workbook is edited or recalculated.

The swaption table shows ATM and skew levels plus absolute changes over 1/5/20/30
source observations for 3m10y/1y10y/3y10y, in workbook annualized normal-vol bp.
Skew is +50bp payer minus -50bp receiver volatility. Horizons retain the source
observation calendar; missing endpoint quotes remain unavailable rather than being
filled or shortening the lookback. Source reference dates are displayed below the
table. Charts and a CSV download are included.

The generated HTML is a single portable file with compressed embedded reports
and no CDN dependencies. Open it in a current Chrome, Edge, Firefox or Safari
with JavaScript enabled. It requires the browser's gzip DecompressionStream API.
Plotly code is stored once per distinct library version and runs in each section's
own document, keeping styles, element IDs and JavaScript isolated. Sections load
on first selection and retain chart zooms, filters and selections when switching.
Original report files remain available for standalone use. Existing source dates
and provenance stay visible inside each analysis; the combined packaging timestamp
is not a market-data timestamp.
