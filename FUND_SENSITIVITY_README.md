# Fund return sensitivities

Run with the Bloomberg-enabled mercury_env Python:

```powershell
& D:\Anaconda\envs\mercury_env\python.exe fund_sensitivity_monitor.py --update
```

`--update` refreshes Bloomberg total-return indices and spread data. Without it,
reuse cached Bloomberg inputs. Both workbook inputs are reread on each run;
`--workbook` changes the fund universe and `--mbs-workbook` changes factor input.
The default output is the interactive `fund_sensitivity_output/fund_sensitivity_monitor.html`.

Each fund's weekly USD total return is regressed on weekly changes in five factors:
CT10 yield, FN current-coupon ZV basis, corporate IG OAS, 2s10s and annualized
1y10y normal volatility. Y is return in basis points; each X is a level change in
basis points. Thus beta is return bps per +1bp factor change. An intercept is fitted.
Simple correlations are calculated using the same joint sample as the regression.

Fund returns use Bloomberg TOT_RETURN_INDEX_GROSS_DVDS, converted from security
CRNCY to USD using currencyUSD Curncy PX_LAST (past observations, max 3 days old).
No NAV-only price return is silently substituted for missing total returns.
Friday-ending levels use each week's last actual observation, max 3 calendar days
old. No filling across missing weeks; missing endpoints invalidate adjacent returns.
Only completed weeks and the last two years of overlap are evaluated.

CT10 and CT2 use mbs_rv.xlsx Data sheet, yield percentages ×100. Volatility uses
Data1 1y10y_atm in annualized normal bps. FN uses .FNCC105 G Index PX_LAST ×100.
IG uses LUACTRUU Index OAS, or LUACOAS Index PX_LAST ×100 if unavailable; source
is recorded in source_manifest.json. Native currency exposures remain embedded
in fund returns. These common US factors may explain little of non-US fund returns.

Full overlap models require at least 40 matched observations and full-rank inputs.
Rolling regressions use the latest 52 calendar weeks with at least 40 matched weeks.
HAC/Newey-West coefficient uncertainty uses 4 matched-observation lags. In-sample
fits include the displayed return; they are descriptive, not out-of-sample forecasts.
Factors can be correlated; distinguish simple correlation from conditional beta.
Many tests are displayed; statistical significance is not multiple-testing adjusted.

Group baskets use fixed membership among funds with estimable full-period models,
and report included/listed counts. A week is retained only when all included
constituent returns exist. Workbook whole-fund AUM is not used to weight
single-share-class returns. Group baskets are hypothetical, not investable indices.

YTD per-fund return uses USD total-return index levels from the last available
previous year-end close to the latest completed Friday-ending week (max 3-day
endpoint staleness). It does not compound a first weekly return that partly spans
the previous year. Group YTD is the mean of the same fixed constituents' YTD returns,
equivalent to equal capital at year start without rebalancing. It is missing if any
included fund lacks YTD inputs; included/data counts are shown. This differs from
the weekly equal-weight rebalanced baskets used for regression.

Raw Bloomberg inputs, weekly returns/factors, coverage, coefficient uncertainty,
per-fund summary and rolling beta CSVs are written next to the HTML.
Tests: `python -m unittest discover -s tests -p test_fund_sensitivity_monitor.py -v`.
