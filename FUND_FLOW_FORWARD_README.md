# Fund flows as a forward market indicator

Run `python fund_flow_forward_analysis.py` in the Bloomberg-enabled mercury_env.
Use `--update` to refresh the flow and sensitivity monitor inputs first.
The interactive output is `fund_flow_forward_output/fund_flow_forward_analysis.html`.

X: Friday-ending active/passive net fund flows in USD billions, plus the five
preceding Friday-to-Friday factor changes. Y: the factor change from the following
Monday close to the next Monday close. Monday holidays use the first common
factor close within two days after Monday, never the preceding Friday.
The five targets are CT10, FN basis, IG OAS, 2s10s and annualized normal 1y10y vol.

Funds initially qualify using the first 12 flow weeks: at least 8 observations,
median date spacing ≤7 days, and at least 80% weekly availability. At each forecast
date, retain qualified funds with at least 80% availability in the previous 12 weeks.
The historical features used to fit that forecast are rebuilt for its reporting
cohort. This rule uses reporting availability, not target performance, and never
future coverage. Fund counts and identifiers are recorded per prediction.
Incomplete included-cohort weeks remain missing. No share-class or currency
flow value is carried forward to fill a hole.

Expanding OLS begins with at least 52 matched training targets. Only targets that
ended strictly before the forecast's scheduled Monday are used. The market-only
baseline and the flow-augmented model are fitted on exactly the same training rows.
Prediction logs, target endpoints, metrics, cohort and input CSVs allow reconciliation.
An extra-week-old flow variant tests a more conservative reporting delay.

Out-of-sample error improvement = 100 × (1 − MSE(flow model)/MSE(market-only)).
Positive means flows improved error; negative means they hurt. OOS R² is relative
to a zero-change prediction. Directional hit rates exclude zero actual changes.
Neither correlations nor forecasts are strategy P&L or proof of causality.

Historical Bloomberg flows are revised snapshots, not publication-time vintages.
Flow availability by the following Monday close is an assumption, not a verified
historical release schedule. Current workbook membership is not point-in-time.
Full-sample lead correlations and coefficient tables use the latest reporting
cohort retrospectively and are exploratory. The OOS test uses historical forecast
cohorts. No best target or parameter is chosen to trade from these results.

Test with `python -m unittest discover -s tests -p test_fund_flow_forward_analysis.py -v`.
