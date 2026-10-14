# Fixed-income fund flows

The monitor rereads `funds_info.xlsx` Active and Passive sheets on every run.
It generates the interactive `fund_flows_output/fund_flows_monitor.html` plus
raw source, per-fund weekly flows, group aggregates, coverage and monthly CSVs.

Refresh from Bloomberg while the Terminal is logged in:

```powershell
& D:\Anaconda\envs\mercury_env\python.exe fund_flows_monitor.py --update
```

Without `--update`, use the saved Bloomberg snapshot. `--workbook` accepts a
different universe file; `--start-date YYYY-MM-DD` controls a fresh download.
Default download history is the last two years. No automatic schedule is installed.

Bloomberg `FUND_FLOW` is net creations/redemptions in millions of the primary
share class's NAV currency. The monitor converts flows to USD using daily
`currencyUSD Curncy PX_LAST`, with only previous quotes up to seven days old.
Week ending Friday groups actual observations; unfinished weeks are excluded.
Missing values remain unavailable; flow values are never forward-filled.
Positive and negative columns sum positive/negative per-fund weekly net amounts,
not gross subscriptions/redemptions.

Weekly aggregation includes funds with at least eight observations in the recent
180 days and median interval no greater than seven days. Other fund observations
are visible individually and in monthly reported-flow totals. A frequency screen
does not prove that every flow was captured. Coverage is reported explicitly.
Sparse fund observations may cover longer intervals and must not be interpreted
as a week's economic flow. Complete weekly coverage refers to the eligible subset,
not necessarily all 60 tickers. Monthly totals use observation months.

The fund workbook names are retained, and Bloomberg names, FIGIs, currencies and
report dates are stored in coverage.csv. Duplicate tickers/FIGIs are rejected.
Flows cover listed share classes, not necessarily every share class in each fund;
workbook AUM figures are reference labels and are never used to scale class flows.
Current universe is used throughout history (no point-in-time universe).

Checks: `python -m unittest discover -s tests -p test_fund_flows_monitor.py -v`.
