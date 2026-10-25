# Combined MBS report

Ask: **Run the latest combined MBS report.**

Open `mbs_combined_output/MBS_Combined_Report.html`. Its five section links are:

- `#mbs-rv`: MBS relative-value monitor.
- `#tba-strategies`: TBA OLS fair-value and level-fade strategy analysis.
- `#jpm-mean-reversion`: JPM performance mean reversion, including coupon and structure groups.
- `#mbs-credit`: MBS versus credit basis analysis and backtest.
- `#fair-value-120`: 120-observation UST/FN fair-value backtest, including its CTA/TBA comparison and strategy correlations.

Refresh all five analyses and build the combined HTML in mercury_env:

```powershell
python mbs_combined_report.py
```

To package the existing reports without fetching data or rerunning backtests:

```powershell
python mbs_combined_report.py --from-existing
```

The sibling `../BackTestingTool` checkout is the default; override it with
`--backtest-root PATH`, or choose another output file with `--output PATH`.
The refresh command supplies the current monitor checkout on PYTHONPATH so
backtesting imports work even when that checkout has another folder name.
Workbook-fed sections read saved Excel data; this command does not recalculate
Excel or refresh its upstream feeds. TBA source inputs and Bloomberg credit
history are explicitly refreshed. A failed source run stops the build.

The latest MBS RV report is selected by its date across both filename conventions,
not alphabetically by the presence of `Interactive`. Backtesting reports have
stable filenames and show their own data dates inside each section. The packaging
timestamp is not a data date. Strategy rules, execution, sizing and analytics
remain those of the individual reports; the combined file does not aggregate
their P&L or assume comparable notionals.

The combined file embeds compressed HTML and chart code and can be moved or
opened offline in a modern browser with JavaScript and DecompressionStream.
The existing credit report's Plotly CDN reference is replaced in the embedded
copy with the installed local Plotly bundle; original standalone files remain
intact. Each section retains its own document, controls, CSS and JavaScript.
Reports load on first selection and retain state between visits. The underlying
HTML-packaging implementation is shared with the combined Treasury report.
