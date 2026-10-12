# 10Y CTA walk-forward extension

Run:

```powershell
& D:\Anaconda\envs\mercury_env\python.exe D:\kimi_code\mkt_monitors\cta\ust10y_walkforward.py
```

Add `--update` to request fresh Bloomberg yields. Without it, use the existing
four tenor caches in BackTestingTool. Original scripts/caches are read-only;
all new outputs are in `cta/walkforward_output` next to this extension.
The HTML is self-contained, with embedded Plotly, and can be shared directly.

The PDF-inspired methods use 576 EWMA and 240 KAMA candidates across four
signal tenors, but always trade 10Y. Training is expanding, minimum 504 common
observations, reselected every 20. Ranking ends before the decision date;
portfolio PnL uses previous close exposure. Parameter audit CSVs record each
selection and cutoff. Original fixed-rule CTA methods and the existing
20bp stop / 10bp TP / 30-day holding scenario are retained for comparison.

Evaluation starts at common observation 505, not a hardcoded 2018 date.
Bloomberg trading calendar and common-tenor filtering can differ from the
PDF's calendar. Strict break ties, initialization, ranking ties and selection
retention are documented in code; results are not claimed to reproduce the PDF.
Do not interpret daily bp Sharpe as a portfolio return on allocated capital.
DV01 is capped at $1,000/bp, with fractional exposure; costs are zero.
Carry, funding, roll and convexity are excluded. Historical 2026 was already
viewed during development and is not an untouched holdout.

Tests:

```powershell
& D:\Anaconda\envs\mercury_env\python.exe -m unittest discover -s D:\kimi_code\mkt_monitors\cta -p test_ust10y_walkforward.py -v
```
