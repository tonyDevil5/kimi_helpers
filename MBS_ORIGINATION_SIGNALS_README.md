# Origination signals

Run from kimi_helpers with the sibling BackTestingTool checkout available:

```powershell
python mbs_origination_trading.py
```

Open `mbs_origination_signal_output/All_Coupon_Origination_Signals.html`.
This standalone dashboard is deliberately excluded from the combined MBS report.
It displays all 20 FN/G2 coupons independently, sortable performance/signal tables,
LONG/SHORT/FLAT badges and interactive signal, net P&L and position histories.
To rebuild the layout from exported results without recalculating:

```powershell
python mbs_origination_dashboard.py
```

The fixed rule uses 20-observation origination volume versus its preceding
250-observation history. Enter a daily short sleeve at z >= 1, long at z <= -1,
otherwise no new sleeve. Five equal sleeves create fractional positions capped
at $10mm per coupon. Signals trade at the next observation's close and first
receive performance one observation later. Current positions are simulated,
not reconciled to a live book. Rebalance exports are model plans, not broker orders.

JPM performance is already in ticks. The shared
`BackTestingTool/performance_position_backtester.py` adapter uses SimpleBacktester
absolute-price accounting: $10mm / 100 / 32 = $3,125 per tick. Default costs are
0.5 tick per full-notional one-way turnover. Parameters: `--notional`, `--cost-ticks`.
No additional carry, financing, roll or explicit hedge contracts are modelled.

The original tracker treats blanks as zero volume. The default replicates that
convention, exports blank counts, and separately tests strict missing data.
Publication times and revisions are unavailable. Zero-filled results must not
be interpreted as validated without confirming that source convention.
Evaluation from 2025-06-12 is retrospective; no untouched holdout is claimed.
HAC uses 20 lags; dashboard FDR covers valid tests in the fixed-rule coupon family.

Exports include `all_coupon_signals.csv`, `all_coupon_performance.csv`,
`all_coupon_daily.csv`, `all_coupon_cost_delay.csv`, per-coupon rebalance logs,
`strict_missing_all_coupon_performance.csv` and `signal_config.json`.
No aggregate portfolio allocation is imposed.

`mbs_origination_signal.py` is the earlier exploratory walk-forward OLS study,
with separate outputs and methodology; it is not the dashboard's fixed trading rule.

Validation:

```powershell
python -m unittest test_mbs_origination_signal test_mbs_origination_trading test_mbs_combined_report -q
```
