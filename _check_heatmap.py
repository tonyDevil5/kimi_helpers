import pandas as pd
from mbs_rv_report import (
    load_mbs_rv,
    build_cpn_fly_summary_table,
    add_ols_metrics_to_summary,
    add_range_metrics_to_summary,
    DEFAULT_SCORE_INDICES,
    _group_cpn_fly_rows,
)

dat = load_mbs_rv()
cpn_fly_summary = build_cpn_fly_summary_table(dat)
cpn_fly_summary = add_ols_metrics_to_summary(dat, cpn_fly_summary)
cpn_fly_summary = add_range_metrics_to_summary(dat, cpn_fly_summary)

time_cols = [f"{prefix}_t{i}" for i in range(len(DEFAULT_SCORE_INDICES)) for prefix in ("vals", "scores")]
model_cols = ["mdl_val", "resid_std", "r2", "resid_z"]
coef_cols = ["coef_price", "coef_slope", "coef_carry", "coef_vol"]
market_cols = ["mkt_val", "carry", "hist_min", "hist_max", "range_ratio"]
cpn_fly_summary = cpn_fly_summary[time_cols + model_cols + market_cols + coef_cols]

groups = _group_cpn_fly_rows(cpn_fly_summary)

for group_name, group_df in groups.items():
    print(f"\n=== {group_name} ===")
    print(f"scores_t0: min={group_df['scores_t0'].min():.3f}, max={group_df['scores_t0'].max():.3f}")
    print(f"resid_z:   min={group_df['resid_z'].min():.3f}, max={group_df['resid_z'].max():.3f}")
    print(f"range_ratio: min={group_df['range_ratio'].min():.3f}, max={group_df['range_ratio'].max():.3f}")
