"""Daily realized vol of CC basis (bps), coupon swaps and coupon flies (ticks).

Vol is the std of daily changes over trailing 20/40/60-day windows:
  - CC basis: std of daily OAS changes, in bps (Bloomberg .FNCC105/.G2CC105).
  - Coupon swaps (JPM adjacent-coupon perf): the series is already a daily
    change in 32nds, so vol = std of the series, in ticks.
  - Coupon flies: std of the daily fly change, in ticks (32nds).
"""

import numpy as np
import pandas as pd

from mbs_rv_report import (
    COUPON_FLY_CENTERS,
    COUPON_SWAP_PAIRS,
    STRUCTURE_AGENCIES,
    build_coupon_fly_price_timeseries,
    build_coupon_swap_jpm_perf_timeseries,
    load_current_coupon_oas,
    load_mbs_rv,
)

VOL_WINDOWS = [20, 40, 60]
OUTPUT_DIR = "mbs_rv_output"


def _vol_row(group: str, structure: str, daily: pd.Series, unit: str) -> dict:
    row = {"group": group, "structure": structure, "unit": unit,
           "latest_daily": float(daily.iloc[-1]), "obs": int(daily.notna().sum())}
    row["vol_all"] = float(daily.std())
    for w in VOL_WINDOWS:
        row[f"vol_{w}d"] = float(daily.rolling(w, min_periods=w).std().iloc[-1])
    return row


def main() -> None:
    dat = load_mbs_rv()
    rows = []

    # CC basis: daily OAS changes in bps
    cc = load_current_coupon_oas(
        dat.index.min().strftime("%Y-%m-%d"),
        dat.index.max().strftime("%Y-%m-%d"),
    )
    cc_vol_ts = pd.DataFrame(index=cc.index)
    for col, name in [("FNCC", "FN CC basis"), ("G2CC", "G2 CC basis")]:
        daily = cc[col].diff().dropna()
        rows.append(_vol_row("cc_basis", name, daily, "bps"))
        for w in VOL_WINDOWS:
            cc_vol_ts[f"{col}_vol_{w}d"] = daily.rolling(w, min_periods=w).std()

    # Coupon swaps: JPM daily perf in 32nds (already daily changes)
    swap = build_coupon_swap_jpm_perf_timeseries(dat)
    swap_vol_ts = pd.DataFrame(index=swap.index)
    for agency in STRUCTURE_AGENCIES:
        for high, low in COUPON_SWAP_PAIRS:
            key = f"{agency.lower()}_{high:g}_{low:g}_swap_perf_32nds"
            daily = swap[key].dropna()
            rows.append(_vol_row("cpn_swap", f"{agency} {high:g}/{low:g}", daily, "ticks"))
            for w in VOL_WINDOWS:
                swap_vol_ts[f"{key}_vol_{w}d"] = daily.rolling(w, min_periods=w).std()

    # Coupon flies: daily fly change in 32nds
    fly = build_coupon_fly_price_timeseries(dat)
    fly_vol_ts = pd.DataFrame(index=fly.index)
    for agency in STRUCTURE_AGENCIES:
        for center in COUPON_FLY_CENTERS:
            key = f"{agency.lower()}_{center:g}_fly_change_32nds"
            daily = fly[key].dropna()
            rows.append(_vol_row("fly", f"{agency} {center:g} fly", daily, "ticks"))
            for w in VOL_WINDOWS:
                fly_vol_ts[f"{key}_vol_{w}d"] = daily.rolling(w, min_periods=w).std()

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 100)
    print("\nDaily realized vol (std of daily changes)")
    print(f"As of {dat.index[-1].date()}  |  cc basis in bps, swaps & flies in ticks")
    print(summary.round(2).to_string(index=False))

    summary_path = f"{OUTPUT_DIR}/structure_daily_vol_summary.csv"
    summary.round(4).to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    vol_ts = cc_vol_ts.join(swap_vol_ts, how="outer").join(fly_vol_ts, how="outer")
    vol_ts.index.name = "date"
    ts_path = f"{OUTPUT_DIR}/structure_daily_vol_timeseries.csv"
    vol_ts.round(4).to_csv(ts_path)
    print(f"Saved vol timeseries: {ts_path}")


if __name__ == "__main__":
    main()
