import pandas as pd
import numpy as np


def forward_perf_on_zscore_v2(
    df,
    target_series,
    zscore_series,
    z_threshold,
    horizons=(5, 10, 20),
    signal_lag=1,
    cooldown=None,
):
    """
    Clean forward-performance backtest when a z-score crosses a threshold.

    Fixes vs. the original version:
    ------------------------------------------------------------
    1.  Conventional forward returns:  fwd = target[t+h] - target[t]
        (Positive = target went UP, negative = target went DOWN.)
    2.  Realistic execution lag: by default the signal is shifted by 1 day
        so you enter on t+1 after observing the z-score at t.
    3.  Optional cooldown: suppress overlapping trigger days so each
        observation is independent (useful for realistic strategy backtests).
    4.  Reports both % positive and % negative so you can read the result
        in the direction of your thesis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target and z-score columns.
    target_series : str
        Column name of the series whose forward change you want to measure
        (e.g. 'mbs' for the MBS spread level).
    zscore_series : str
        Column name containing the z-score (e.g. 'zscore').
    z_threshold : float
        Trigger threshold.
        * If negative, fires when zscore <= z_threshold (e.g. -2 for extreme low).
        * If positive, fires when zscore >= z_threshold (e.g. +2 for extreme high).
    horizons : tuple of int
        Forward holding periods in days (default: 5, 10, 20).
    signal_lag : int, default 1
        Number of days to lag the signal before entry.
        0 = enter on the same day you observe the z-score (aggressive).
        1 = enter on the next day (conservative, recommended for EOD data).
    cooldown : int or None, default None
        Minimum number of calendar days between two accepted triggers.
        * None  = every trigger day is counted (event-study style, allows overlaps).
        * e.g. 20 = once a trigger is accepted, ignore further signals for 20 days.

    Returns
    -------
    pd.DataFrame
        One row per horizon with columns:
        - count        : number of non-overlapping (or all) triggers
        - mean         : average forward return
        - median       : median forward return
        - std          : standard deviation of forward returns
        - pct_positive : % of observations where target went UP
        - pct_negative : % of observations where target went DOWN
        - p10          : 10th percentile
        - p90          : 90th percentile
    """
    target = pd.to_numeric(df[target_series], errors="coerce")
    zscore = pd.to_numeric(df[zscore_series], errors="coerce")

    # Conventional forward returns: what you earn from t to t+h
    fwd = {}
    for h in horizons:
        fwd[h] = target.shift(-h) - target
    fwd_df = pd.DataFrame(fwd, index=df.index)

    # Raw signal based on threshold sign
    if z_threshold < 0:
        raw_signal = zscore <= z_threshold
    else:
        raw_signal = zscore >= z_threshold

    # Apply execution lag (shift signal forward in time)
    signal = raw_signal.shift(signal_lag)

    # Apply cooldown to remove overlapping / consecutive triggers
    if cooldown is not None and cooldown > 0:
        trigger_idx = []
        last_trigger = None
        for dt in signal[signal.fillna(False)].index:
            if last_trigger is None or (dt - last_trigger).days >= cooldown:
                trigger_idx.append(dt)
                last_trigger = dt
        trigger_mask = pd.Series(False, index=df.index)
        if trigger_idx:
            trigger_mask.loc[trigger_idx] = True
    else:
        trigger_mask = signal.fillna(False)

    # Summarise per horizon
    rows = []
    for h in horizons:
        triggered_vals = fwd_df.loc[trigger_mask, h].dropna()
        n = len(triggered_vals)

        row = {
            "horizon": h,
            "count": n,
            "mean": float(triggered_vals.mean()) if n > 0 else np.nan,
            "median": float(triggered_vals.median()) if n > 0 else np.nan,
            "std": float(triggered_vals.std()) if n > 0 else np.nan,
            "pct_positive": float((triggered_vals > 0).mean()) if n > 0 else np.nan,
            "pct_negative": float((triggered_vals < 0).mean()) if n > 0 else np.nan,
            "p10": float(triggered_vals.quantile(0.10)) if n > 0 else np.nan,
            "p90": float(triggered_vals.quantile(0.90)) if n > 0 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Example usage (copy-paste into your notebook):
# ---------------------------------------------------------------------------
# from backtest_utils import forward_perf_on_zscore_v2
#
# # Your data: result has columns ['mbs', 'ig', 'spread', 'zscore']
# # zscore < -1  =>  MBS tight / rich vs IG
# # We expect mean-reversion: MBS should cheapen (go UP) afterward.
# 
# # 1. Event-study style (overlapping allowed, no cooldown)
# forward_perf_on_zscore_v2(
#     result,
#     target_series="mbs",
#     zscore_series="zscore",
#     z_threshold=-1,
#     horizons=[5, 10, 20],
#     signal_lag=1,       # enter next day
#     cooldown=None,      # count every trigger day
# )
#
# # 2. Strategy style (no overlapping signals, 20-day cooldown)
# forward_perf_on_zscore_v2(
#     result,
#     target_series="mbs",
#     zscore_series="zscore",
#     z_threshold=-1,
#     horizons=[5, 10, 20],
#     signal_lag=1,
#     cooldown=20,        # wait 20 days before taking a new signal
# )
