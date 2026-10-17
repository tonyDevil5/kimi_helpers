"""Ad-hoc: .FNCC105 index performance 1m before -> 1m after Fed starts hiking.

Ticker = .FNCC105 G Index = "FN30 CC Spread to 5/10" (Fannie 30yr
current-coupon spread to the 5s/10s Treasury blend).
Unit convention: px_last is in percent; bp change = diff * 100
(e.g. 0.978% -> 1.144% = +16.6bp).
Hiking-cycle start dates from 2010 onward:
  - 2015-12-16 (first hike of the 2015-2018 cycle)
  - 2022-03-16 (first hike of the 2022-2023 cycle)
"""

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
from xbbg import blp

TICKER = ".FNCC105 G Index"

# First rate increase of each hiking cycle since 2010
HIKE_STARTS = [
    ("2015-12-16", "2015-2018 cycle"),
    ("2022-03-16", "2022-2023 cycle"),
]

START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")


def load_fncc105(start_date: str, end_date: str) -> pd.Series:
    """Download .FNCC105 G Index px_last (in percent) from Bloomberg."""
    df = blp.bdh(tickers=TICKER, flds=["px_last"],
                 start_date=start_date, end_date=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0].sort_index().rename("FNCC105")


def nearest_level(s: pd.Series, dt: pd.Timestamp) -> float:
    """Level on the closest trading day to dt (looking both directions)."""
    if dt in s.index:
        return s.loc[dt]
    before = s.index[s.index <= dt]
    after = s.index[s.index >= dt]
    if len(before) == 0:
        return s.loc[after[0]]
    if len(after) == 0:
        return s.loc[before[-1]]
    b, a = before[-1], after[0]
    return s.loc[b] if (dt - b) <= (a - dt) else s.loc[a]


def main() -> None:
    px = load_fncc105(START_DATE, END_DATE)
    print(f"Loaded {TICKER}: {len(px)} obs, "
          f"{px.index[0].date()} -> {px.index[-1].date()}, "
          f"last = {px.iloc[-1]:.3f}\n")

    rows = []
    for hike_date, label in HIKE_STARTS:
        t0 = pd.Timestamp(hike_date)
        t_pre = t0 - pd.DateOffset(months=1)
        t_post = t0 + pd.DateOffset(months=1)

        lvl_pre = nearest_level(px, t_pre)
        lvl_hike = nearest_level(px, t0)
        lvl_post = nearest_level(px, t_post)

        chg_total_bp = (lvl_post - lvl_pre) * 100.0
        chg_total_pct = (lvl_post / lvl_pre - 1.0) * 100.0
        chg_pre_bp = (lvl_hike - lvl_pre) * 100.0
        chg_post_bp = (lvl_post - lvl_hike) * 100.0

        rows.append({
            "cycle": label,
            "first_hike": hike_date,
            "level_1m_before": round(lvl_pre, 3),
            "level_hike_day": round(lvl_hike, 3),
            "level_1m_after": round(lvl_post, 3),
            "chg_pre_bp": round(chg_pre_bp, 1),
            "chg_post_bp": round(chg_post_bp, 1),
            "chg_1m_pre_to_1m_post_bp": round(chg_total_bp, 1),
            "chg_pct": round(chg_total_pct, 2),
        })

    summary = pd.DataFrame(rows).set_index("cycle")
    print("=== .FNCC105 around Fed hiking-cycle starts ===")
    print(summary.to_string())
    print("\nAverage change (1m before -> 1m after): "
          f"{summary['chg_1m_pre_to_1m_post_bp'].mean():.1f} bp")

    # Chart: indexed path around each event (t-1m to t+1m, hike day = 0)
    fig, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 4.5),
                             sharey=False)
    if len(rows) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, summary.iterrows()):
        t0 = pd.Timestamp(r["first_hike"])
        window = px.loc[t0 - pd.DateOffset(days=45): t0 + pd.DateOffset(days=45)]
        ax.plot(window.index, window.values, lw=1.5)
        ax.axvline(t0, color="red", ls="--", lw=1, label="first hike")
        ax.axvline(t0 - pd.DateOffset(months=1), color="gray", ls=":", lw=1)
        ax.axvline(t0 + pd.DateOffset(months=1), color="gray", ls=":", lw=1)
        ax.set_title(f"{r.name} (first hike {r['first_hike']})")
        ax.set_ylabel("FN30 CC spread to 5/10 (%)")
        ax.legend()
    fig.tight_layout()
    out = r"D:\python\kimi_helpers\fncc105_fed_hike_output.png"
    fig.savefig(out, dpi=150)
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()
