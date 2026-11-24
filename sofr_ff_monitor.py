r"""
SOFR vs Fed Funds Futures Spread Monitor
========================================
Interactive monitor of the spread between 3M SOFR futures and 30-day Fed
Funds futures across the first 12 generic contract months.

Data (probed live on Bloomberg; all confirmed working):
    SOFR futures:    generic SER{n} Comdty, n = 1..12
                     (actual contract format SER<code><year-digit> Comdty,
                      e.g. SERZ6 = Dec-26; verified SER2 PX_LAST == SERZ6)
    FF futures:      generic FF{n} Comdty, n = 1..12
                     (actual contract format FF<code><year-digit> Comdty,
                      e.g. FFZ6 = Dec-26; verified FF2 PX_LAST == FFZ6)
    NOTE: "FFA{n} Comdty" is NOT the Fed Funds contract - FFA is the
    WTI Midland (Argus) financial crude future. Use FF{n}.

Slot definition: slot n = the CURRENT calendar month + (n-1), mapped
directly to generic SER{n}/FF{n} (user confirmed generic 1 = the
current-month contract is the intended slot 1; as of 2026-11-23 slot 1 =
Nov-26 = SERX6/FFX6, slot 12 = Oct-27 = SERV7/FFV7). Slot labels are read
live from FUT_CONTRACT_DT; if that probe is unavailable (e.g. --offline),
labels fall back to current month + (n-1) computed from the run date.

History choice: the generic tickers carry continuous 3y+ history on
Bloomberg, while specific contracts (SERZ6 etc.) only have history for the
life of that contract (~18 months). The monitor therefore uses the generic
SER{n}/FF{n} series for both live prices and history. The history of a
slot is the history of the nth contract on the curve and rolls monthly;
the spread stats are curve-slot stats, not per-calendar-month stats. This
is standard for generic-futures monitors and is documented in the report
footer.

Analytics:
    Futures price = 100 - rate, so implied SOFR rate - FF rate equals
    FF price - SOFR price. spread_bp(n) = (FF{n} px - SER{n} px) * 100.

Report content:
    - Summary table: one row per ROLLING generic slot (labeled SER1..SER12,
      not calendar months - each slot rolls monthly): SOFR px, FF px, implied
      rates (100 - px), spread (bp, with DoD change), 250d min/median/max,
      percentile rank, z-score, 20d realized daily vol (bp/day). The current
      actual-contract mapping is shown as secondary info.
    - Spread curve across the 12 rolling slots vs 250d min/max band and
      median.
    - History section with a Series toggle: "Rolling slot" (default; generic
      SER{n}/FF{n} history) or "Actual contract" (the listed pair, e.g.
      SERX6/FFX6, life-of-contract history), each with a per-slot/contract
      dropdown and rolling 250d mean +/-1/2 sigma bands.

History is cached in sofr_ff_monitor_output/history/sofr_ff_history.csv;
reruns only fetch the delta since the last cached date (7-day overlap).

Run:
    python sofr_ff_monitor.py            (live Bloomberg)
    python sofr_ff_monitor.py --offline  (use cache only)

Output:
    ./sofr_ff_monitor_output/SOFR_FF_Monitor_<latest data date>.html
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
OUTPUT_DIR = _ROOT / "sofr_ff_monitor_output"
CACHE_CSV = OUTPUT_DIR / "history" / "sofr_ff_history.csv"
ACTUALS_CACHE_CSV = OUTPUT_DIR / "history" / "sofr_ff_actuals_history.csv"

MONTHS = list(range(1, 13))
# Slot n = current calendar month + (n-1) = generic SER{n}/FF{n} (user
# confirmed generic 1 = the current-month contract is the intended slot 1).
SER_TICKERS = {n: f"SER{n} Comdty" for n in MONTHS}
FF_TICKERS = {n: f"FF{n} Comdty" for n in MONTHS}
ALL_TICKERS = [SER_TICKERS[n] for n in MONTHS] + [FF_TICKERS[n] for n in MONTHS]

_MONTH_CODES = "FGHJKMNQUVXZ"


def actual_ticker(root: str, month_ts: pd.Timestamp) -> str:
    """Actual contract ticker, e.g. SERZ6 for Dec-26 (decade rollover on the
    year digit: Jan-27 -> F7)."""
    return f"{root}{_MONTH_CODES[month_ts.month - 1]}{month_ts.year % 10} Comdty"


def actual_tickers(months: Dict[int, pd.Timestamp]) -> Dict[int, "tuple[str, str]"]:
    """Slot n -> (actual SER ticker, actual FF ticker) for its contract month."""
    return {n: (actual_ticker("SER", ts), actual_ticker("FF", ts))
            for n, ts in months.items()}


def _fallback_slot_months(run_date: Optional[pd.Timestamp] = None) -> Dict[int, pd.Timestamp]:
    """Slot n -> contract month: current calendar month + (n-1)."""
    base = (run_date or pd.Timestamp.today()).normalize().replace(day=1)
    return {n: base + pd.DateOffset(months=n - 1) for n in MONTHS}


def fetch_slot_months() -> Dict[int, pd.Timestamp]:
    """Slot months read live from FUT_CONTRACT_DT; falls back to current
    month + (n-1) from the run date when the probe is unavailable."""
    try:
        from xbbg import blp
        frame = blp.bdp([SER_TICKERS[n] for n in MONTHS], "FUT_CONTRACT_DT")
        if not isinstance(frame, pd.DataFrame):
            frame = frame.to_pandas()
        inv = {v: n for n, v in SER_TICKERS.items()}
        months: Dict[int, pd.Timestamp] = {}
        for _, row in frame.iterrows():
            n = inv[row["ticker"]]
            try:
                m, y = str(row["value"]).split("/")  # e.g. "11/2026"
                months[n] = pd.Timestamp(int(y), int(m), 1)
            except ValueError:
                pass
        if len(months) == len(MONTHS):
            return months
    except Exception as exc:  # Bloomberg unavailable; use the fallback
        print(f"FUT_CONTRACT_DT probe failed ({exc}); using run-date fallback.")
    return _fallback_slot_months()


def slot_labels(months: Dict[int, pd.Timestamp]) -> Dict[int, str]:
    return {n: ts.strftime("%b-%y") for n, ts in months.items()}


def _maturity_window(month_ts: pd.Timestamp,
                     s: pd.Series) -> "tuple[pd.Timestamp, pd.Timestamp]":
    """Display x-window: the 12 months leading up to the contract's maturity.
    The right edge clips at the last data date (today) when maturity is in
    the future, avoiding dead space on the chart."""
    x0 = month_ts - pd.DateOffset(months=12)
    maturity = month_ts + pd.offsets.MonthEnd(0)
    x1 = min(maturity, s.dropna().index[-1]) if len(s.dropna()) else maturity
    return x0, x1

Z_WINDOW = 250                 # trailing window for min/median/max, pctile, z
HIST_DAYS = 3 * 365 + 30       # calendar days of history on cold start/backfill
MIN_OBS = 60                   # minimum observations for rolling stats

TITLE = "SOFR vs Fed Funds Futures Spread Monitor"
METHODOLOGY = ("spread(bp) = (FF{n} px - SER{n} px) x 100, n = 1..12 generic "
               "contracts (Bloomberg FF{n} Comdty / SER{n} Comdty; "
               "generic 1 = current-month contract)")

SLOT_COLORS = {
    1: "#2563eb", 2: "#0891b2", 3: "#059669", 4: "#d97706",
    5: "#dc2626", 6: "#7c3aed", 7: "#db2777", 8: "#475569",
    9: "#0d9488", 10: "#65a30d", 11: "#9333ea", 12: "#111827",
}


# ---------------------------------------------------------------------------
# Bloomberg fetch + cache
# ---------------------------------------------------------------------------
def _to_wide(frame, tickers: List[str]) -> pd.DataFrame:
    """Normalize a bdh result (narwhals long or wide MultiIndex) to date x ticker."""
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    if {"ticker", "date", "value"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="ticker", values="value")
    elif isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reindex(columns=tickers)
    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def _fetch_bdh(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    print(f"Bloomberg history request: {start} to {end} "
          f"({len(tickers)} tickers)", flush=True)
    frame = blp.bdh(tickers=tickers, flds=["PX_LAST"],
                    start_date=start, end_date=end)
    frame = _to_wide(frame, tickers)
    if frame.dropna(how="all").empty:
        raise RuntimeError("Bloomberg returned no history; local cache was not modified.")
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=True, index_label="date",
                 date_format="%Y-%m-%d", float_format="%.10g")
    os.replace(tmp, path)


def _load_cached(tickers: List[str], cache_csv: Path,
                 min_obs: Optional[int], offline: bool) -> pd.DataFrame:
    """Load cached wide history (date x ticker), fetching only the delta.
    min_obs: if set, a required column with fewer valid observations triggers
    a one-time full backfill (ticker schema change). If None, only fully
    missing columns trigger the backfill (used for actual contracts, whose
    life-of-contract history is legitimately short)."""
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame()
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv, parse_dates=["date"]).set_index("date").sort_index()
        cached = cached.reindex(columns=tickers)

    if not offline:
        today = pd.Timestamp.today().normalize()
        cold_start = (today - pd.Timedelta(days=HIST_DAYS)).date().isoformat()
        counts = cached.notna().sum()
        short = (counts < min_obs) if min_obs is not None else (counts == 0)
        if (cached.empty
                or cached.index.min() > pd.Timestamp(cold_start) + pd.Timedelta(days=7)
                or short.any()):
            # Cold start, cache begins after the desired history start, or a
            # required ticker is missing/short in the cache (e.g. ticker
            # schema change): one-time full backfill (merged below).
            start = cold_start
        else:
            start = max(
                pd.Timestamp(cold_start),
                cached.index.max() - pd.Timedelta(days=7),
            ).date().isoformat()
        fresh = _fetch_bdh(tickers, start, today.date().isoformat())
        merged = fresh if cached.empty else fresh.combine_first(cached)
        merged = merged.sort_index()
        _atomic_csv(merged, cache_csv)
        print(f"Cache updated: {cache_csv.name}: {merged.index[0].date()} to "
              f"{merged.index[-1].date()} ({len(merged)} rows)", flush=True)
        return merged

    if cached.empty:
        raise RuntimeError(f"No local history cache ({cache_csv.name}); "
                           "run once with Bloomberg connected.")
    return cached


def load_history(offline: bool = False) -> pd.DataFrame:
    """Rolling-slot (generic SER{n}/FF{n}) history."""
    return _load_cached(ALL_TICKERS, CACHE_CSV, MIN_OBS, offline)


def load_actual_history(months: Dict[int, pd.Timestamp],
                        offline: bool = False) -> pd.DataFrame:
    """Life-of-contract history for the actual contract pair of each slot.
    Short series are expected (SER ~18 months, FF less), so no minimum
    coverage is enforced beyond the column being present."""
    tickers = [t for pair in actual_tickers(months).values() for t in pair]
    return _load_cached(tickers, ACTUALS_CACHE_CSV, None, offline)


# ---------------------------------------------------------------------------
# Spread series and stats
# ---------------------------------------------------------------------------
def build_spreads(hist: pd.DataFrame) -> pd.DataFrame:
    """Spread (bp) per slot n: (FF{n} px - SER{n} px) * 100."""
    out = pd.DataFrame(index=hist.index)
    for n in MONTHS:
        out[n] = (hist[FF_TICKERS[n]] - hist[SER_TICKERS[n]]) * 100.0
    # No global dropna: a far slot can lag a session; each slot's stats use
    # its own latest valid values.
    return out


def build_actual_spreads(hist: pd.DataFrame,
                         months: Dict[int, pd.Timestamp]) -> Dict[int, pd.Series]:
    """Per-contract spread (bp): (actual FF px - actual SER px) * 100, on the
    pair's common trading days (life-of-contract, so short for front FF)."""
    out: Dict[int, pd.Series] = {}
    for n, (ser_tk, ff_tk) in actual_tickers(months).items():
        pair = pd.concat([hist[ser_tk], hist[ff_tk]], axis=1).dropna()
        out[n] = (pair.iloc[:, 1] - pair.iloc[:, 0]) * 100.0
    return out


def _zscore(s: pd.Series, window: int) -> float:
    recent = s.dropna().tail(window)
    if len(recent) < 2:
        return np.nan
    std = recent.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float((s.dropna().iloc[-1] - recent.mean()) / std)


def _pct_rank(s: pd.Series, window: int) -> float:
    recent = s.dropna().tail(window)
    if len(recent) < 2:
        return np.nan
    return float((recent <= recent.iloc[-1]).mean() * 100.0)


def _vol20d(s: pd.Series) -> float:
    """Realized daily volatility: std of daily changes over the trailing 20 obs."""
    diffs = s.dropna().diff().dropna().tail(20)
    if len(diffs) < 2:
        return np.nan
    return float(diffs.std(ddof=1))


def _dod(s: pd.Series) -> float:
    """Day-over-day change of the latest observation."""
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-2])


def build_stats(hist: pd.DataFrame, spreads: pd.DataFrame,
                months: Dict[int, pd.Timestamp]) -> Dict[str, Dict[str, float]]:
    """Per-rolling-slot stats (keyed by generic name SER1..SER12); range stats
    computed on the generic spread (bp) series. The current contract month is
    kept per row as secondary (actual-contract mapping) info."""
    stats: Dict[str, Dict[str, float]] = {}
    for n in MONTHS:
        ser = hist[SER_TICKERS[n]].dropna()
        ff = hist[FF_TICKERS[n]].dropna()
        s = spreads[n].dropna()
        recent = s.tail(Z_WINDOW)
        stats[f"SER{n}"] = {
            "slot": n,
            "month_ts": months[n],
            "ser_px": float(ser.iloc[-1]),
            "ff_px": float(ff.iloc[-1]),
            "ser_rate": 100.0 - float(ser.iloc[-1]),
            "ff_rate": 100.0 - float(ff.iloc[-1]),
            "spread": float(s.iloc[-1]),
            "spread_dod": _dod(s),
            "min_250d": float(recent.min()),
            "median_250d": float(recent.median()),
            "max_250d": float(recent.max()),
            "pct_rank": _pct_rank(s, Z_WINDOW),
            "z_250d": _zscore(s, Z_WINDOW),
            "vol_20d": _vol20d(s),
            "obs": len(s),
        }
    return stats


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: go.Figure, title: str, height: int = 460,
                  subtitle: Optional[str] = None) -> go.Figure:
    title_text = title.replace("\n", "<br>")
    if subtitle:
        title_text += f'<br><span style="font-size:12px;color:#64748b">{subtitle}</span>'
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(font=dict(size=11)),
    )
    return fig


def fig_spread_curve(spreads: pd.DataFrame,
                     months: Dict[int, pd.Timestamp]) -> go.Figure:
    """Current rolling-slot spread curve (generic SER1..SER12 on the x axis,
    current contract-month mapping in hover) vs 250d min/max band + median."""
    labels = [f"SER{n}" for n in MONTHS]
    hover = [f"SER{n}/FF{n} (currently {months[n].strftime('%b-%y')})" for n in MONTHS]
    recent = spreads.tail(Z_WINDOW)
    current = [float(spreads[n].dropna().iloc[-1]) for n in MONTHS]
    lo = [float(recent[n].min()) for n in MONTHS]
    hi = [float(recent[n].max()) for n in MONTHS]
    med = [float(recent[n].median()) for n in MONTHS]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1], y=hi + lo[::-1], fill="toself",
        fillcolor="rgba(148,163,184,0.25)", line=dict(width=0),
        name="250d min/max band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=med, mode="lines", name="250d median",
        line=dict(color="#94a3b8", width=1.4, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=current, mode="lines+markers", name="Current curve",
        line=dict(color="#dc2626", width=2.2), marker=dict(size=7),
        hovertemplate="%{text}: %{y:.1f} bp<extra>Current</extra>", text=hover,
    ))
    fig.update_xaxes(title_text="Rolling slot (generic SER{n} / FF{n})")
    fig.update_yaxes(title_text="SOFR - FF spread (bp)")
    as_of = max(spreads[n].dropna().index[-1] for n in MONTHS).date()
    mapping = ", ".join(f"SER{n}={months[n].strftime('%b-%y')}" for n in (1, 12))
    _apply_layout(
        fig, "SOFR - Fed Funds Spread Curve vs 250d Range (Rolling Slots)",
        subtitle=f"As of {as_of}  |  band = trailing {Z_WINDOW}d min/max of each "
                 "generic slot's spread (bp); slots roll monthly with the "
                 f"contracts (currently {mapping})",
    )
    return fig


def fig_history(label: str, s: pd.Series, color: str,
                note: Optional[str] = None,
                x_window: "Optional[tuple[pd.Timestamp, pd.Timestamp]]" = None) -> go.Figure:
    """History of a spread series (bp) with rolling 250d mean +/-1/2 sigma
    bands; latest sigma/band values are shown in the legend and subtitle.
    Series shorter than MIN_OBS get no bands and an insufficient-sample
    note; shorter than Z_WINDOW get bands on available history, flagged
    partial via note. Bands are always computed on the FULL history;
    x_window (x0, x1) only restricts the displayed x range (the 12 months
    leading up to the contract's maturity)."""
    s = s.dropna()
    mean = s.rolling(window=Z_WINDOW, min_periods=MIN_OBS).mean()
    std = s.rolling(window=Z_WINDOW, min_periods=MIN_OBS).std(ddof=1)
    fig = go.Figure()
    have_bands = len(mean.dropna()) > 0 and len(std.dropna()) > 0
    if have_bands:
        m_now, sd_now = float(mean.dropna().iloc[-1]), float(std.dropna().iloc[-1])
        fig.add_trace(go.Scatter(
            x=_xs(s.index) + _xs(s.index[::-1]),
            y=_ys(mean + 2 * std) + _ys((mean - 2 * std).iloc[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.10)", line=dict(width=0),
            name=f"±2σ: {m_now + 2 * sd_now:.1f} / {m_now - 2 * sd_now:.1f}",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=_xs(s.index) + _xs(s.index[::-1]),
            y=_ys(mean + std) + _ys((mean - std).iloc[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.18)", line=dict(width=0),
            name=f"±1σ: {m_now + sd_now:.1f} / {m_now - sd_now:.1f}",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=_xs(s.index), y=_ys(mean), mode="lines",
            name=f"Rolling {Z_WINDOW}d mean: {m_now:.1f}",
            line=dict(color="#94a3b8", width=1.3, dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index), y=_ys(s), mode="lines", name=label,
        line=dict(color=color, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index[-1:]), y=[float(s.iloc[-1])], mode="markers",
        marker=dict(color="#dc2626", size=8), showlegend=False, hoverinfo="skip",
    ))
    fig.update_yaxes(title_text="SOFR - FF spread (bp)")
    if x_window is not None:
        x0, x1 = x_window
        fig.update_xaxes(range=[x0.strftime("%Y-%m-%d"), x1.strftime("%Y-%m-%d")])
    if have_bands:
        stats_txt = (f"σ = {sd_now:.1f}  |  "
                     f"+1σ/−1σ = {m_now + sd_now:.1f} / {m_now - sd_now:.1f}  |  "
                     f"+2σ/−2σ = {m_now + 2 * sd_now:.1f} / {m_now - 2 * sd_now:.1f}  |  ")
    else:
        stats_txt = (f"insufficient sample for rolling stats "
                     f"({len(s)} obs < {MIN_OBS})  |  ")
    note_txt = f"{note}  |  " if note else ""
    _apply_layout(
        fig, f"{label} SOFR - FF Spread - History vs Rolling {Z_WINDOW}d Bands",
        subtitle=f"Latest: {float(s.iloc[-1]):.1f} bp  |  {stats_txt}{note_txt}"
                 f"{s.index[0].date()} to {s.index[-1].date()} ({len(s)} obs)",
    )
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _z_color(z: float) -> Optional[str]:
    """Diverging green-white-red shading for z-scores, capped at +/-3."""
    if pd.isna(z):
        return None
    t = max(-1.0, min(1.0, float(z) / 3.0))
    if t >= 0:
        r, g, b = 255, int(255 - 150 * t), int(255 - 150 * t)
    else:
        r, g, b = int(255 + 150 * t), 255, int(255 + 150 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(x: float, fmt: str = "{:.4f}", na: str = "n/a") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _with_dod(value: float, dod: float, fmt: str, dod_fmt: str, na: str = "—") -> str:
    """Leading value plus a small colored day-over-day change in parentheses."""
    base = html.escape(_fmt(value, fmt, na))
    if pd.isna(dod):
        return base
    color = "#059669" if dod > 0 else "#dc2626" if dod < 0 else "#64748b"
    return (f'{base} <span style="color:{color};font-size:10px">'
            f'({dod:+{dod_fmt}})</span>')


def _summary_table_html(stats: Dict[str, Dict[str, float]]) -> str:
    cols = ["Slot (generic)", "Actual contract (current)", "SOFR Px", "FF Px",
            "Impl SOFR %", "Impl FF %",
            "Spread (bp)", "Vol 20d (bp/d)",
            f"Min {Z_WINDOW}d (bp)", f"Median {Z_WINDOW}d (bp)",
            f"Max {Z_WINDOW}d (bp)", "Pctile Rank (%)", f"z{Z_WINDOW}d"]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for label, row in stats.items():
        values = {
            "Slot (generic)": html.escape(f"{label} / FF{row['slot']}"),
            "Actual contract (current)": html.escape(
                f"{actual_ticker('SER', row['month_ts']).replace(' Comdty', '')}"
                f" / {actual_ticker('FF', row['month_ts']).replace(' Comdty', '')}"
                f"  ({row['month_ts'].strftime('%b-%y')})"),
            "SOFR Px": html.escape(f"{row['ser_px']:.4f}".rstrip("0").rstrip(".")),
            "FF Px": html.escape(f"{row['ff_px']:.4f}".rstrip("0").rstrip(".")),
            "Impl SOFR %": html.escape(f"{row['ser_rate']:.3f}"),
            "Impl FF %": html.escape(f"{row['ff_rate']:.3f}"),
            "Spread (bp)": _with_dod(row["spread"], row["spread_dod"], "{:.1f}", ".1f"),
            "Vol 20d (bp/d)": html.escape(_fmt(row["vol_20d"], "{:.1f}", "—")),
            f"Min {Z_WINDOW}d (bp)": html.escape(_fmt(row["min_250d"], "{:.1f}", "—")),
            f"Median {Z_WINDOW}d (bp)": html.escape(_fmt(row["median_250d"], "{:.1f}", "—")),
            f"Max {Z_WINDOW}d (bp)": html.escape(_fmt(row["max_250d"], "{:.1f}", "—")),
            "Pctile Rank (%)": html.escape(_fmt(row["pct_rank"], "{:.0f}", "—")),
            f"z{Z_WINDOW}d": html.escape(_fmt(row["z_250d"], "{:+.2f}", "—")),
        }
        parts.append("<tr>")
        for c in cols:
            style = ""
            if c == f"z{Z_WINDOW}d":
                color = _z_color(row["z_250d"])
                if color:
                    style = f' style="background:{color}"'
            elif c == "Slot (generic)":
                style = ' style="font-weight:bold"'
            parts.append(f"<td{style}>{values[c]}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--panel:#fff;--bg:#eef2f6;--accent:#174a6e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
.header{position:sticky;top:0;z-index:100;background:#102b3c;color:#e7f0f5;padding:10px 18px;
box-shadow:0 2px 8px rgba(15,23,42,.25)}
.header .tb-title{font-size:16px;font-weight:bold}
.header .tb-sub{font-size:12px;color:#b9cad5;margin-top:2px}
main{padding:20px;max-width:1500px;margin:0 auto}
.section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
box-shadow:0 3px 12px rgba(15,23,42,.06);padding:16px 18px;margin-bottom:20px}
.section h2{font-size:17px;margin:2px 0 12px;color:var(--accent)}
.chart{width:100%;margin-bottom:18px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px;margin-bottom:12px}
.table-tools{display:flex;gap:8px;align-items:center;margin:2px 0 6px;font-size:12px}
.table-tools input[type=search]{padding:5px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:12px;min-width:180px}
.table-tools button{border:1px solid #b6c5cf;background:#fff;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px}
.table-count{color:var(--muted)}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 8px;font-weight:bold;text-align:center;white-space:nowrap}
.sortable-table th{cursor:pointer;user-select:none}
.sortable-table th.sort-asc::after{content:" ▲";font-size:9px}
.sortable-table th.sort-desc::after{content:" ▼";font-size:9px}
.data-table td{border:1px solid #e2e8f0;padding:4px 8px;text-align:center;white-space:nowrap}
.note{font-size:12px;color:var(--muted);font-style:italic}
.dropdown-bar{margin:2px 0 14px;font-size:13px}
.dropdown-bar select{padding:6px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:13px;min-width:180px;background:#fff}
@media print{.header{position:static}.section{break-after:page}}
"""

_JS = r"""
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
function renderBlock(block){if(!block)return;block.querySelectorAll('.chart').forEach(el=>renderChart(el.id));}
function showStruct(group,slug){
document.querySelectorAll('.struct-block[data-group="'+group+'"]').forEach(bl=>{
const on=bl.dataset.struct===slug;bl.style.display=on?'':'none';if(on)renderBlock(bl);
});
}
document.querySelectorAll('select.struct-select').forEach(sel=>{
sel.addEventListener('change',()=>showStruct(sel.dataset.group,sel.value));
showStruct(sel.dataset.group,sel.value);
});
document.querySelectorAll('select.series-select').forEach(sel=>{
sel.addEventListener('change',()=>{
document.querySelectorAll('.series-pane').forEach(p=>{
const on=p.dataset.series===sel.value;p.style.display=on?'':'none';
if(on)p.querySelectorAll('.struct-block').forEach(bl=>{if(bl.style.display!=='none')renderBlock(bl);});
});
});
});
document.querySelectorAll('.chart-always').forEach(el=>renderChart(el.id));
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text===''||text==='—'||text==='N.A.'||text==='n/a')return {blank:true,num:false,value:''};const lead=text.match(/^[-+]?\d*\.?\d+/);if(lead)return {blank:false,num:true,value:Number(lead[0])};const n=Number(text.replace(/(%|bp)$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
function csvCell(value){return '"'+value.replace(/"/g,'""')+'"';}
document.querySelectorAll('.sortable-table').forEach((table)=>{
const wrap=table.closest('.table-wrap');if(!wrap)return;const rows=[...table.tBodies[0].rows];const tools=document.createElement('div');tools.className='table-tools';
const search=document.createElement('input');search.type='search';search.placeholder='Filter this table';search.setAttribute('aria-label','Filter this table');
const reset=document.createElement('button');reset.type='button';reset.textContent='Reset sort';
const csv=document.createElement('button');csv.type='button';csv.textContent='Download CSV';
const count=document.createElement('span');count.className='table-count';tools.append(search,reset,csv,count);wrap.parentNode.insertBefore(tools,wrap);
const refresh=()=>{let shown=0;rows.forEach(row=>{if(row.style.display!=='none')shown++;});count.textContent=`${shown} of ${rows.length} rows`;};refresh();
search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();rows.forEach(row=>{row.style.display=(!q||row.textContent.toLowerCase().includes(q))?'':'none';});refresh();});
table.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{const asc=!th.classList.contains('sort-asc');table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));th.classList.add(asc?'sort-asc':'sort-desc');const sorted=[...rows].sort((a,b)=>{const av=cellValue(a.cells[col]),bv=cellValue(b.cells[col]);if(av.blank!==bv.blank)return av.blank?1:-1;let c;if(av.num&&bv.num)c=av.value-bv.value;else c=String(av.value).localeCompare(String(bv.value));return asc?c:-c;});sorted.forEach(row=>table.tBodies[0].appendChild(row));}));
reset.addEventListener('click',()=>{rows.forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));});
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='sofr-ff-monitor.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
});
"""


class _ReportBuilder:
    """Collects figures as JSON and emits self-contained HTML blocks."""

    def __init__(self) -> None:
        self._fig_count = 0

    def fig_block(self, fig: go.Figure, always: bool = False) -> str:
        self._fig_count += 1
        fid = f"fig-{self._fig_count:03d}"
        fig_json = pio.to_json(fig).replace("</", "<\\/")
        cls = "chart chart-always" if always else "chart"
        return (
            f'<div class="{cls}" id="{fid}"></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{fig_json}</script>"
        )


def build_html(spreads: pd.DataFrame, stats: Dict[str, Dict[str, float]],
               months: Dict[int, pd.Timestamp],
               actual_spreads: Dict[int, pd.Series],
               report_path: Path) -> Path:
    builder = _ReportBuilder()
    month_labels = slot_labels(months)
    as_of = max(spreads[n].dropna().index[-1] for n in MONTHS).date()

    summary_section = (
        '<section class="section"><h2>Summary - SOFR vs Fed Funds, Rolling Slots (generic)</h2>'
        + _summary_table_html(stats)
        + f'<p class="note">Both legs quote as 100 − rate, so the implied '
          'SOFR − FF spread equals FF price − SOFR price, shown in bp. Rows are '
          'ROLLING generic slots SER{n}/FF{n} (the nth contract on the curve; '
          'generic 1 = the current-month contract) — not fixed calendar '
          'months: each slot rolls monthly, and the "Actual contract '
          '(current)" column shows which listed pair the slot currently maps '
          'to (read live from FUT_CONTRACT_DT; fallback: computed from the '
          'run date). History uses the continuous generic series, so each '
          'row\'s stats are curve-slot stats that roll monthly with the '
          'contracts. Range '
          'stats, '
          f'percentile rank and z-score are vs the trailing {Z_WINDOW} trading '
          'days; z capped at ±3 for coloring. Vol 20d (bp/d) = std of daily '
          'spread changes over the trailing 20 trading days. DoD change is '
          'shown in parentheses next to the current spread.</p>'
        + "</section>"
    )

    curve_section = (
        '<section class="section"><h2>Spread Curve vs 250d Range (Rolling Slots)</h2>'
        + builder.fig_block(fig_spread_curve(spreads, months), always=True)
        + "</section>"
    )

    # Rolling pane: labeled by generic slot name (SER1..SER12), NOT calendar
    # months - the series is a rolling front-series.
    slot_names = [f"SER{n}" for n in MONTHS]
    month_labels = slot_labels(months)
    labels = [f"{month_labels[n]}" for n in MONTHS]
    default_label = slot_names[-1]
    options = "".join(
        f'<option value="{lbl}"{" selected" if lbl == default_label else ""}>{lbl}</option>'
        for lbl in slot_names
    )
    block_parts = []
    for n, lbl in zip(MONTHS, slot_names):
        hidden = "" if lbl == default_label else ' style="display:none"'
        fig = fig_history(
            lbl, spreads[n], SLOT_COLORS.get(n, "#2563eb"),
            note=f"rolling generic slot SER{n}/FF{n} "
                 f"(currently {month_labels[n]})",
            x_window=_maturity_window(months[n], spreads[n]))
        block_parts.append(
            f'<div class="struct-block" data-group="month" data-struct="{lbl}"{hidden}>'
            f"{builder.fig_block(fig)}"
            f"</div>"
        )
    rolling_pane = (
        f'<div class="dropdown-bar"><label>Slot: '
        f'<select class="struct-select" data-group="month">{options}</select></label></div>'
        + "".join(block_parts)
        + f'<p class="note">Rolling slot series: generic SER{{n}}/FF{{n}} '
          '(the nth contract on the curve), which rolls monthly — labels are '
          'generic slot names, not calendar months; the current contract '
          'month is shown in each chart subtitle. Stats vs '
          f'the trailing {Z_WINDOW} trading days.</p>'
    )

    act_tks = actual_tickers(months)
    # Actual pane: labeled by contract month + listed tickers (fixed contracts).
    act_labels = [f"{month_labels[n]} ({act_tks[n][0].replace(' Comdty', '')})"
                  for n in MONTHS]
    act_default = act_labels[-1]
    act_options = "".join(
        f'<option value="{lbl}"{" selected" if lbl == act_default else ""}>{lbl}</option>'
        for lbl in act_labels
    )
    act_block_parts = []
    for n, lbl in zip(MONTHS, act_labels):
        hidden = "" if lbl == act_default else ' style="display:none"'
        s = actual_spreads[n].dropna()
        ser_tk, ff_tk = (t.replace(" Comdty", "") for t in act_tks[n])
        obs = len(s)
        if obs < MIN_OBS:
            note = f"actual contract {ser_tk}/{ff_tk}, insufficient sample"
        elif obs < Z_WINDOW:
            note = (f"actual contract {ser_tk}/{ff_tk}, partial sample "
                    f"(life-of-contract < {Z_WINDOW} obs)")
        else:
            note = f"actual contract {ser_tk}/{ff_tk}"
        fig = fig_history(lbl, s, SLOT_COLORS.get(n, "#2563eb"), note=note,
                          x_window=_maturity_window(months[n], s))
        act_block_parts.append(
            f'<div class="struct-block" data-group="actual" data-struct="{lbl}"{hidden}>'
            f"{builder.fig_block(fig)}"
            f"</div>"
        )
    actual_pane = (
        f'<div class="dropdown-bar"><label>Contract: '
        f'<select class="struct-select" data-group="actual">{act_options}</select></label></div>'
        + "".join(act_block_parts)
        + '<p class="note">Actual contract series: the life-of-contract spread '
          'of the specific SER/FF pair for each slot\'s contract month '
          '(e.g. SERX6/FFX6). History is limited by contract listing (SER '
          f'~18 months, front FF much less); rolling {Z_WINDOW}d bands are '
          'computed on whatever history exists and are flagged partial when '
          f'under {Z_WINDOW} obs, insufficient under {MIN_OBS} obs.</p>'
    )

    history_section = (
        '<section class="section"><h2>Spread History vs Rolling 250d Bands</h2>'
        '<div class="dropdown-bar"><label>Series: '
        '<select class="series-select">'
        '<option value="rolling" selected>Rolling slot</option>'
        '<option value="actual">Actual contract</option>'
        '</select></label></div>'
        f'<div class="series-pane" data-series="rolling">{rolling_pane}</div>'
        f'<div class="series-pane" data-series="actual" style="display:none">'
        f'{actual_pane}</div>'
        + f'<p class="note">Spread history (bp) with rolling {Z_WINDOW}d '
          'mean ±1σ/±2σ bands. "Rolling slot" (default) = generic curve slots '
          'SER1..SER12 (rolling front-series, rolls monthly); '
          '"Actual contract" = the specific listed SER/FF pair for each '
          'slot\'s current contract month. In both views the displayed '
          'x-window is the 12 months leading up to the contract\'s maturity '
          '(right edge clips at the last data date); the bands are still '
          f'computed on the full {Z_WINDOW}d rolling history, so a band\'s '
          'early portion may be cut off by the window.</p></section>'
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(TITLE)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<nav class="header"><div class="tb-title">{html.escape(TITLE)}</div>
<div class="tb-sub">as of {as_of}  |  {html.escape(METHODOLOGY)}</div></nav>
<main>
{summary_section}
{curve_section}
{history_section}
</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_console_summary(spreads: pd.DataFrame,
                          stats: Dict[str, Dict[str, float]],
                          report_path: Path) -> None:
    print("\n" + "=" * 96)
    print(f"{TITLE} - summary (as of "
          f"{max(spreads[n].dropna().index[-1] for n in MONTHS).date()})")
    print("=" * 96)
    header = (f"{'Slot':<16}{'SOFR px':>10}{'FF px':>10}{'Spread':>9}{'DoD':>7}"
              f"{'Vol20':>7}{'Min250':>9}{'Med250':>9}{'Max250':>9}"
              f"{'Pctile':>8}{'z250':>7}")
    print(header)
    print("-" * len(header))
    for label, r in stats.items():
        slot_txt = f"{label} ({r['month_ts'].strftime('%b-%y')})"
        print(
            f"{slot_txt:<16}{r['ser_px']:>10.4f}{r['ff_px']:>10.4f}"
            f"{r['spread']:>9.1f}{_fmt(r['spread_dod'], '{:+.1f}', '—'):>7}"
            f"{_fmt(r['vol_20d'], '{:.1f}', '—'):>7}"
            f"{_fmt(r['min_250d'], '{:.1f}', '—'):>9}"
            f"{_fmt(r['median_250d'], '{:.1f}', '—'):>9}"
            f"{_fmt(r['max_250d'], '{:.1f}', '—'):>9}"
            f"{_fmt(r['pct_rank'], '{:.0f}', '—'):>8}"
            f"{_fmt(r['z_250d'], '{:+.2f}', '—'):>7}"
        )
    print(f"\nHTML report saved to: {report_path}")
    print("=" * 96)


def headline(stats: Dict[str, Dict[str, float]]) -> str:
    """One-line headline: the rolling slot with the most extreme |z|."""
    valid = {k: v["z_250d"] for k, v in stats.items() if pd.notna(v["z_250d"])}
    if not valid:
        return "no valid z-scores"
    worst = max(valid, key=lambda k: abs(valid[k]))
    r = stats[worst]
    return (f"most extreme: {worst} (currently "
            f"{r['month_ts'].strftime('%b-%y')}) spread {r['spread']:.1f} bp, "
            f"z {r['z_250d']:+.2f} (250d median {r['median_250d']:.1f} bp, "
            f"range {r['min_250d']:.1f} to {r['max_250d']:.1f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(offline: bool = False) -> Path:
    months = (_fallback_slot_months() if offline else fetch_slot_months())
    month_labels = slot_labels(months)
    hist = load_history(offline=offline)
    spreads = build_spreads(hist)
    if spreads.notna().sum().min() < MIN_OBS:
        raise RuntimeError(
            f"Only {spreads.notna().sum().min()} valid rows in the worst slot; "
            f"need {MIN_OBS}.")

    print("Computing summary statistics...")
    stats = build_stats(hist, spreads, months)

    print("Fetching actual-contract history...")
    actual_hist = load_actual_history(months, offline=offline)
    actual_spreads = build_actual_spreads(actual_hist, months)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    as_of = max(spreads[n].dropna().index[-1] for n in MONTHS).strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"SOFR_FF_Monitor_{as_of}.html"

    print("Building interactive HTML report...")
    build_html(spreads, stats, months, actual_spreads, report_path)

    print_console_summary(spreads, stats, report_path)
    print("\nActual-contract history coverage (life-of-contract):")
    for n in MONTHS:
        ser_tk, ff_tk = (t.replace(" Comdty", "") for t in actual_tickers(months)[n])
        s = actual_spreads[n].dropna()
        tag = "" if len(s) >= Z_WINDOW else (
            "  [partial <250 obs]" if len(s) >= MIN_OBS else "  [insufficient]")
        rng = f"{s.index[0].date()} to {s.index[-1].date()}" if len(s) else "n/a"
        print(f"  {month_labels[n]:<7} {ser_tk}/{ff_tk:<13} {len(s):>4} obs  {rng}{tag}")
    print("HEADLINE: " + headline(stats))
    print("=" * 96)
    return report_path


def main(argv: Optional[List[str]] = None) -> Path:
    argv = list(sys.argv[1:] if argv is None else argv)
    return run(offline="--offline" in argv)


if __name__ == "__main__":
    main()
