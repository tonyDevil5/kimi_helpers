r"""
FX Forward Monitor (multi-pair)
===============================
Interactive monitor of an FX pair's forward curve across monthly tenors
(1M..12M), built from Bloomberg spot and forward points. Supported pairs:
USD/HKD, USD/CNH, USD/JPY and EUR/USD (quoted as EURUSD per market
convention).

Data (probed live on Bloomberg; all confirmed working):
    Spot:            USDHKD Curncy / USDCNH Curncy / USDJPY Curncy / EURUSD Curncy
    Forward points:  {CCY}{n}M BGN Curncy (PX_LAST), n = 1..12
                     ({CCY} = HKD / CNH / JPY / EUR; "12M", not "1Y").
    Pip scale:       HKD 0.0001, CNH 0.0001, JPY 0.01, EUR 0.0001
                     (outright = spot + points * pip scale).

History is cached per pair in fx_<pair>_output/history/<pair>_history.csv;
reruns only fetch the delta since the last cached date (7-day overlap).

Report content:
    - Summary table: one row per tenor (Spot, 1M..12M): current outright,
      forward points (pips), annualized carry for the forward buyer
      ((spot - outright)/spot, annualized), vol 20d, and 250d
      min/median/max, percentile rank, z-score computed on the POINTS series.
      The Spot row has no points series, so its range stats are dashed out.
    - Current forward-points curve vs 250d min/max band and median (pips).
    - History chart per tenor (dropdown): forward points with rolling 250d
      mean +/-1/2 sigma bands; Spot shows the spot outright.
    - "Forward Curve - Spreads" section: 1M leg (2M-1M..12M-11M) and 3M leg
      (4M-1M..12M-9M, including standard 6M-3M, 9M-6M, 12M-9M).
    - "Forward Flies" section: 1M leg (1M2M3M..10M11M12M) and 3M leg
      (1M4M7M..6M9M12M, including standard 3M6M9M, 6M9M12M).

Run:
    python fx_forward_monitor.py USDHKD        (or USDCNH / USDJPY / EURUSD / ALL)

Output:
    ./fx_<pair>_output/<PAIR>_Forward_Monitor_<latest data date>.html
"""

from __future__ import annotations

import html
import os
import sys
from dataclasses import dataclass
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
@dataclass(frozen=True)
class PairCfg:
    key: str            # e.g. "USDHKD"
    pair: str           # display label, e.g. "USD/HKD"
    base: str           # base currency of the quote (first)
    terms: str          # terms currency of the quote (second)
    spot_ticker: str
    points_ccy: str     # currency prefix of the forward-points tickers
    pip_scale: float    # points -> outright units
    output_dir: Path
    quote_note: str = ""

    @property
    def points_tickers(self) -> Dict[int, str]:
        return {n: f"{self.points_ccy}{n}M BGN Curncy" for n in TENOR_MONTHS}

    @property
    def all_tickers(self) -> List[str]:
        return [self.spot_ticker] + [self.points_tickers[n] for n in TENOR_MONTHS]

    @property
    def title(self) -> str:
        return f"{self.pair} Forward Monitor"

    @property
    def methodology(self) -> str:
        return (
            f"Outright = spot + forward points * {self.pip_scale:g} ({self.points_ccy} pips), "
            f"Bloomberg {self.spot_ticker} + {self.points_ccy}{{n}}M BGN Curncy"
        )

    @property
    def cache_csv(self) -> Path:
        return self.output_dir / "history" / f"{self.key.lower()}_history.csv"

    @property
    def slug(self) -> str:
        return self.key.lower()


TENOR_MONTHS = list(range(1, 13))

_ROOT = Path(__file__).parent

PAIRS: Dict[str, PairCfg] = {
    "USDHKD": PairCfg("USDHKD", "USD/HKD", "USD", "HKD", "USDHKD Curncy", "HKD",
                      1e-4, _ROOT / "fx_usdhkd_output"),
    "USDCNH": PairCfg("USDCNH", "USD/CNH", "USD", "CNH", "USDCNH Curncy", "CNH",
                      1e-4, _ROOT / "fx_usdcnh_output"),
    "USDJPY": PairCfg("USDJPY", "USD/JPY", "USD", "JPY", "USDJPY Curncy", "JPY",
                      1e-2, _ROOT / "fx_usdjpy_output"),
    "EURUSD": PairCfg("EURUSD", "EUR/USD", "EUR", "USD", "EURUSD Curncy", "EUR",
                      1e-4, _ROOT / "fx_eurusd_output",
                      quote_note="EUR is quoted as EURUSD per market convention "
                                 "(the pair is EUR/USD, not USD/EUR)."),
}

Z_WINDOW = 250          # trailing window for min/median/max, percentile, z-score
HIST_DAYS = 3 * 365 + 30  # calendar days of history to pull on a cold start/backfill
MIN_OBS = 60            # minimum observations for stats

TENOR_COLORS = {
    1: "#2563eb", 2: "#0891b2", 3: "#059669", 4: "#d97706",
    5: "#dc2626", 6: "#7c3aed", 7: "#db2777", 8: "#475569",
    9: "#0d9488", 10: "#65a30d", 11: "#9333ea", 12: "#111827",
}


# ---------------------------------------------------------------------------
# Bloomberg fetch + cache
# ---------------------------------------------------------------------------
def _to_wide(frame, cfg: PairCfg) -> pd.DataFrame:
    """Normalize a bdh result (narwhals long or wide MultiIndex) to date x ticker."""
    if not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    if {"ticker", "date", "value"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="ticker", values="value")
    elif isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.reindex(columns=cfg.all_tickers)
    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def _fetch_bdh(cfg: PairCfg, start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    print(f"[{cfg.key}] Bloomberg history request: {start} to {end} "
          f"({len(cfg.all_tickers)} tickers)", flush=True)
    frame = blp.bdh(tickers=cfg.all_tickers, flds=["PX_LAST"], start_date=start, end_date=end)
    frame = _to_wide(frame, cfg)
    if frame.dropna(how="all").empty:
        raise RuntimeError("Bloomberg returned no history; local cache was not modified.")
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=True, index_label="date",
                 date_format="%Y-%m-%d", float_format="%.10g")
    os.replace(tmp, path)


def load_history(cfg: PairCfg, offline: bool = False) -> pd.DataFrame:
    """Load cached wide history (date x ticker), fetching only the delta."""
    cache_csv = cfg.cache_csv
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame()
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv, parse_dates=["date"]).set_index("date").sort_index()
        cached = cached.reindex(columns=cfg.all_tickers)

    if not offline:
        today = pd.Timestamp.today().normalize()
        cold_start = (today - pd.Timedelta(days=HIST_DAYS)).date().isoformat()
        if cached.empty or cached.index.min() > pd.Timestamp(cold_start) + pd.Timedelta(days=7):
            # Cold start, or the cache begins after the desired history start:
            # one-time full backfill (merged into the cache below).
            start = cold_start
        else:
            start = max(
                pd.Timestamp(cold_start),
                cached.index.max() - pd.Timedelta(days=7),
            ).date().isoformat()
        fresh = _fetch_bdh(cfg, start, today.date().isoformat())
        if cached.empty:
            merged = fresh
        else:
            merged = fresh.combine_first(cached)
        merged = merged.sort_index()
        _atomic_csv(merged, cache_csv)
        print(f"[{cfg.key}] Cache updated: {merged.index[0].date()} to "
              f"{merged.index[-1].date()} ({len(merged)} rows)", flush=True)
        return merged

    if cached.empty:
        raise RuntimeError("No local history cache; run once with Bloomberg connected.")
    return cached


# ---------------------------------------------------------------------------
# Outrights and stats
# ---------------------------------------------------------------------------
def build_outrights(hist: pd.DataFrame, cfg: PairCfg) -> pd.DataFrame:
    """DataFrame of outright forwards (Spot + 1M..12M columns), fully populated rows."""
    out = pd.DataFrame(index=hist.index)
    out["Spot"] = hist[cfg.spot_ticker]
    for n in TENOR_MONTHS:
        out[f"{n}M"] = hist[cfg.spot_ticker] + hist[cfg.points_tickers[n]] * cfg.pip_scale
    return out.dropna(how="any")


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


def build_points(hist: pd.DataFrame, cfg: PairCfg) -> pd.DataFrame:
    """DataFrame of forward points in pips (1M..12M columns), fully populated rows."""
    out = pd.DataFrame(index=hist.index)
    for n in TENOR_MONTHS:
        out[f"{n}M"] = hist[cfg.points_tickers[n]]
    return out.dropna(how="any")


def build_stats(outrights: pd.DataFrame, points: pd.DataFrame,
                cfg: PairCfg) -> Dict[str, Dict[str, float]]:
    """Per-tenor stats. Range stats (min/median/max, pctile, z) are computed on the
    forward-points (pips) series; the Spot row has no points, so those are NaN."""
    stats: Dict[str, Dict[str, float]] = {}
    spot_now = float(outrights["Spot"].dropna().iloc[-1])
    for label in outrights.columns:
        s = outrights[label].dropna()
        row: Dict[str, float] = {
            "latest": float(s.iloc[-1]),
            "min_250d": np.nan,
            "median_250d": np.nan,
            "max_250d": np.nan,
            "pct_rank": np.nan,
            "z_250d": np.nan,
            "obs": len(s),
            "points": np.nan,
            "carry_ann": np.nan,
            "vol_20d": np.nan,
        }
        if label != "Spot":
            n = int(label[:-1])
            pts = points[label].dropna()
            recent = pts.tail(Z_WINDOW)
            row.update({
                "points": float(pts.iloc[-1]),
                "min_250d": float(recent.min()),
                "median_250d": float(recent.median()),
                "max_250d": float(recent.max()),
                "pct_rank": _pct_rank(pts, Z_WINDOW),
                "z_250d": _zscore(pts, Z_WINDOW),
                # Forward buyer's roll-down carry: (spot - outright)/spot annualized;
                # equals the base-terms rate differential on the quoted pair.
                "carry_ann": float(-pts.iloc[-1] * cfg.pip_scale / spot_now * (12.0 / n) * 100.0),
                "vol_20d": _vol20d(pts),
            })
        stats[label] = row
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


def fig_forward_curve(points: pd.DataFrame, cfg: PairCfg) -> go.Figure:
    """Current forward-points curve vs 250d min/max band and median (all in pips)."""
    months = TENOR_MONTHS
    labels = [f"{n}M" for n in TENOR_MONTHS]
    recent = points.tail(Z_WINDOW)
    current = [float(points[c].iloc[-1]) for c in labels]
    lo = [float(recent[c].min()) for c in labels]
    hi = [float(recent[c].max()) for c in labels]
    med = [float(recent[c].median()) for c in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months + months[::-1], y=hi + lo[::-1], fill="toself",
        fillcolor="rgba(148,163,184,0.25)", line=dict(width=0),
        name="250d min/max band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=med, mode="lines", name="250d median",
        line=dict(color="#94a3b8", width=1.4, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=current, mode="lines+markers", name="Current curve",
        line=dict(color="#dc2626", width=2.2), marker=dict(size=7),
        hovertemplate="%{text}: %{y:+.1f} pips<extra>Current</extra>", text=labels,
    ))
    fig.update_xaxes(title_text="Tenor (months)", dtick=1)
    fig.update_yaxes(title_text="Forward points (pips)")
    as_of = points.index[-1].date()
    _apply_layout(
        fig, f"{cfg.pair} Forward Points Curve vs 250d Range",
        subtitle=f"As of {as_of}  |  band = trailing {Z_WINDOW}d min/max of each tenor's points (pips)",
    )
    return fig


def fig_history(label: str, s: pd.Series, unit: str, fmt: str,
                what: Optional[str] = None, pair: str = "") -> go.Figure:
    """History of a series (points in pips, or spot outright) with rolling
    250d mean +/-1/2 sigma bands."""
    s = s.dropna()
    mean = s.rolling(window=Z_WINDOW, min_periods=MIN_OBS).mean()
    std = s.rolling(window=Z_WINDOW, min_periods=MIN_OBS).std(ddof=1)
    m_now, sd_now = float(mean.dropna().iloc[-1]), float(std.dropna().iloc[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_xs(s.index) + _xs(s.index[::-1]),
        y=_ys(mean + 2 * std) + _ys((mean - 2 * std).iloc[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.10)", line=dict(width=0),
        name=f"±2σ: {m_now + 2 * sd_now:{fmt}} / {m_now - 2 * sd_now:{fmt}}",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index) + _xs(s.index[::-1]),
        y=_ys(mean + std) + _ys((mean - std).iloc[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.18)", line=dict(width=0),
        name=f"±1σ: {m_now + sd_now:{fmt}} / {m_now - sd_now:{fmt}}",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index), y=_ys(mean), mode="lines",
        name=f"Rolling {Z_WINDOW}d mean: {m_now:{fmt}}",
        line=dict(color="#94a3b8", width=1.3, dash="dash"),
    ))
    try:
        color = TENOR_COLORS.get(int(label[:-1]), "#2563eb") if label != "Spot" else "#111827"
    except ValueError:
        color = "#2563eb"
    fig.add_trace(go.Scatter(
        x=_xs(s.index), y=_ys(s), mode="lines", name=label,
        line=dict(color=color, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=_xs(s.index[-1:]), y=[float(s.iloc[-1])], mode="markers",
        marker=dict(color="#dc2626", size=8), showlegend=False, hoverinfo="skip",
    ))
    fig.update_yaxes(title_text=unit)
    if what is None:
        what = "Spot Outright" if label == "Spot" else f"{label} Forward Points"
    _apply_layout(
        fig, f"{pair} {what} - History vs Rolling {Z_WINDOW}d Bands",
        subtitle=f"Latest: {float(s.iloc[-1]):{fmt}}  |  "
                 f"σ = {sd_now:{fmt}}  |  "
                 f"+1σ/−1σ = {m_now + sd_now:{fmt}} / {m_now - sd_now:{fmt}}  |  "
                 f"+2σ/−2σ = {m_now + 2 * sd_now:{fmt}} / {m_now - 2 * sd_now:{fmt}}  |  "
                 f"{s.index[0].date()} to {s.index[-1].date()}",
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
        # white -> red
        r, g, b = 255, int(255 - 150 * t), int(255 - 150 * t)
    else:
        # white -> green
        r, g, b = int(255 + 150 * t), 255, int(255 + 150 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(x: float, fmt: str = "{:.4f}", na: str = "n/a") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _summary_table_html(stats: Dict[str, Dict[str, float]]) -> str:
    cols = ["Tenor", "Outright", "Fwd Points (pips)", "Ann. Carry (%, fwd buyer)",
            "Vol 20d (pips)",
            f"Min {Z_WINDOW}d (pips)", f"Median {Z_WINDOW}d (pips)",
            f"Max {Z_WINDOW}d (pips)", "Pctile Rank (%)", f"z{Z_WINDOW}d"]
    labels = ["Spot"] + [f"{n}M" for n in TENOR_MONTHS]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for label in labels:
        row = stats[label]
        values = {
            "Tenor": label,
            "Outright": _fmt(row["latest"]),
            "Fwd Points (pips)": _fmt(row["points"], "{:+.1f}", "—"),
            "Ann. Carry (%, fwd buyer)": _fmt(row["carry_ann"], "{:+.2f}", "—"),
            "Vol 20d (pips)": _fmt(row["vol_20d"], "{:.1f}", "—"),
            f"Min {Z_WINDOW}d (pips)": _fmt(row["min_250d"], "{:+.1f}", "—"),
            f"Median {Z_WINDOW}d (pips)": _fmt(row["median_250d"], "{:+.1f}", "—"),
            f"Max {Z_WINDOW}d (pips)": _fmt(row["max_250d"], "{:+.1f}", "—"),
            "Pctile Rank (%)": _fmt(row["pct_rank"], "{:.0f}", "—"),
            f"z{Z_WINDOW}d": _fmt(row["z_250d"], "{:+.2f}", "—"),
        }
        parts.append("<tr>")
        for c in cols:
            style = ""
            if c == f"z{Z_WINDOW}d":
                color = _z_color(row["z_250d"])
                if color:
                    style = f' style="background:{color}"'
            elif c == "Tenor":
                style = ' style="font-weight:bold"'
            parts.append(f"<td{style}>{html.escape(values[c])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Curve spreads and flies (on forward points, pips)
# ---------------------------------------------------------------------------
def _pts(points: pd.DataFrame, n: int) -> pd.Series:
    return points[f"{n}M"]


def build_curve_structures(points: pd.DataFrame) -> "Dict[str, pd.Series]":
    """Curve spreads in two legs: 1M-spaced adjacent (2M-1M..12M-11M) and
    3M-spaced (4M-1M..12M-9M). The standard 6M-3M, 9M-6M, 12M-9M structures
    are part of the 3M leg, not duplicated."""
    structs: Dict[str, pd.Series] = {}
    for n in range(1, 12):
        structs[f"{n + 1}M-{n}M"] = _pts(points, n + 1) - _pts(points, n)
    for n in range(1, 10):
        structs[f"{n + 3}M-{n}M"] = _pts(points, n + 3) - _pts(points, n)
    return structs


CURVE_LEG = {
    **{f"{n + 1}M-{n}M": "1M" for n in range(1, 12)},
    **{f"{n + 3}M-{n}M": "3M" for n in range(1, 10)},
}


def build_fly_structures(points: pd.DataFrame) -> "Dict[str, pd.Series]":
    """Flies in two legs: 1M-spaced (nM(n+1)M(n+2)M, n=1..10) and 3M-spaced
    (nM(n+3)M(n+6)M, n=1..6). Fly = 2*belly - wings. The standard 3M6M9M and
    6M9M12M structures are part of the 3M leg, not duplicated."""
    structs: Dict[str, pd.Series] = {}
    for n in range(1, 11):
        structs[f"{n}M{n + 1}M{n + 2}M"] = (
            2.0 * _pts(points, n + 1) - _pts(points, n) - _pts(points, n + 2)
        )
    for n in range(1, 7):
        structs[f"{n}M{n + 3}M{n + 6}M"] = (
            2.0 * _pts(points, n + 3) - _pts(points, n) - _pts(points, n + 6)
        )
    return structs


FLY_LEG = {
    **{f"{n}M{n + 1}M{n + 2}M": "1M" for n in range(1, 11)},
    **{f"{n}M{n + 3}M{n + 6}M": "3M" for n in range(1, 7)},
}


def build_structure_stats(structs: "Dict[str, pd.Series]") -> Dict[str, Dict[str, float]]:
    """Current value plus 250d min/median/max, pctile rank and z per structure."""
    stats: Dict[str, Dict[str, float]] = {}
    for name, series in structs.items():
        s = series.dropna()
        recent = s.tail(Z_WINDOW)
        stats[name] = {
            "latest": float(s.iloc[-1]),
            "min_250d": float(recent.min()),
            "median_250d": float(recent.median()),
            "max_250d": float(recent.max()),
            "pct_rank": _pct_rank(s, Z_WINDOW),
            "z_250d": _zscore(s, Z_WINDOW),
            "vol_20d": _vol20d(s),
        }
    return stats


def _structure_table_html(structs: "Dict[str, pd.Series]",
                          stats: Dict[str, Dict[str, float]],
                          leg_map: Optional[Dict[str, str]] = None) -> str:
    cols = (["Structure", "Leg"] if leg_map else ["Structure"]) + \
           ["Current (pips)", "Vol 20d (pips)",
            f"Min {Z_WINDOW}d (pips)", f"Median {Z_WINDOW}d (pips)",
            f"Max {Z_WINDOW}d (pips)", "Pctile Rank (%)", f"z{Z_WINDOW}d"]
    parts = ['<div class="table-wrap"><table class="data-table sortable-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for name in structs:
        row = stats[name]
        values = {
            "Structure": name,
            "Leg": leg_map[name] if leg_map else "",
            "Current (pips)": _fmt(row["latest"], "{:+.1f}"),
            "Vol 20d (pips)": _fmt(row["vol_20d"], "{:.1f}", "—"),
            f"Min {Z_WINDOW}d (pips)": _fmt(row["min_250d"], "{:+.1f}"),
            f"Median {Z_WINDOW}d (pips)": _fmt(row["median_250d"], "{:+.1f}"),
            f"Max {Z_WINDOW}d (pips)": _fmt(row["max_250d"], "{:+.1f}"),
            "Pctile Rank (%)": _fmt(row["pct_rank"], "{:.0f}"),
            f"z{Z_WINDOW}d": _fmt(row["z_250d"], "{:+.2f}"),
        }
        parts.append("<tr>")
        for c in cols:
            style = ""
            if c == f"z{Z_WINDOW}d":
                color = _z_color(row["z_250d"])
                if color:
                    style = f' style="background:{color}"'
            elif c == "Structure":
                style = ' style="font-weight:bold"'
            parts.append(f"<td{style}>{html.escape(values[c])}</td>")
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

_JS = """
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
document.querySelectorAll('.chart-always').forEach(el=>renderChart(el.id));
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text===''||text==='—'||text==='N.A.'||text==='n/a')return {blank:true,num:false,value:''};const n=Number(text.replace(/(%|bp)$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
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
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='fx-forward-monitor.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
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


def build_html(cfg: PairCfg, outrights: pd.DataFrame, points: pd.DataFrame,
               stats: Dict[str, Dict[str, float]],
               curve_structs: "Dict[str, pd.Series]",
               fly_structs: "Dict[str, pd.Series]",
               report_path: Path) -> Path:
    builder = _ReportBuilder()
    as_of = outrights.index[-1].date()

    quote_note_html = f'<p class="note">{html.escape(cfg.quote_note)}</p>' if cfg.quote_note else ""
    summary_section = (
        f'<section class="section"><h2>Summary - {html.escape(cfg.pair)} Spot and Forward Points</h2>'
        + _summary_table_html(stats)
        + f'<p class="note">Outright = spot + points × {cfg.pip_scale:g}. '
          f"Negative points = {cfg.terms} forward premium ({cfg.pair} forward below "
          f"spot), i.e. implied {cfg.base} rate above {cfg.terms} rate over the "
          "tenor (positive points: the reverse). Ann. Carry (%, fwd buyer) = "
          "(spot − outright)/spot annualized: positive = buying the forward earns "
          "this annualized roll if spot is unchanged; equals the "
          f"{cfg.base}−{cfg.terms} rate differential. Range stats, percentile rank and "
          f"z-score are computed on the forward-points (pips) series vs the trailing "
          f"{Z_WINDOW} trading days; z capped at ±3 for coloring. Vol 20d (pips) = "
          "std of daily point changes over the trailing 20 trading days. The Spot row has no "
          "points series, so its range stats are dashed out (its outright is shown "
          "for reference only).</p>"
        + quote_note_html
        + "</section>"
    )

    curve_section = (
        '<section class="section"><h2>Forward Points Curve vs 250d Range</h2>'
        + builder.fig_block(fig_forward_curve(points, cfg), always=True)
        + "</section>"
    )

    def _history_dropdown(group: str, series_map: "Dict[str, pd.Series]",
                          default: str, title_prefix: str) -> str:
        options = "".join(
            f'<option value="{html.escape(name)}"{" selected" if name == default else ""}>'
            f"{html.escape(name)}</option>"
            for name in series_map
        )
        block_parts = []
        for name, series in series_map.items():
            hidden = "" if name == default else ' style="display:none"'
            fig = fig_history(
                name, series, "pips", "+.1f",
                what=f"{title_prefix} {name}", pair=cfg.pair,
            )
            block_parts.append(
                f'<div class="struct-block" data-group="{group}" data-struct="{html.escape(name)}"{hidden}>'
                f"{builder.fig_block(fig)}"
                f"</div>"
            )
        return (
            f'<div class="dropdown-bar"><label>Structure: '
            f'<select class="struct-select" data-group="{group}">{options}</select></label></div>'
            + "".join(block_parts)
        )

    # Tenor history section (points per tenor; Spot shows the outright)
    tenor_labels = ["Spot"] + [f"{n}M" for n in TENOR_MONTHS]
    options = "".join(
        f'<option value="{lbl}"{" selected" if lbl == "12M" else ""}>{lbl}</option>'
        for lbl in tenor_labels
    )
    block_parts = []
    for lbl in tenor_labels:
        hidden = "" if lbl == "12M" else ' style="display:none"'
        if lbl == "Spot":
            fig = fig_history(lbl, outrights["Spot"], f"{cfg.pair} spot", ".4f",
                              pair=cfg.pair)
        else:
            fig = fig_history(lbl, points[lbl], "Forward points (pips)", "+.1f",
                              pair=cfg.pair)
        block_parts.append(
            f'<div class="struct-block" data-group="tenor" data-struct="{lbl}"{hidden}>'
            f"{builder.fig_block(fig)}"
            f"</div>"
        )
    history_section = (
        '<section class="section"><h2>Forward Points History vs Rolling 250d Bands</h2>'
        f'<div class="dropdown-bar"><label>Tenor: '
        f'<select class="struct-select" data-group="tenor">{options}</select></label></div>'
        + "".join(block_parts)
        + '<p class="note">Per-tenor history is shown in forward points (pips) with '
          f"rolling {Z_WINDOW}d mean ±1σ/±2σ bands; the Spot entry shows the spot "
          "outright itself (no points series exists for spot).</p></section>"
    )

    # Curve spreads section
    curve_stats = build_structure_stats(curve_structs)
    spread_section = (
        '<section class="section"><h2>Forward Curve — Spreads (points, pips)</h2>'
        + _structure_table_html(curve_structs, curve_stats, leg_map=CURVE_LEG)
        + _history_dropdown("curve", curve_structs, "2M-1M", "Curve Spread")
        + '<p class="note">Spread aM-bM = pts(aM) − pts(bM). A negative spread '
          "means the curve is inverted (more negative points at longer tenors). "
          "Leg \"1M\" = 11 adjacent monthly spreads (2M-1M..12M-11M); leg \"3M\" = "
          "nine 3-month-spaced spreads (4M-1M..12M-9M), which includes the "
          "standard 6M-3M, 9M-6M and 12M-9M structures. Vol 20d (pips) = std of "
          "daily spread changes over the trailing 20 trading days.</p></section>"
    )

    # Flies section
    fly_stats = build_structure_stats(fly_structs)
    fly_section = (
        '<section class="section"><h2>Forward Flies (points, pips)</h2>'
        + _structure_table_html(fly_structs, fly_stats, leg_map=FLY_LEG)
        + _history_dropdown("fly", fly_structs, "1M2M3M", "Fly")
        + '<p class="note">Fly aMbMcM = 2×pts(bM) − pts(aM) − pts(cM); a positive '
          "fly means the belly tenor's points are rich (less negative) vs the "
          "wings. Leg \"1M\" = ten monthly flies (1M2M3M..10M11M12M); leg \"3M\" = "
          "six 3-month-spaced flies (1M4M7M..6M9M12M), which includes the "
          "standard 3M6M9M and 6M9M12M structures. Vol 20d (pips) = std of "
          "daily fly changes over the trailing 20 trading days.</p></section>"
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg.title)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<nav class="header"><div class="tb-title">{html.escape(cfg.title)}</div>
<div class="tb-sub">as of {as_of}  |  {html.escape(cfg.methodology)}</div></nav>
<main>
{summary_section}
{curve_section}
{history_section}
{spread_section}
{fly_section}
</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_console_summary(cfg: PairCfg, outrights: pd.DataFrame,
                          stats: Dict[str, Dict[str, float]],
                          report_path: Path) -> None:
    labels = ["Spot"] + [f"{n}M" for n in TENOR_MONTHS]
    print("\n" + "=" * 96)
    print(f"{cfg.title} - summary (as of {outrights.index[-1].date()})")
    print("=" * 96)
    header = (f"{'Tenor':<7}{'Outright':>10}{'Points':>9}{'Carry%':>8}{'Vol20':>7}"
              f"{'Min250':>10}{'Med250':>10}{'Max250':>10}{'Pctile':>8}{'z250':>7}")
    print(header)
    print("-" * len(header))
    for lbl in labels:
        r = stats[lbl]
        print(
            f"{lbl:<7}{_fmt(r['latest']):>10}"
            f"{_fmt(r['points'], '{:+.1f}', '—'):>9}"
            f"{_fmt(r['carry_ann'], '{:+.2f}', '—'):>8}"
            f"{_fmt(r['vol_20d'], '{:.1f}', '—'):>7}"
            f"{_fmt(r['min_250d'], '{:+.1f}', '—'):>10}"
            f"{_fmt(r['median_250d'], '{:+.1f}', '—'):>10}"
            f"{_fmt(r['max_250d'], '{:+.1f}', '—'):>10}"
            f"{_fmt(r['pct_rank'], '{:.0f}', '—'):>8}"
            f"{_fmt(r['z_250d'], '{:+.2f}', '—'):>7}"
        )
    r12 = stats["12M"]
    print("\nHeadline:")
    print(f"  Spot {cfg.key}: {stats['Spot']['latest']:.4f}")
    print(f"  12M points: {r12['points']:+.1f} pips  "
          f"(250d range {r12['min_250d']:+.1f} to {r12['max_250d']:+.1f}, "
          f"median {r12['median_250d']:+.1f}, "
          f"pctile {_fmt(r12['pct_rank'], '{:.0f}')}%, z {_fmt(r12['z_250d'], '{:+.2f}')})")
    print(f"\nHTML report saved to: {report_path}")
    print("=" * 96)


def _print_structure_summary(title: str, structs: "Dict[str, pd.Series]") -> None:
    stats = build_structure_stats(structs)
    print(f"\n{title}:")
    for name, r in stats.items():
        print(f"  {name:<9} current {r['latest']:+8.1f} pips  "
              f"vol20 {_fmt(r['vol_20d'], '{:.1f}'):>5}  "
              f"(250d range {r['min_250d']:+.1f} to {r['max_250d']:+.1f}, "
              f"median {r['median_250d']:+.1f}, "
              f"pctile {_fmt(r['pct_rank'], '{:.0f}')}%, "
              f"z {_fmt(r['z_250d'], '{:+.2f}')})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def headline(cfg: PairCfg, outrights: pd.DataFrame, points: pd.DataFrame,
             stats: Dict[str, Dict[str, float]],
             curve_structs: "Dict[str, pd.Series]",
             fly_structs: "Dict[str, pd.Series]") -> str:
    """One-line headline: spot, 12M carry, most stretched structure by |z|."""
    spot = stats["Spot"]["latest"]
    carry12 = stats["12M"]["carry_ann"]
    structs = {**build_structure_stats(curve_structs), **build_structure_stats(fly_structs)}
    tenor_z = {f"{n}M points": stats[f"{n}M"]["z_250d"] for n in TENOR_MONTHS}
    all_z = {**tenor_z, **{k: v["z_250d"] for k, v in structs.items()}}
    all_z = {k: v for k, v in all_z.items() if pd.notna(v)}
    worst = max(all_z, key=lambda k: abs(all_z[k])) if all_z else None
    txt = (f"{cfg.pair}: spot {spot:.4f} | 12M carry (fwd buyer) {carry12:+.2f}%")
    if worst:
        txt += f" | most stretched: {worst} z {all_z[worst]:+.2f}"
    return txt


def run_pair(cfg: PairCfg, offline: bool = False) -> Path:
    hist = load_history(cfg, offline=offline)
    outrights = build_outrights(hist, cfg)
    if len(outrights) < MIN_OBS:
        raise RuntimeError(f"Only {len(outrights)} fully populated rows; need {MIN_OBS}.")

    print("Computing summary statistics...")
    points = build_points(hist, cfg)
    stats = build_stats(outrights, points, cfg)
    curve_structs = build_curve_structures(points)
    fly_structs = build_fly_structures(points)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    as_of = outrights.index[-1].strftime("%Y%m%d")
    report_path = cfg.output_dir / f"{cfg.key}_Forward_Monitor_{as_of}.html"

    print("Building interactive HTML report...")
    build_html(cfg, outrights, points, stats, curve_structs, fly_structs, report_path)

    print_console_summary(cfg, outrights, stats, report_path)
    _print_structure_summary("Curve spreads (standard)",
                             {k: curve_structs[k] for k in ("6M-3M", "9M-6M", "12M-9M")})
    _print_structure_summary("Flies (standard)",
                             {k: fly_structs[k] for k in ("3M6M9M", "6M9M12M")})
    print("HEADLINE: " + headline(cfg, outrights, points, stats,
                                  curve_structs, fly_structs))
    print("=" * 96)
    return report_path


def main(argv: Optional[List[str]] = None) -> List[Path]:
    argv = list(sys.argv[1:] if argv is None else argv)
    key = argv[0].upper() if argv else "USDHKD"
    keys = list(PAIRS) if key == "ALL" else [key]
    paths = []
    for k in keys:
        if k not in PAIRS:
            raise SystemExit(f"Unknown pair {k!r}; choose from {', '.join(PAIRS)} or ALL")
        paths.append(run_pair(PAIRS[k], offline="--offline" in argv))
    return paths


if __name__ == "__main__":
    main()
