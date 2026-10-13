r"""
Sovereign & MBS Technical Monitor
=================================
Interactive technical-analysis monitor for benchmark government yields at
2Y/5Y/10Y/20Y/30Y across five countries — US (UST), France (OAT), Italy (BTP),
Germany (Bund) and Japan (JGB) — plus MBS current-coupon OAS (FN and G2).

1. Pulls generic government yield tickers (GT* Govt family) and the
   current-coupon OAS indices (.FNCC105 / .G2CC105 G Index) from Bloomberg
   via xbbg. CC OAS tickers are quoted in percent and converted to bp
   (x100), matching mbs_rv_report.load_current_coupon_oas.
2. Computes per series SMAs, Bollinger Bands, Wilder RSI(14), realized vol
   of daily changes, and Fibonacci retracements off the trailing
   252-trading-day swing high/low. Sovereigns are shown in %, MBS in bp.
3. Writes a single self-contained interactive HTML file with a two-level
   toggle bar (country row incl. MBS / tenor row).

Run:
    python ust_technical_monitor.py
    python ust_technical_monitor.py --start-date 2015-01-01

Output:
    ./ust_technical_output/UST_Technical_Monitor_<latest data date>.html
"""

from __future__ import annotations

import argparse
import html
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "ust_technical_output"
REPORT_TITLE = "Sovereign & MBS Technical Monitor"

START_DATE = "2010-01-01"

TENORS = ["2Y", "5Y", "10Y", "20Y", "30Y"]
MBS_SERIES = ["FN CC", "G2 CC"]

# country code -> series label -> ticker
COUNTRY_TICKERS: Dict[str, Dict[str, str]] = {
    "ust": {t: f"GT{t[:-1]} Govt" for t in TENORS},
    "frf": {t: f"GTFRF{t[:-1]}Y Govt" for t in TENORS},
    "btp": {t: f"GTITL{t[:-1]}Y Govt" for t in TENORS},
    "bund": {t: f"GTDEM{t[:-1]}Y Govt" for t in TENORS},
    "jgb": {t: f"GTJPY{t[:-1]}Y Govt" for t in TENORS},
    "mbs": {"FN CC": ".FNCC105 G Index", "G2 CC": ".G2CC105 G Index"},
}
COUNTRY_SERIES_ORDER: Dict[str, List[str]] = {
    **{c: TENORS for c in ("ust", "frf", "btp", "bund", "jgb")},
    "mbs": MBS_SERIES,
}
COUNTRY_LABELS = {"ust": "UST", "frf": "FRF", "btp": "BTP", "bund": "Bund", "jgb": "JGB", "mbs": "MBS"}
COUNTRY_NAMES = {
    "ust": "UST (US)",
    "frf": "OAT (France)",
    "btp": "BTP (Italy)",
    "bund": "Bund (Germany)",
    "jgb": "JGB (Japan)",
    "mbs": "MBS",
}
COUNTRY_SHORT = {"ust": "UST", "frf": "OAT", "btp": "BTP", "bund": "Bund", "jgb": "JGB", "mbs": "MBS"}
COUNTRY_COLORS = {
    "ust": "#dc2626",
    "frf": "#2563eb",
    "btp": "#059669",
    "bund": "#d97706",
    "jgb": "#7c3aed",
}
COUNTRY_UNITS = {**{c: "%" for c in ("ust", "frf", "btp", "bund", "jgb")}, "mbs": "bp"}

SMA_WINDOWS = [20, 50, 100, 200]
BOLL_WINDOW = 20
BOLL_NSTD = 2.0
RSI_PERIOD = 14
VOL_WINDOWS = [20, 60]
FIB_WINDOW = 252
FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
RECENT_YEARS = 2  # x-range for the RSI / vol panels
MIN_OBS = 30  # below this a series is treated as unavailable

TENOR_COLORS = {
    "2Y": "#2563eb",
    "5Y": "#059669",
    "10Y": "#dc2626",
    "20Y": "#d97706",
    "30Y": "#7c3aed",
}
MBS_COLORS = {"FN CC": "#2563eb", "G2 CC": "#dc2626"}
SMA_COLORS = {20: "#f59e0b", 50: "#10b981", 100: "#ef4444", 200: "#8b5cf6"}
FIB_COLOR = "#9333ea"


def _series_color(country: str, name: str) -> str:
    return MBS_COLORS[name] if country == "mbs" else TENOR_COLORS[name]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_bdh(df, tickers=None) -> pd.DataFrame:
    """Normalize xbbg bdh output to a wide pandas DataFrame.

    xbbg >= 1.x returns a narwhals DataFrame in long format
    (ticker / date / field / value); older versions return a pandas
    DataFrame with MultiIndex columns.
    """
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(
            index="date", columns="ticker", values="value", aggfunc="last"
        )
        if tickers is not None:
            df = df.reindex(columns=[t for t in tickers if t in df.columns])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_yields(
    start_date: str = START_DATE,
) -> Tuple[Dict[str, Dict[str, Optional[pd.Series]]], List[str]]:
    """Pull all country x series data; return per-country series dicts.

    Each series is cleaned independently (dropna) — no cross-country or
    cross-series alignment. MBS CC OAS tickers are quoted in percent and
    converted to bp. Tickers that fail to resolve or are too sparse come
    back as None and are also listed in the returned warnings.
    """
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    tickers = [t for ct in COUNTRY_TICKERS.values() for t in ct.values()]
    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers from Bloomberg...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)
    df = _normalize_bdh(df, tickers=tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    data: Dict[str, Dict[str, Optional[pd.Series]]] = {}
    warnings_: List[str] = []
    for country, series_tickers in COUNTRY_TICKERS.items():
        data[country] = {}
        for name, ticker in series_tickers.items():
            if ticker not in df.columns:
                data[country][name] = None
                warnings_.append(f"{COUNTRY_NAMES[country]} {name}: {ticker} did not resolve")
                continue
            s = pd.to_numeric(df[ticker], errors="coerce").dropna()
            if len(s) < MIN_OBS:
                data[country][name] = None
                warnings_.append(
                    f"{COUNTRY_NAMES[country]} {name}: {ticker} has only {len(s)} observations"
                )
                continue
            if COUNTRY_UNITS[country] == "bp":
                s = s * 100.0  # CC OAS quoted in percent -> bp
            data[country][name] = s

    n_ok = sum(1 for ct in data.values() for s in ct.values() if s is not None)
    print(f"Loaded {n_ok}/{len(tickers)} series")
    for w in warnings_:
        print(f"  WARNING: {w}")
    return data, warnings_


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_smas(s: pd.Series) -> Dict[int, pd.Series]:
    return {w: s.rolling(window=w, min_periods=w).mean() for w in SMA_WINDOWS}


def compute_bollinger(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (mid, upper, lower, pct_b) for a 20d +/- 2 sigma band."""
    mid = s.rolling(window=BOLL_WINDOW, min_periods=BOLL_WINDOW).mean()
    sd = s.rolling(window=BOLL_WINDOW, min_periods=BOLL_WINDOW).std()
    upper = mid + BOLL_NSTD * sd
    lower = mid - BOLL_NSTD * sd
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def compute_rsi(s: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Zero average loss over the window => uninterrupted gains => RSI 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def compute_realized_vol(chg: pd.Series) -> Dict[int, pd.Series]:
    """Annualized vol of daily changes (already in display units, bp)."""
    return {
        w: chg.rolling(window=w, min_periods=w).std() * np.sqrt(252.0)
        for w in VOL_WINDOWS
    }


def compute_fib_levels(s: pd.Series) -> Dict[str, object]:
    """Fibonacci retracements off the trailing 252-trading-day swing."""
    recent = s.dropna().tail(FIB_WINDOW)
    swing_high = float(recent.max())
    swing_low = float(recent.min())
    high_date = recent.idxmax()
    low_date = recent.idxmin()
    levels = {r: swing_low + r * (swing_high - swing_low) for r in FIB_RATIOS}
    return {
        "high": swing_high,
        "low": swing_low,
        "high_date": high_date,
        "low_date": low_date,
        "levels": levels,
        "window_start": recent.index[0],
    }


def fib_zone_text(fib: Dict[str, object], latest: float) -> str:
    """Describe where the latest value sits among the retracement levels."""
    levels: Dict[float, float] = fib["levels"]  # type: ignore[assignment]
    ordered = sorted(levels.items(), key=lambda kv: kv[1])  # ascending value
    if latest > ordered[-1][1]:
        return (
            f"above 23.6% ({ordered[-1][1]:.2f}) - near/above the "
            f"{FIB_WINDOW}d swing high ({fib['high']:.2f})"
        )
    if latest < ordered[0][1]:
        return (
            f"below 78.6% ({ordered[0][1]:.2f}) - near/below the "
            f"{FIB_WINDOW}d swing low ({fib['low']:.2f})"
        )
    for (r_lo, v_lo), (r_hi, v_hi) in zip(ordered, ordered[1:]):
        if v_lo <= latest <= v_hi:
            return (
                f"between {r_hi * 100:.1f}% ({v_hi:.2f}) and "
                f"{r_lo * 100:.1f}% ({v_lo:.2f}) from the {FIB_WINDOW}d swing"
            )
    return "n/a"


# ---------------------------------------------------------------------------
# Per-series summary stats
# ---------------------------------------------------------------------------
def _chg(s: pd.Series, days: int, factor: float) -> float:
    if len(s) <= days:
        return np.nan
    return float((s.iloc[-1] - s.iloc[-1 - days]) * factor)


def _dist(s: pd.Series, ma: pd.Series, factor: float) -> float:
    ma_last = ma.dropna()
    if ma_last.empty:
        return np.nan
    return float((s.iloc[-1] - ma_last.iloc[-1]) * factor)


def _last_valid(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.iloc[-1]) if not v.empty else np.nan


def build_series_stats(s: pd.Series, in_bp: bool = False) -> Dict[str, object]:
    """All per-series indicator series plus the latest snapshot values.

    in_bp: series already in bp (MBS OAS); otherwise a % yield whose changes
    and distances are converted to bp (x100).
    """
    factor = 1.0 if in_bp else 100.0
    smas = compute_smas(s)
    boll_mid, boll_up, boll_lo, pct_b = compute_bollinger(s)
    rsi = compute_rsi(s)
    vols = compute_realized_vol(s.diff() * factor)
    fib = compute_fib_levels(s)

    latest = float(s.iloc[-1])
    stats: Dict[str, object] = {
        "series": s,
        "in_bp": in_bp,
        "smas": smas,
        "boll_mid": boll_mid,
        "boll_up": boll_up,
        "boll_lo": boll_lo,
        "pct_b": pct_b,
        "rsi": rsi,
        "vols": vols,
        "fib": fib,
        "latest": latest,
        "last_date": s.index[-1],
        "chg_1d": _chg(s, 1, factor),
        "chg_5d": _chg(s, 5, factor),
        "chg_20d": _chg(s, 20, factor),
        "chg_60d": _chg(s, 60, factor),
        "rsi_last": _last_valid(rsi),
        "pct_b_last": _last_valid(pct_b),
        "vol20_last": _last_valid(vols[20]),
        "vol60_last": _last_valid(vols[60]),
        "fib_zone": fib_zone_text(fib, latest),
    }
    for w in SMA_WINDOWS:
        stats[f"dist_sma{w}"] = _dist(s, smas[w], factor)
    return stats


# ---------------------------------------------------------------------------
# Small HTML / formatting helpers
# ---------------------------------------------------------------------------
def _fmt(x: float, fmt: str = "{:.2f}", na: str = "n/a") -> str:
    return fmt.format(x) if pd.notna(x) else na


def _fmt_bp(x: float) -> str:
    return _fmt(x, "{:+.1f}")


def _latest_fmt(stats: Dict[str, object]) -> str:
    return "{:.1f}" if stats["in_bp"] else "{:.3f}"


def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: go.Figure, title: str, height: int = 520, subtitle: Optional[str] = None) -> go.Figure:
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


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _fig_series_main(label: str, color: str, stats: Dict[str, object]) -> go.Figure:
    s: pd.Series = stats["series"]  # type: ignore[assignment]
    in_bp = bool(stats["in_bp"])
    smas: Dict[int, pd.Series] = stats["smas"]  # type: ignore[assignment]
    boll_mid: pd.Series = stats["boll_mid"]  # type: ignore[assignment]
    boll_up: pd.Series = stats["boll_up"]  # type: ignore[assignment]
    boll_lo: pd.Series = stats["boll_lo"]  # type: ignore[assignment]
    fib: Dict[str, object] = stats["fib"]  # type: ignore[assignment]
    unit_name = "OAS" if in_bp else "Yield"
    unit_label = "OAS (bp)" if in_bp else "Yield (%)"

    boll_legend = (
        f"Bollinger {BOLL_WINDOW}d ±{BOLL_NSTD:g}σ "
        f"(U {_fmt(_last_valid(boll_up))} / M {_fmt(_last_valid(boll_mid))} / "
        f"L {_fmt(_last_valid(boll_lo))})"
    )

    fig = go.Figure()

    # Bollinger band as a shaded fill between lower and upper
    fig.add_trace(
        go.Scatter(
            x=_xs(boll_lo.index), y=_ys(boll_lo), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_xs(boll_up.index), y=_ys(boll_up), mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.18)",
            name=boll_legend, hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=_xs(s.index), y=_ys(s), mode="lines", name=f"{label} {unit_name.lower()}",
            line=dict(color=color, width=1.4),
        )
    )
    for w in SMA_WINDOWS:
        fig.add_trace(
            go.Scatter(
                x=_xs(smas[w].index), y=_ys(smas[w]), mode="lines", name=f"SMA{w}",
                line=dict(color=SMA_COLORS[w], width=1.1),
            )
        )

    # Swing high / low markers over the fib window
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(fib["high_date"]).strftime("%Y-%m-%d")], y=[float(fib["high"])],
            mode="markers", name=f"{FIB_WINDOW}d swing high",
            marker=dict(color="#dc2626", size=10, symbol="triangle-down",
                        line=dict(color="black", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(fib["low_date"]).strftime("%Y-%m-%d")], y=[float(fib["low"])],
            mode="markers", name=f"{FIB_WINDOW}d swing low",
            marker=dict(color="#059669", size=10, symbol="triangle-up",
                        line=dict(color="black", width=1)),
        )
    )

    # Fibonacci retracement levels over the trailing window only
    x0 = pd.Timestamp(fib["window_start"]).strftime("%Y-%m-%d")
    x1 = pd.Timestamp(s.index[-1]).strftime("%Y-%m-%d")
    label_x = (pd.Timestamp(s.index[-1]) + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    levels: Dict[float, float] = fib["levels"]  # type: ignore[assignment]
    for ratio, value in levels.items():
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=value, y1=value,
            line=dict(color=FIB_COLOR, width=1, dash="dash"), opacity=0.65,
        )
        fig.add_annotation(
            x=label_x, y=value, text=f"{ratio * 100:.1f}%: {value:.2f}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=10, color=FIB_COLOR),
        )
    # Leave room for the right-edge fib labels
    x_max = (pd.Timestamp(s.index[-1]) + pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[_xs(s.index[:1])[0], x_max])

    fig.update_yaxes(title_text=unit_label)
    _apply_layout(
        fig,
        f"{label} {unit_name}: Moving Averages, Bollinger Bands, Fibonacci",
        height=560,
        subtitle=(
            f"Fib levels off trailing {FIB_WINDOW}d swing "
            f"(low {fib['low']:.2f} on {pd.Timestamp(fib['low_date']).date()} / "
            f"high {fib['high']:.2f} on {pd.Timestamp(fib['high_date']).date()})"
        ),
    )
    return fig


def _recent_range(s: pd.Series) -> Tuple[str, str]:
    end = pd.Timestamp(s.index[-1])
    start = end - pd.Timedelta(days=int(RECENT_YEARS * 365.25))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fig_series_rsi(label: str, color: str, stats: Dict[str, object]) -> go.Figure:
    rsi: pd.Series = stats["rsi"]  # type: ignore[assignment]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_xs(rsi.index), y=_ys(rsi), mode="lines", name=f"RSI({RSI_PERIOD})",
            line=dict(color=color, width=1.3),
        )
    )
    x0, x1 = _recent_range(rsi)
    for level, line_color in ((70, "#dc2626"), (50, "#64748b"), (30, "#059669")):
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=level, y1=level,
            line=dict(color=line_color, width=1, dash="dash"),
        )
    fig.add_annotation(
        x=x1, y=70, text="70", showarrow=False, xanchor="left",
        font=dict(size=10, color="#dc2626"),
    )
    fig.add_annotation(
        x=x1, y=30, text="30", showarrow=False, xanchor="left",
        font=dict(size=10, color="#059669"),
    )
    fig.update_xaxes(range=[x0, (pd.Timestamp(x1) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")])
    fig.update_yaxes(title_text=f"RSI({RSI_PERIOD})", range=[0, 100])
    _apply_layout(
        fig, f"{label} RSI({RSI_PERIOD}, Wilder)", height=340,
        subtitle=f"Last ~{RECENT_YEARS} years  |  Latest: {_fmt(stats['rsi_last'], '{:.1f}')}",
    )
    return fig


def _fig_series_vol(label: str, stats: Dict[str, object]) -> go.Figure:
    vols: Dict[int, pd.Series] = stats["vols"]  # type: ignore[assignment]
    s: pd.Series = stats["series"]  # type: ignore[assignment]
    unit_name = "OAS" if stats["in_bp"] else "Yield"
    fig = go.Figure()
    colors = {20: "#dc2626", 60: "#2563eb"}
    for w in VOL_WINDOWS:
        fig.add_trace(
            go.Scatter(
                x=_xs(vols[w].index), y=_ys(vols[w]), mode="lines", name=f"{w}d vol",
                line=dict(color=colors[w], width=1.3),
            )
        )
    x0, x1 = _recent_range(s)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(title_text="Realized vol (bp/yr)")
    _apply_layout(
        fig, f"{label} Realized Vol of Daily {unit_name} Changes", height=340,
        subtitle=(
            f"Last ~{RECENT_YEARS} years  |  std of daily change in bp x sqrt(252)  |  "
            f"Latest 20d: {_fmt(stats['vol20_last'], '{:.0f}')} bp/yr, "
            f"60d: {_fmt(stats['vol60_last'], '{:.0f}')} bp/yr"
        ),
    )
    return fig


def _fig_overview_10y(all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]]) -> go.Figure:
    fig = go.Figure()
    for country in COUNTRY_TICKERS:
        if country == "mbs":
            continue
        stats = all_stats[country].get("10Y")
        if stats is None:
            continue
        s: pd.Series = stats["series"]  # type: ignore[assignment]
        fig.add_trace(
            go.Scatter(
                x=_xs(s.index), y=_ys(s), mode="lines", name=COUNTRY_SHORT[country],
                line=dict(color=COUNTRY_COLORS[country], width=1.3),
            )
        )
    fig.update_yaxes(title_text="Yield (%)")
    _apply_layout(fig, "10Y Benchmark Yields - UST / OAT / BTP / Bund / JGB", height=560)
    return fig


def _fig_overview_mbs(all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]]) -> Optional[go.Figure]:
    fig = go.Figure()
    n = 0
    for name in MBS_SERIES:
        stats = all_stats["mbs"].get(name)
        if stats is None:
            continue
        s: pd.Series = stats["series"]  # type: ignore[assignment]
        fig.add_trace(
            go.Scatter(
                x=_xs(s.index), y=_ys(s), mode="lines", name=f"{name} OAS",
                line=dict(color=MBS_COLORS[name], width=1.3),
            )
        )
        n += 1
    if n == 0:
        return None
    fig.update_yaxes(title_text="OAS (bp)")
    _apply_layout(fig, "MBS Current-Coupon OAS - FN vs G2", height=520)
    return fig


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def _series_strip_html(stats: Dict[str, object]) -> str:
    fib: Dict[str, object] = stats["fib"]  # type: ignore[assignment]
    in_bp = bool(stats["in_bp"])
    latest_label = "Latest OAS" if in_bp else "Latest yield"
    latest_val = (
        f"{_fmt(stats['latest'], '{:.1f}')} bp" if in_bp
        else f"{_fmt(stats['latest'], '{:.3f}')}%"
    )
    cells = [
        (latest_label, latest_val),
        ("Chg 1d (bp)", _fmt_bp(stats["chg_1d"])),
        ("Chg 5d (bp)", _fmt_bp(stats["chg_5d"])),
        ("Chg 20d (bp)", _fmt_bp(stats["chg_20d"])),
        ("Chg 60d (bp)", _fmt_bp(stats["chg_60d"])),
        (f"RSI({RSI_PERIOD})", _fmt(stats["rsi_last"], "{:.1f}")),
        ("%B", _fmt(stats["pct_b_last"], "{:.2f}")),
        ("vs SMA20 (bp)", _fmt_bp(stats["dist_sma20"])),
        ("vs SMA50 (bp)", _fmt_bp(stats["dist_sma50"])),
        ("vs SMA100 (bp)", _fmt_bp(stats["dist_sma100"])),
        ("vs SMA200 (bp)", _fmt_bp(stats["dist_sma200"])),
        ("Vol 20d (bp/yr)", _fmt(stats["vol20_last"], "{:.0f}")),
        ("Vol 60d (bp/yr)", _fmt(stats["vol60_last"], "{:.0f}")),
    ]
    header = "".join(f"<th>{html.escape(k)}</th>" for k, _ in cells)
    row = "".join(f"<td>{html.escape(v)}</td>" for _, v in cells)
    return (
        f'<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{header}</tr></thead><tbody><tr>{row}</tr></tbody></table></div>"
        f'<p class="fib-note">Fibonacci zone ({FIB_WINDOW}d swing '
        f'low {fib["low"]:.2f} / high {fib["high"]:.2f}): '
        f'{html.escape(stats["fib_zone"])}</p>'
    )


def _overview_matrix_html(all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]]) -> str:
    parts = ['<div class="table-wrap"><table class="data-table"><thead><tr><th>Latest (%)</th>']
    parts += [f"<th>{t}</th>" for t in TENORS]
    parts.append("</tr></thead><tbody>")
    for country in COUNTRY_TICKERS:
        if country == "mbs":
            continue
        parts.append(f"<tr><td><b>{html.escape(COUNTRY_NAMES[country])}</b></td>")
        for tenor in TENORS:
            stats = all_stats[country].get(tenor)
            val = _fmt(stats["latest"], "{:.3f}", na="—") if stats else "—"
            parts.append(f"<td>{val}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _overview_table_html(all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]]) -> str:
    cols = [
        "Country", "Series", "Latest", "Chg 1d (bp)", "Chg 5d (bp)", "Chg 20d (bp)",
        "Chg 60d (bp)", f"RSI({RSI_PERIOD})", "%B", "vs SMA50 (bp)",
        "vs SMA200 (bp)", "Vol 20d (bp/yr)", "Fib zone",
    ]
    parts = ['<div class="table-wrap"><table class="data-table"><thead><tr>']
    parts += [f"<th>{html.escape(c)}</th>" for c in cols]
    parts.append("</tr></thead><tbody>")
    for country, series_names in COUNTRY_SERIES_ORDER.items():
        for name in series_names:
            stats = all_stats[country].get(name)
            if stats is None:
                cells = [COUNTRY_NAMES[country], name] + ["—"] * (len(cols) - 2)
            else:
                latest_txt = _fmt(stats["latest"], _latest_fmt(stats))
                latest_txt += " bp" if stats["in_bp"] else "%"
                cells = [
                    COUNTRY_NAMES[country],
                    name,
                    latest_txt,
                    _fmt_bp(stats["chg_1d"]),
                    _fmt_bp(stats["chg_5d"]),
                    _fmt_bp(stats["chg_20d"]),
                    _fmt_bp(stats["chg_60d"]),
                    _fmt(stats["rsi_last"], "{:.1f}"),
                    _fmt(stats["pct_b_last"], "{:.2f}"),
                    _fmt_bp(stats["dist_sma50"]),
                    _fmt_bp(stats["dist_sma200"]),
                    _fmt(stats["vol20_last"], "{:.0f}"),
                    stats["fib_zone"],
                ]
            parts.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--panel:#fff;--bg:#eef2f6;--accent:#174a6e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
.toggle-bar{position:sticky;top:0;z-index:100;display:flex;flex-direction:column;gap:6px;
background:#102b3c;color:#e7f0f5;padding:10px 18px;box-shadow:0 2px 8px rgba(15,23,42,.25)}
.tb-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tb-title{font-size:16px;font-weight:bold}
.tb-date{font-size:12px;color:#b9cad5;margin-left:12px}
.tb-label{font-size:11px;color:#8fc1dc;text-transform:uppercase;letter-spacing:.08em;width:64px}
.tgl-btn{border:1px solid #507084;background:#1e4b65;color:#dce9f0;padding:5px 15px;border-radius:6px;
cursor:pointer;font-size:13px;font-weight:bold}
.tgl-btn:hover{background:#2a6284}
.tgl-btn.active{background:#fff;color:#102b3c;border-color:#fff}
main{padding:20px;max-width:1500px;margin:0 auto}
.country-section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
box-shadow:0 3px 12px rgba(15,23,42,.06);padding:16px 18px}
.country-section h2{font-size:18px;margin:2px 0 12px;color:var(--accent)}
.tenor-block{margin-bottom:26px;border-top:1px solid var(--line);padding-top:12px}
.tenor-block:first-of-type{border-top:0;padding-top:0}
.tenor-block h3{font-size:15px;margin:2px 0 10px;color:var(--ink)}
.chart{width:100%;margin-bottom:18px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px;margin-bottom:12px}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 8px;font-weight:bold;text-align:center;white-space:nowrap}
.data-table td{border:1px solid #e2e8f0;padding:4px 8px;text-align:center;white-space:nowrap}
.data-table tbody tr:nth-child(even){background:#f0f0f0}
.fib-note{font-size:12px;color:var(--muted);margin:4px 0 16px}
.no-data{font-size:13px;color:var(--muted);font-style:italic;padding:14px;border:1px dashed var(--line);
border-radius:6px;background:#f8fafc}
h4.ovh{font-size:13px;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
@media print{.toggle-bar{position:static}.country-section{display:block !important;break-after:page}
.tenor-block{display:block !important}}
"""

_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
const state={country:'all',tenor:'all'};
function setCountry(c){state.country=c;applyView();}
function setTenor(t){state.tenor=t;applyView();}
function applyView(){
document.querySelectorAll('.country-btn').forEach(b=>{b.classList.toggle('active',b.dataset.country===state.country);});
document.querySelectorAll('.tenor-btn').forEach(b=>{b.classList.toggle('active',b.dataset.tenor===state.tenor);});
const sec=document.getElementById('country-'+state.country);
document.querySelectorAll('.country-section').forEach(s=>{s.style.display=(s===sec)?'':'none';});
if(sec){
const filterBlocks=(state.country!=='all');
sec.querySelectorAll('.tenor-block').forEach(bl=>{
const show=bl.dataset.always==='1'||!filterBlocks||state.tenor==='all'||bl.dataset.tenor===state.tenor;
bl.style.display=show?'':'none';
if(show)bl.querySelectorAll('.chart').forEach(el=>renderChart(el.id));
});
}
window.scrollTo({top:0,behavior:'smooth'});
}
applyView();
"""


class _ReportBuilder:
    """Collects figures as JSON and emits self-contained section HTML."""

    def __init__(self) -> None:
        self._fig_count = 0

    def fig_block(self, fig: go.Figure) -> str:
        self._fig_count += 1
        fid = f"fig-{self._fig_count:03d}"
        fig_json = pio.to_json(fig).replace("</", "<\\/")
        return (
            f'<div class="chart" id="{fid}"></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{fig_json}</script>"
        )


def _country_section_html(
    builder: _ReportBuilder,
    country: str,
    country_stats: Dict[str, Optional[Dict[str, object]]],
) -> str:
    parts = [f"<h2>{html.escape(COUNTRY_NAMES[country])} - Technical Summary</h2>"]
    is_mbs = country == "mbs"
    for name in COUNTRY_SERIES_ORDER[country]:
        slug = name.lower().replace(" ", "")
        label = f"{name} OAS" if is_mbs else f"{COUNTRY_SHORT[country]} {name}"
        # Tenor filtering does not apply to the MBS page
        block_attrs = ' data-always="1"' if is_mbs else f' data-tenor="{slug}"'
        stats = country_stats.get(name)
        if stats is None:
            ticker = COUNTRY_TICKERS[country][name]
            parts.append(
                f'<div class="tenor-block"{block_attrs}><h3>{html.escape(label)}</h3>'
                f'<p class="no-data">No data available for {html.escape(ticker)} '
                f"(ticker did not resolve or has insufficient history).</p></div>"
            )
            continue
        color = _series_color(country, name)
        parts.append(
            f'<div class="tenor-block"{block_attrs}><h3>{html.escape(label)}</h3>'
            f"{_series_strip_html(stats)}"
            f"{builder.fig_block(_fig_series_main(label, color, stats))}"
            f"{builder.fig_block(_fig_series_rsi(label, color, stats))}"
            f"{builder.fig_block(_fig_series_vol(label, stats))}"
            f"</div>"
        )
    return "".join(parts)


def build_html(
    all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]],
    as_of: date,
    report_path: Path,
) -> Path:
    builder = _ReportBuilder()

    # ALL overview section (default landing)
    overview_parts = [
        "<h2>Overview - All Countries</h2>",
        '<h4 class="ovh">Latest sovereign yields (%)</h4>',
        _overview_matrix_html(all_stats),
        '<h4 class="ovh">Cross-country technical summary</h4>',
        _overview_table_html(all_stats),
        builder.fig_block(_fig_overview_10y(all_stats)),
    ]
    mbs_fig = _fig_overview_mbs(all_stats)
    if mbs_fig is not None:
        overview_parts.append(builder.fig_block(mbs_fig))
    section_parts = [
        f'<section class="country-section" id="country-all" style="display:block">'
        f'<div class="tenor-block" data-always="1">{"".join(overview_parts)}</div></section>'
    ]
    for country in COUNTRY_TICKERS:
        body = _country_section_html(builder, country, all_stats[country])
        section_parts.append(
            f'<section class="country-section" id="country-{country}" style="display:none">{body}</section>'
        )

    country_buttons = [
        '<button class="tgl-btn country-btn active" data-country="all" '
        'onclick="setCountry(\'all\')">ALL</button>'
    ]
    for country in COUNTRY_TICKERS:
        country_buttons.append(
            f'<button class="tgl-btn country-btn" data-country="{country}" '
            f'onclick="setCountry(\'{country}\')">{COUNTRY_LABELS[country]}</button>'
        )
    tenor_buttons = [
        '<button class="tgl-btn tenor-btn active" data-tenor="all" '
        'onclick="setTenor(\'all\')">ALL</button>'
    ]
    for tenor in TENORS:
        slug = tenor.lower()
        tenor_buttons.append(
            f'<button class="tgl-btn tenor-btn" data-tenor="{slug}" '
            f'onclick="setTenor(\'{slug}\')">{tenor}</button>'
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<nav class="toggle-bar">
<div class="tb-row"><span class="tb-title">{html.escape(REPORT_TITLE)}</span>
<span class="tb-date">as of {as_of}</span></div>
<div class="tb-row"><span class="tb-label">Country</span>{''.join(country_buttons)}</div>
<div class="tb-row"><span class="tb-label">Tenor</span>{''.join(tenor_buttons)}</div>
</nav>
<main>{''.join(section_parts)}</main>
<script>{_JS}</script>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_console_summary(
    all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]],
    warnings_: List[str],
    report_path: Path,
) -> None:
    print("\n" + "=" * 78)
    print(f"{REPORT_TITLE} - snapshot")
    print("=" * 78)
    header = f"{'Country':<15}{'Series':<7}{'Latest':>9}{'RSI':>7}{'%B':>7}{'vsSMA50':>9}{'Vol20':>7}"
    print(header)
    print("-" * len(header))
    for country, series_names in COUNTRY_SERIES_ORDER.items():
        for name in series_names:
            stats = all_stats[country].get(name)
            if stats is None:
                print(f"{COUNTRY_NAMES[country]:<15}{name:<7}{'no data':>9}")
                continue
            latest_txt = _fmt(stats["latest"], _latest_fmt(stats))
            if stats["in_bp"]:
                latest_txt += "bp"
            print(
                f"{COUNTRY_NAMES[country]:<15}{name:<7}"
                f"{latest_txt:>9}"
                f"{_fmt(stats['rsi_last'], '{:.1f}'):>7}"
                f"{_fmt(stats['pct_b_last'], '{:.2f}'):>7}"
                f"{_fmt_bp(stats['dist_sma50']):>9}"
                f"{_fmt(stats['vol20_last'], '{:.0f}'):>7}"
            )
    if warnings_:
        print("\nWarnings:")
        for w in warnings_:
            print(f"  - {w}")
    print(f"\nHTML report saved to: {report_path}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(start_date: str = START_DATE) -> Path:
    data, warnings_ = load_yields(start_date=start_date)

    print("Computing technical indicators...")
    all_stats: Dict[str, Dict[str, Optional[Dict[str, object]]]] = {}
    for country, series_names in COUNTRY_SERIES_ORDER.items():
        all_stats[country] = {}
        in_bp = COUNTRY_UNITS[country] == "bp"
        for name in series_names:
            s = data[country][name]
            all_stats[country][name] = build_series_stats(s, in_bp=in_bp) if s is not None else None

    last_dates = [
        stats["last_date"]
        for ct in all_stats.values()
        for stats in ct.values()
        if stats is not None
    ]
    as_of = pd.Timestamp(max(last_dates)).date()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"UST_Technical_Monitor_{as_of.strftime('%Y%m%d')}.html"

    print("Building interactive HTML report...")
    build_html(all_stats, as_of, report_path)

    print_console_summary(all_stats, warnings_, report_path)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the sovereign & MBS technical monitor (interactive HTML).")
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"History start date for the Bloomberg pull (default: {START_DATE}).",
    )
    args = parser.parse_args()
    main(start_date=args.start_date)
