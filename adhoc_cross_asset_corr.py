r"""
Ad-hoc Cross-Asset Correlation Report (PDF)
===========================================
Daily-change correlations of rates/spreads/vol series vs UST 10Y, 5y daily
sample from Bloomberg. Matplotlib PdfPages output in the repo's PDF style.

Series (display units in brackets):
    UST 10Y yield (%)                       GT10 Govt
    FN current-coupon ZV spread (bp)        .FNCC105 G Index  x100
    IG credit OAS (bp)                      LUACOAS Index     x100
    USD 10Y swap spread (bp)                USSFCT10 Curncy (fallbacks USSC10/USWS10)
    OAT 10Y swap spread (bp)                FRSS10 Curncy
    UST 5s10s30s fly (bp)                   2 x GT10 - GT5 - GT30  x100
    1y1y swaption vol (bp)                  USSNA110 FMCA Curncy
    10y10y - 1y1y vol diff (bp)             USSNJ10 - USSNA110 (fallback USSNA1010)
    OAT-Bund 10Y spread (bp)                (GTFRF10Y - GTDEM10Y) x100
    JGB 10s2s (bp)                          (GTJPY10Y - GTJPY2Y) x100
    HKD 6M-3M rate spread (bp)              probe pairs (currently unresolved)
    CNH 6M-3M rate spread (bp)              CNHH6M - CNHH3M (fallback CCNH pair) x100

Run:
    python adhoc_cross_asset_corr.py

Output:
    ./adhoc_output/Cross_Asset_Corr_Report_<latest data date>.pdf
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "adhoc_output"
REPORT_TITLE = "Cross-Asset Correlation Report"
START_DATE = "2021-10-27"
ROLL_WINDOW = 90
ROLL_MINMAX_YEARS = 1
BASE_SERIES = "UST 10Y"

# (name, ticker candidates in preference order, scale to display units, units label)
SINGLE_TICKER_SPECS: List[Tuple[str, List[str], float, str]] = [
    ("UST 10Y", ["GT10 Govt"], 1.0, "%"),
    ("FNCC ZV", [".FNCC105 G Index"], 100.0, "bp"),
    ("IG OAS", ["LUACOAS Index"], 100.0, "bp"),
    ("USD 10Y swap sprd", ["USSFCT10 Curncy", "USSC10 Curncy", "USWS10 Curncy"], 1.0, "bp"),
    ("OAT 10Y swap sprd", ["FRSS10 Curncy"], 1.0, "bp"),
    ("1y1y swaption vol", ["USSNA110 FMCA Curncy"], 1.0, "bp"),
    ("10y10y swaption vol", ["USSNJ10 FMCA Curncy", "USSNA1010 FMCA Curncy"], 1.0, "bp"),
    ("GT5", ["GT5 Govt"], 1.0, "%"),
    ("GT30", ["GT30 Govt"], 1.0, "%"),
    ("GTFRF10Y", ["GTFRF10Y Govt"], 1.0, "%"),
    ("GTDEM10Y", ["GTDEM10Y Govt"], 1.0, "%"),
    ("GTJPY10Y", ["GTJPY10Y Govt"], 1.0, "%"),
    ("GTJPY2Y", ["GTJPY2Y Govt"], 1.0, "%"),
    ("HKD 6M", ["HIHK6M Index", "HIHD6M Index", "HBHD6M Curncy"], 1.0, "%"),
    ("HKD 3M", ["HIHK3M Index", "HIHD3M Index", "HBHD3M Curncy"], 1.0, "%"),
    ("CNH 6M", ["CNHH6M Index", "CCNH6M Curncy"], 1.0, "%"),
    ("CNH 3M", ["CNHH3M Index", "CCNH3M Curncy"], 1.0, "%"),
]


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


def load_raw(start_date: str = START_DATE) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """One bdh call for all candidate tickers.

    Returns (wide df, name -> resolved ticker, list of failed names).
    """
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    tickers = sorted({t for _, cands, _, _ in SINGLE_TICKER_SPECS for t in cands})
    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} candidate tickers from Bloomberg...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)
    df = _normalize_bdh(df, tickers=tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    resolved: Dict[str, str] = {}
    failed: List[str] = []
    for name, cands, _, _ in SINGLE_TICKER_SPECS:
        hit = next((t for t in cands if t in df.columns and df[t].notna().sum() > 30), None)
        if hit:
            resolved[name] = hit
            if hit != cands[0]:
                print(f"  {name}: primary {cands[0]} failed, using {hit}")
        else:
            failed.append(name)
            print(f"  WARNING: {name} unresolved (tried {', '.join(cands)})")
    return df, resolved, failed


def build_panel(raw: pd.DataFrame, resolved: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Assemble the display-unit panel. Returns (panel, units map, failed series)."""
    series: Dict[str, pd.Series] = {}
    units: Dict[str, str] = {}
    failed: List[str] = []

    def get(name: str) -> Optional[pd.Series]:
        if name not in resolved:
            return None
        scale = next(s for n, _, s, _ in SINGLE_TICKER_SPECS if n == name)
        s = pd.to_numeric(raw[resolved[name]], errors="coerce").dropna() * scale
        return s

    # Single-ticker report series (10y10y vol is downloaded as an input for the
    # vol-diff series below, not displayed on its own)
    for name, _, scale, unit in SINGLE_TICKER_SPECS[:6]:
        s = get(name)
        if s is None:
            failed.append(name)
            continue
        series[name] = s
        units[name] = unit

    # Computed series
    gt5, gt10, gt30 = get("GT5"), get("UST 10Y"), get("GT30")
    if gt5 is not None and gt30 is not None:
        series["UST 5s10s30s fly"] = ((2 * series["UST 10Y"] - gt5 - gt30) * 100.0).dropna()
        units["UST 5s10s30s fly"] = "bp"
    else:
        failed.append("UST 5s10s30s fly")

    v1, v10 = get("1y1y swaption vol"), get("10y10y swaption vol")
    if v1 is not None and v10 is not None:
        series["10y10y-1y1y vol diff"] = (v10 - v1).dropna()
        units["10y10y-1y1y vol diff"] = "bp"
    else:
        failed.append("10y10y-1y1y vol diff")

    oat, bund = get("GTFRF10Y"), get("GTDEM10Y")
    if oat is not None and bund is not None:
        series["OAT-Bund 10Y"] = ((oat - bund) * 100.0).dropna()
        units["OAT-Bund 10Y"] = "bp"
    else:
        failed.append("OAT-Bund 10Y")

    j10, j2 = get("GTJPY10Y"), get("GTJPY2Y")
    if j10 is not None and j2 is not None:
        series["JGB 10s2s"] = ((j10 - j2) * 100.0).dropna()
        units["JGB 10s2s"] = "bp"
    else:
        failed.append("JGB 10s2s")

    h6, h3 = get("HKD 6M"), get("HKD 3M")
    if h6 is not None and h3 is not None:
        series["HKD 6M-3M"] = ((h6 - h3) * 100.0).dropna()
        units["HKD 6M-3M"] = "bp"
    else:
        failed.append("HKD 6M-3M")

    c6, c3 = get("CNH 6M"), get("CNH 3M")
    if c6 is not None and c3 is not None:
        series["CNH 6M-3M"] = ((c6 - c3) * 100.0).dropna()
        units["CNH 6M-3M"] = "bp"
    else:
        failed.append("CNH 6M-3M")

    panel = pd.concat(series, axis=1, sort=True).dropna()
    print(f"Aligned panel: {panel.shape[0]} obs x {panel.shape[1]} series, "
          f"{panel.index[0].date()} to {panel.index[-1].date()}")
    return panel, units, failed


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_summary(
    panel: pd.DataFrame, units: Dict[str, str], changes: pd.DataFrame, rolling: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Per-series latest / changes / correlation-vs-UST10Y summary."""
    rows = []
    for col in panel.columns:
        s = panel[col]
        rc = rolling.get(col)
        recent_rc = rc.dropna().tail(252) if rc is not None else pd.Series(dtype=float)
        rows.append({
            "series": col,
            "units": units.get(col, ""),
            "latest": float(s.iloc[-1]),
            "chg_5d": float(s.iloc[-1] - s.iloc[-6]) if len(s) > 5 else np.nan,
            "chg_20d": float(s.iloc[-1] - s.iloc[-21]) if len(s) > 20 else np.nan,
            "corr_chg_full": float(changes[col].corr(changes[BASE_SERIES])),
            "corr_90d_latest": float(rc.dropna().iloc[-1]) if rc is not None and not rc.dropna().empty else np.nan,
            "corr_90d_1y_min": float(recent_rc.min()) if not recent_rc.empty else np.nan,
            "corr_90d_1y_max": float(recent_rc.max()) if not recent_rc.empty else np.nan,
        })
    return pd.DataFrame(rows).set_index("series")


def compute_rolling_corr(changes: pd.DataFrame, window: int = ROLL_WINDOW) -> Dict[str, pd.Series]:
    """Rolling correlation of each series' daily change vs UST 10Y daily change."""
    out = {}
    base = changes[BASE_SERIES]
    for col in changes.columns:
        if col == BASE_SERIES:
            continue
        out[col] = changes[col].rolling(window=window, min_periods=window).corr(base)
    return out


# ---------------------------------------------------------------------------
# Equal-weight portfolio variance analysis
# ---------------------------------------------------------------------------
def compute_portfolio(changes: pd.DataFrame, units: Dict[str, str]) -> Dict[str, object]:
    """Equal-weight (1/N) portfolio of daily changes, all in bp.

    %-quoted series' changes are converted to bp (x100). Variance contribution
    is w_i (Sigma w)_i / sigma_p^2; diversification ratio is sigma_p divided by
    the equal-weight average individual vol.
    """
    changes_bp = changes.copy()
    for col in changes_bp.columns:
        if units.get(col) == "%":
            changes_bp[col] = changes_bp[col] * 100.0

    n = changes_bp.shape[1]
    w = np.full(n, 1.0 / n)
    cov = changes_bp.cov().values  # daily, bp^2
    port = changes_bp.values @ w
    port_s = pd.Series(port, index=changes_bp.index, name="EW portfolio")
    var_p = float(w @ cov @ w)
    sigma_p = float(np.sqrt(var_p))
    sigma_i = changes_bp.std(ddof=1)

    marginal = cov @ w  # (Sigma w)_i
    var_contrib = w * marginal / var_p  # shares, sum to 1

    per_asset = pd.DataFrame({
        "asset": changes_bp.columns,
        "vol_bp_day": sigma_i.values,
        "vol_bp_yr": sigma_i.values * np.sqrt(252.0),
        "variance_bp2": np.diag(cov) * (1.0 / n) ** 2,
        "var_contrib_pct": var_contrib * 100.0,
        "corr_vs_portfolio": [changes_bp[c].corr(port_s) for c in changes_bp.columns],
    }).set_index("asset")

    div_ratio = sigma_p / float(sigma_i.mean()) if sigma_i.mean() > 0 else np.nan

    # Rolling 90d vols, annualized
    port_roll_vol = port_s.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std(ddof=1) * np.sqrt(252.0)
    avg_roll_vol = changes_bp.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std(ddof=1).mean(axis=1) * np.sqrt(252.0)

    # Cumulative: portfolio vs top contributors (weighted contributions, bp)
    contrib_daily = changes_bp * w  # w_i x d_i per asset
    top = per_asset["var_contrib_pct"].sort_values(ascending=False).head(3).index.tolist()
    cum = pd.DataFrame({"EW portfolio": port_s.cumsum()})
    for a in top:
        cum[f"{a} (weighted)"] = contrib_daily[a].cumsum()

    worst_5d = port_s.rolling(5).sum()
    worst_idx = worst_5d.idxmin()

    summary = {
        "n_assets": n,
        "vol_bp_day": sigma_p,
        "vol_bp_yr": sigma_p * np.sqrt(252.0),
        "avg_indiv_vol_bp_day": float(sigma_i.mean()),
        "avg_indiv_vol_bp_yr": float(sigma_i.mean() * np.sqrt(252.0)),
        "variance_bp2": var_p,
        "diversification_ratio": div_ratio,
        "top_contributors": [
            (a, float(per_asset.loc[a, "var_contrib_pct"])) for a in top
        ],
        "worst_5d_bp": float(worst_5d.loc[worst_idx]),
        "worst_5d_end": str(worst_idx.date()),
    }
    return {
        "changes_bp": changes_bp,
        "port_daily": port_s,
        "per_asset": per_asset,
        "port_roll_vol": port_roll_vol,
        "avg_roll_vol": avg_roll_vol,
        "cum": cum,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# PDF helpers (repo style)
# ---------------------------------------------------------------------------
def _idx_date(idx_value) -> date:
    return idx_value.date() if hasattr(idx_value, "date") else idx_value


def _add_cover_page(
    pdf: PdfPages, panel: pd.DataFrame, units: Dict[str, str],
    resolved: Dict[str, str], failed: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.93, REPORT_TITLE, fontsize=22, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.88,
            f"Latest: {_idx_date(panel.index[-1])}  |  Observations: {len(panel)}  |  "
            f"Sample: {_idx_date(panel.index[0])} to {_idx_date(panel.index[-1])}",
            fontsize=11, ha="center", color="gray", transform=ax.transAxes)

    lines = ["Series (display units / ticker)\n"]
    for col in panel.columns:
        lines.append(f"  {col:<24} [{units.get(col, '')}]")
    contents = (
        "Report contents\n\n"
        "1. Correlation matrix - daily changes (full sample + trailing 90 obs)\n"
        "2. Correlation matrix - levels (reference; non-stationary)\n"
        "3. Rolling 90d correlation vs UST 10Y daily change\n"
        "4. Equal-weight portfolio variance analysis\n"
        "5. Summary table"
    )
    ax.text(0.07, 0.78, "\n".join(lines), fontsize=9, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2ff", edgecolor="#40466e"))
    ax.text(0.60, 0.78, contents, fontsize=10, family="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8ee", edgecolor="#8b5a00"))

    notes = [
        "Correlations are computed on daily CHANGES (levels are non-stationary);",
        "the levels matrix is a secondary reference view.",
        f"Rolling window: {ROLL_WINDOW} observations; 1y min/max over trailing 252 obs.",
        "",
        "Unresolved / excluded series:",
    ]
    notes += [f"  - {f}" for f in failed] if failed else ["  (none - all series resolved)"]
    ax.text(0.07, 0.38, "\n".join(notes), fontsize=9, family="monospace",
            verticalalignment="top", transform=ax.transAxes, color="#8b2500")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_heatmap_page(pdf: PdfPages, corr: pd.DataFrame, title: str, subtitle: str) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=13, weight="bold", x=0.04, ha="left")
    ax.set_title(subtitle, fontsize=8, color="#555555", loc="left", pad=28)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_rolling_corr_pages(
    pdf: PdfPages, rolling: Dict[str, pd.Series], full_corr: pd.Series, per_page: int = 3
) -> None:
    cols = list(rolling.keys())
    for start in range(0, len(cols), per_page):
        chunk = cols[start:start + per_page]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(11.0, 3.0 * len(chunk)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, chunk):
            rc = rolling[col].dropna()
            ax.plot(rc.index, rc.values, color="navy", linewidth=1.0)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axhline(full_corr[col], color="red", linestyle="--", linewidth=0.9,
                       label=f"full-sample corr = {full_corr[col]:+.2f}")
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("corr")
            ax.set_title(f"{col} - rolling {ROLL_WINDOW}d corr of daily changes vs {BASE_SERIES}",
                         fontsize=10)
            ax.legend(fontsize=8, loc="lower left")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _add_table_page(pdf: PdfPages, summary: pd.DataFrame) -> None:
    display = summary.reset_index().copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(13.0, 4.5))
    ax.axis("off")
    fig.suptitle("Summary - levels, changes and correlation vs UST 10Y", fontsize=13,
                 weight="bold", x=0.03, ha="left")
    table = ax.table(
        cellText=[display.columns.tolist()] + display.values.tolist(),
        loc="center", cellLoc="center",
        colWidths=[0.16] + [0.84 / (display.shape[1] - 1)] * (display.shape[1] - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for j in range(display.shape[1]):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")
    for i in range(1, display.shape[0] + 1):
        for j in range(display.shape[1]):
            table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_portfolio_table_page(pdf: PdfPages, port: Dict[str, object]) -> None:
    per_asset: pd.DataFrame = port["per_asset"]  # type: ignore[assignment]
    ps = port["summary"]  # type: ignore[assignment]
    display = per_asset.reset_index().copy()
    display.columns = ["asset", "vol (bp/day)", "vol (bp/yr)", "variance (bp^2)",
                       "var contrib (%)", "corr vs portfolio"]
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    port_row = pd.DataFrame([{
        "asset": "EW PORTFOLIO",
        "vol (bp/day)": f"{ps['vol_bp_day']:.2f}",
        "vol (bp/yr)": f"{ps['vol_bp_yr']:.1f}",
        "variance (bp^2)": f"{ps['variance_bp2']:.3f}",
        "var contrib (%)": "100.00",
        "corr vs portfolio": "1.00",
    }])
    display = pd.concat([display, port_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.axis("off")
    fig.suptitle("Equal-Weight Portfolio Variance - Daily Changes (all in bp)", fontsize=13,
                 weight="bold", x=0.03, ha="left")
    subtitle = (
        f"Weights = 1/N, N = {ps['n_assets']}  |  %-quoted series converted to bp  |  "
        f"variance contribution = w_i (Σw)_i / σ_p²  |  "
        f"diversification ratio σ_p / mean(σ_i) = {ps['diversification_ratio']:.2f} "
        f"(portfolio vol {ps['vol_bp_day']:.2f} bp/day vs avg individual "
        f"{ps['avg_indiv_vol_bp_day']:.2f} bp/day)"
    )
    ax.set_title(subtitle, fontsize=8, color="#555555", loc="left", pad=16)
    table = ax.table(
        cellText=[display.columns.tolist()] + display.values.tolist(),
        loc="center", cellLoc="center",
        colWidths=[0.22] + [0.78 / (display.shape[1] - 1)] * (display.shape[1] - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for j in range(display.shape[1]):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")
    for i in range(1, display.shape[0] + 1):
        for j in range(display.shape[1]):
            cell = table[(i, j)]
            cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")
            if i == display.shape[0]:  # portfolio row
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_portfolio_roll_vol_page(pdf: PdfPages, port: Dict[str, object]) -> None:
    prv: pd.Series = port["port_roll_vol"]  # type: ignore[assignment]
    arv: pd.Series = port["avg_roll_vol"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(prv.index, prv.values, color="navy", linewidth=1.3,
            label="EW portfolio rolling vol")
    ax.plot(arv.index, arv.values, color="#d97706", linewidth=1.1, linestyle="--",
            label="Equal-weight avg of individual rolling vols")
    ax.set_ylabel("annualized vol (bp/yr)")
    ax.set_title(f"Rolling {ROLL_WINDOW}d Volatility - Portfolio vs Undiversified Average",
                 fontsize=12, weight="bold", loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.fill_between(prv.index, prv.values, arv.values, color="#059669", alpha=0.12)
    ax.text(0.02, 0.04, "Shaded gap = diversification benefit", transform=ax.transAxes,
            fontsize=8, color="#059669")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_portfolio_cumulative_page(pdf: PdfPages, port: Dict[str, object]) -> None:
    cum: pd.DataFrame = port["cum"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(cum.index, cum["EW portfolio"], color="navy", linewidth=1.4,
            label="EW portfolio")
    colors = ["#dc2626", "#d97706", "#7c3aed"]
    for i, col in enumerate(c for c in cum.columns if c != "EW portfolio"):
        ax.plot(cum.index, cum[col], linewidth=1.0, color=colors[i % 3],
                label=col)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("cumulative daily change (bp)")
    ax.set_title("Cumulative Daily Change - EW Portfolio vs Largest Variance Contributors",
                 fontsize=12, weight="bold", loc="left")
    ax.text(0.02, 0.96, "Contributors shown as weighted contributions (w_i × Δ_i), "
            "so they sum into the portfolio line", transform=ax.transAxes,
            fontsize=8, color="#555555", va="top")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    panel: pd.DataFrame,
    units: Dict[str, str],
    resolved: Dict[str, str],
    failed: List[str],
    pdf_path: Optional[Path] = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = panel.index[-1].strftime("%Y%m%d")
    if pdf_path is None:
        pdf_path = OUTPUT_DIR / f"Cross_Asset_Corr_Report_{report_date}.pdf"

    changes = panel.diff().dropna()
    corr_chg = changes.corr()
    changes_90 = changes.tail(90)
    corr_chg_90 = changes_90.corr()
    corr_lvl = panel.corr()
    rolling = compute_rolling_corr(changes)
    summary = compute_summary(panel, units, changes, rolling)
    port = compute_portfolio(changes, units)

    def _render(path: Path) -> Path:
        with PdfPages(path) as pdf:
            _add_cover_page(pdf, panel, units, resolved, failed)
            _add_heatmap_page(
                pdf, corr_chg,
                "Pairwise Correlation - Daily Changes (full sample)",
                f"{_idx_date(changes.index[0])} to {_idx_date(changes.index[-1])}  |  "
                f"{len(changes)} daily changes",
            )
            _add_heatmap_page(
                pdf, corr_chg_90,
                "Pairwise Correlation - Daily Changes (trailing 90 observations)",
                f"{_idx_date(changes_90.index[0])} to {_idx_date(changes_90.index[-1])}  |  "
                f"{len(changes_90)} daily changes",
            )
            _add_heatmap_page(
                pdf, corr_lvl,
                "Pairwise Correlation - Levels (reference only)",
                "Levels are non-stationary; use the daily-change matrix for inference.",
            )
            _add_rolling_corr_pages(pdf, rolling, corr_chg[BASE_SERIES])
            _add_portfolio_table_page(pdf, port)
            _add_portfolio_roll_vol_page(pdf, port)
            _add_portfolio_cumulative_page(pdf, port)
            _add_table_page(pdf, summary)
        return path

    try:
        out = _render(pdf_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = OUTPUT_DIR / f"Cross_Asset_Corr_Report_{report_date}_{ts}.pdf"
        print(f"Warning: could not write to dated PDF (file may be open). Saving to {pdf_path}")
        out = _render(pdf_path)
    return out, summary, port


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    raw, resolved, load_failed = load_raw()
    panel, units, failed = build_panel(raw, resolved)
    failed = sorted(set(failed + load_failed))

    pdf_path, summary, port = generate_pdf_report(panel, units, resolved, failed)

    print("\n" + "=" * 88)
    print(REPORT_TITLE)
    print("=" * 88)
    print("Resolved tickers:")
    for name, ticker in resolved.items():
        print(f"  {name:<22} -> {ticker}")
    if failed:
        print("Failed / excluded series:")
        for f in failed:
            print(f"  - {f}")
    print("\nSummary table:")
    print(summary.round(3).to_string())

    ps = port["summary"]
    print("\nEqual-weight portfolio (daily changes, all in bp):")
    print(f"  N assets            : {ps['n_assets']}")
    print(f"  Portfolio vol       : {ps['vol_bp_day']:.2f} bp/day  |  {ps['vol_bp_yr']:.1f} bp/yr")
    print(f"  Avg individual vol  : {ps['avg_indiv_vol_bp_day']:.2f} bp/day  |  "
          f"{ps['avg_indiv_vol_bp_yr']:.1f} bp/yr")
    print(f"  Diversification ratio (sigma_p / mean sigma_i): {ps['diversification_ratio']:.3f}")
    print("  Top-3 variance contributors:")
    for name, share in ps["top_contributors"]:
        print(f"    {name:<26} {share:5.1f}%")
    print(f"  Worst 5d move       : {ps['worst_5d_bp']:+.1f} bp (ending {ps['worst_5d_end']})")
    print(f"\nPDF report saved to: {pdf_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
