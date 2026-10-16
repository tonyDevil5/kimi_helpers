r"""
Swap Spread Report (Draft)
==========================
Analyzes swap spreads across 2Y/3Y/5Y/7Y/10Y/20Y/30Y tenors.

Data source is controlled by USE_BBG:
    - USE_BBG = True : download from Bloomberg using configured ticker roots
                     (USD USSFCT BGN, Bund EUSS BLC, OAT FRSS BLC,
                      Gilt BPOS BLC, BTP ILSS BLC, JGB JYSST BLC)
    - USE_BBG = False: read USD SOFR and GBP SONIA swap spreads from
                       Sheet4 of mbs_rv_citi.xlsx

Report content:
    - 250-day percentile summary (min, 25th, 50th, 75th, max)
    - 20/60/90-day rolling z-scores
    - 20/60/90-day annualized standard deviation of levels
    - 20/60/90-day convergence sharpe ratio: (-latest - 10bps) / ann_std
    - USD/cross-asset correlation tables (ESA, CT5, CDX IG 5Y, FNCC105 ZV)

Run:
    python swap_spread_report.py                          # interactive HTML (default)
    python swap_spread_report.py --format pdf             # PDF only
    python swap_spread_report.py --format interactive pdf # both

Output:
    ./swap_spread_output/SwapSpread_Report_<latest data date>.html
    ./swap_spread_output/SwapSpread_Report_<date>.pdf
"""

from __future__ import annotations

import argparse
import html
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FNAME = r"D:\python\notebook\data\mbs\mbs_rv_citi.xlsx"
OUTPUT_DIR = Path(__file__).parent / "swap_spread_output"

USE_BBG = True  # If True, download swap spreads from Bloomberg; else read from mbs_rv_citi.xlsx Sheet4

PERCENTILE_WINDOW = 250
ZSCORE_WINDOWS = [20, 60, 90]
TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# Fair-value target (bps) used for the convergence sharpe ratio:
# sharpe_w = (-latest - 10bps) / ann_std_w = (FV_TARGET_BPS - latest) / ann_std_w
FV_TARGET_BPS = -10.0

# Bloomberg ticker roots for swap spreads by currency
BBG_TICKER_ROOTS = {
    "USD": "USSFCT",
    "Bund": "EUSS",
    "OAT": "FRSS",
    "Gilt": "BPOS",
    "BTP": "ILSS",
    "JGB": "JYSST",
}

# Tenors available per currency (defaults to TENORS for all others)
CCY_TENORS: Dict[str, List[str]] = {}

# Bloomberg ticker suffix / pricing source.
# USD SOFR swap spreads use BGN; European and JGB swap spreads use BLC.
BBG_TICKER_SUFFIXES = {
    "USD": "BGN Curncy",
    "Bund": "BLC Curncy",
    "OAT": "BLC Curncy",
    "Gilt": "BLC Curncy",
    "BTP": "BLC Curncy",
    "JGB": "BLC Curncy",
}

# Currencies/markets to include in the report
CURRENCIES = list(BBG_TICKER_ROOTS.keys()) if USE_BBG else ["USD", "GBP"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _build_bbg_tickers() -> List[str]:
    """Build full Bloomberg ticker list from BBG_TICKER_ROOTS and TENORS."""
    tickers = []
    for ccy, root in BBG_TICKER_ROOTS.items():
        suffix = BBG_TICKER_SUFFIXES.get(ccy, "Curncy")
        # USD tickers require a leading zero (e.g. USSFCT02); European/JGB tickers
        # use the bare tenor digit (e.g. EUSS2, FRSS3, BPOS5, ILSS7, JYSST2).
        pad = (ccy == "USD")
        for tenor in CCY_TENORS.get(ccy, TENORS):
            tenor_num = tenor.replace("Y", "")
            if pad:
                tenor_num = tenor_num.zfill(2)
            tickers.append(f"{root}{tenor_num} {suffix}")
    return tickers


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
        df = df.sort_index()
        if tickers is not None:
            if isinstance(tickers, str):
                tickers = [tickers]
            df = df.reindex(columns=[t for t in tickers if t in df.columns])
    return df


def load_swap_spread_data_from_bbg(start_date: str = "2018-05-21") -> pd.DataFrame:
    """Download swap spreads from Bloomberg."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    tickers = _build_bbg_tickers()
    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} swap spread tickers from Bloomberg...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)
    df = _normalize_bdh(df, tickers=tickers)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rename columns to {ccy}_{tenor}
    rename_map = {}
    for ticker in tickers:
        if ticker not in df.columns:
            continue
        for ccy, root in BBG_TICKER_ROOTS.items():
            if root in ticker:
                suffix = BBG_TICKER_SUFFIXES.get(ccy, "Curncy")
                m = re.search(rf"{root}(\d+) {re.escape(suffix)}", ticker)
                if m:
                    tenor = f"{int(m.group(1))}Y"
                    rename_map[ticker] = f"{ccy}_{tenor}"
                break

    df = df[list(rename_map.keys())].rename(columns=rename_map)
    df = df.dropna(how="all")
    print(f"Loaded Bloomberg swap spread data: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


def load_swap_spread_data_from_excel(fpath: str = FNAME) -> pd.DataFrame:
    """Load USD SOFR and GBP SONIA swap spreads from Sheet4 of mbs_rv_citi.xlsx (Excel mode)."""
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")

    df = pd.read_excel(fpath, sheet_name="Sheet4", header=1)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        ccy = None
        if "USD_SOFR" in col_str:
            ccy = "USD"
        elif "GBP_SONIA" in col_str:
            ccy = "GBP"

        m = re.search(r"SWAP_SPREAD\.(\d+Y)", col_str)
        if ccy and m:
            tenor = m.group(1)
            rename_map[col] = f"{ccy}_{tenor}"

    df = df[list(rename_map.keys())].rename(columns=rename_map)
    df = df.dropna(how="all")
    print(f"Loaded Excel swap spread data: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


def load_swap_spread_data(fpath: str = FNAME) -> pd.DataFrame:
    """Load swap spread data from Bloomberg or Excel depending on USE_BBG."""
    if USE_BBG:
        return load_swap_spread_data_from_bbg()
    return load_swap_spread_data_from_excel(fpath)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_percentile_summary(df: pd.DataFrame, window: int = PERCENTILE_WINDOW) -> pd.DataFrame:
    """
    Compute trailing `window`-day percentile summary for each swap spread series.
    Returns the latest value plus min / 25th / 50th / 75th / max of the trailing window.
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        recent = series.tail(window)
        latest = float(series.iloc[-1])

        rows.append({
            "series": col,
            "latest": latest,
            f"min_{window}d": float(recent.min()),
            f"p25_{window}d": float(recent.quantile(0.25)),
            f"p50_{window}d": float(recent.quantile(0.50)),
            f"p75_{window}d": float(recent.quantile(0.75)),
            f"max_{window}d": float(recent.max()),
            "range_pct": (latest - float(recent.min())) / (float(recent.max()) - float(recent.min()))
                         if float(recent.max()) != float(recent.min()) else np.nan,
        })

    return pd.DataFrame(rows).set_index("series")


def compute_zscore_summary(df: pd.DataFrame, windows: List[int] = ZSCORE_WINDOWS) -> pd.DataFrame:
    """
    Compute rolling z-scores and annualized standard deviations for each swap spread series.

    z = (latest - rolling_mean) / rolling_std
    ann_std = rolling_std(level) * sqrt(252), in bps
    sharpe = (-latest - 10bps) / ann_std, i.e. expected bp gain if the
             spread converges to FV_TARGET_BPS, divided by annualized std
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        latest = float(series.iloc[-1])
        row = {"series": col, "latest": latest}
        for w in windows:
            mean = series.rolling(window=w, min_periods=w).mean().iloc[-1]
            std = series.rolling(window=w, min_periods=w).std().iloc[-1]
            z = (series.iloc[-1] - mean) / std if pd.notna(std) and std != 0 else np.nan
            row[f"z_{w}d"] = float(z)

            ann_std = float(std * np.sqrt(252)) if pd.notna(std) else np.nan
            row[f"ann_std_{w}d"] = ann_std

            sharpe = (-latest - 10.0) / ann_std if pd.notna(ann_std) and ann_std != 0 else np.nan
            row[f"sharpe_{w}d"] = float(sharpe)
        rows.append(row)

    return pd.DataFrame(rows).set_index("series")


# ---------------------------------------------------------------------------
# PDF report helpers
# ---------------------------------------------------------------------------
def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    fontsize: int = 8,
    header_color: str = "#40466e",
    row_height: float = 0.35,
    gradient_columns: Optional[List[str]] = None,
    compact: bool = False,
    include_index: bool = True,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib figure (one page)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    if include_index:
        df = df.reset_index()

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    fig = plt.figure(figsize=pagesize)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.84])
    ax.axis("off")
    fig.suptitle(title, fontsize=13, weight="bold", y=0.985, ha="left", x=0.04)
    if subtitle:
        fig.text(0.04, 0.92, subtitle, fontsize=8, ha="left", color="#555555")

    table_data = [display_df.columns.tolist()] + display_df.values.tolist()

    if n_cols > 1:
        name_width = 0.22 if compact and n_cols >= 8 else 0.35
        col_widths = [name_width] + [(1.0 - name_width) / (n_cols - 1)] * (n_cols - 1)
    else:
        col_widths = [1.0]

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )

    total_cells = n_rows + 1
    if compact:
        cell_h = row_height
        auto_font = fontsize
    else:
        cell_h = max(row_height, 0.88 / total_cells)
        auto_font = min(fontsize, int(cell_h * 280))

    table.auto_set_font_size(False)
    table.set_fontsize(auto_font)

    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight="bold", color="white")
        cell.set_height(cell_h)

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_height(cell_h)
            base_color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
            cell.set_facecolor(base_color)

    return fig


def _add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    gradient_columns: Optional[List[str]] = None,
    fontsize: int = 8,
    row_height: float = 0.35,
    pagesize: Tuple[float, float] = (11.0, 8.5),
    compact: bool = False,
    include_index: bool = True,
) -> None:
    """Add a DataFrame as a single PDF page."""
    fig = _df_to_figure(
        df,
        title,
        subtitle=subtitle,
        gradient_columns=gradient_columns,
        fontsize=fontsize,
        row_height=row_height,
        pagesize=pagesize,
        compact=compact,
        include_index=include_index,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load_esa_index(start_date: str, end_date: str) -> pd.Series:
    """Download ESA Index (E-mini S&P 500) px_last from Bloomberg."""
    from xbbg import blp
    df = blp.bdh(tickers="ESA Index", flds=["px_last"], start_date=start_date, end_date=end_date)
    df = _normalize_bdh(df, tickers="ESA Index")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0].sort_index().rename("ESA")


def load_ct5_yields(start_date: str = "2018-05-21", end_date: Optional[str] = None) -> pd.Series:
    """Download CT5 proxy (US 5-year Treasury yield) px_last from Bloomberg."""
    from xbbg import blp
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    df = blp.bdh(
        tickers="USGG5YR Index",
        flds=["px_last"],
        start_date=start_date,
        end_date=end_date,
    )
    df = _normalize_bdh(df, tickers="USGG5YR Index")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0].sort_index().rename("CT5")


def compute_usd_esa_correlation(df_swap: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Compute correlation between USD swap spreads and the ESA Index.

    Returns, for each USD tenor:
        - level_corr: full-sample correlation of levels
        - chg_corr: full-sample correlation of daily changes
        - rolling_corr: trailing `window`-day correlation of daily changes
    """
    usd_cols = [c for c in df_swap.columns if c.startswith("USD_")]
    if not usd_cols:
        return pd.DataFrame()

    start = df_swap.index.min().strftime("%Y-%m-%d")
    end = df_swap.index.max().strftime("%Y-%m-%d")
    try:
        esa = load_esa_index(start, end)
    except Exception:
        return pd.DataFrame()

    data = df_swap[usd_cols].join(esa, how="inner")
    if data.empty or len(data) < window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    rows = []
    for col in usd_cols:
        tenor = col.replace("USD_", "")
        rows.append({
            "tenor": tenor,
            "level_corr": data[col].corr(data["ESA"]),
            "chg_corr": chg[col].corr(chg["ESA"]),
            f"rolling_{window}d_corr": chg[col].rolling(window=window, min_periods=window).corr(chg["ESA"]).iloc[-1],
        })
    return pd.DataFrame(rows).set_index("tenor")


def compute_ct5_esa_correlation(windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between CT5 yields (5y Treasury) and the ESA Index.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    try:
        ct5 = load_ct5_yields()
    except Exception:
        return pd.DataFrame()

    start = ct5.index.min().strftime("%Y-%m-%d")
    end = ct5.index.max().strftime("%Y-%m-%d")
    try:
        esa = load_esa_index(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([ct5, esa], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["CT5"].corr(data["ESA"]),
        "chg_corr": chg["CT5"].corr(chg["ESA"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["CT5"].rolling(window=w, min_periods=w).corr(chg["ESA"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["CT5 vs ESA"])


def load_cdx_ig_5y(start_date: str, end_date: str) -> pd.Series:
    """Download CDX IG 5Y (CDS index) px_last from Bloomberg."""
    from xbbg import blp
    df = blp.bdh(
        tickers="CDX IG CDSI GEN 5Y Corp",
        flds=["px_last"],
        start_date=start_date,
        end_date=end_date,
    )
    df = _normalize_bdh(df, tickers="CDX IG CDSI GEN 5Y Corp")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0].sort_index().rename("CDX_IG_5Y")


def load_fncc105_zv(start_date: str, end_date: str) -> pd.Series:
    """Download FNCC current-coupon ZV spread (.FNCC105 G Index) from Bloomberg."""
    from xbbg import blp
    df = blp.bdh(
        tickers=".FNCC105 G Index",
        flds=["px_last"],
        start_date=start_date,
        end_date=end_date,
    )
    df = _normalize_bdh(df, tickers=".FNCC105 G Index")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)
    df.index = pd.to_datetime(df.index)
    # px_last is in percent points; convert to bps to match report convention
    return (df.iloc[:, 0] * 100.0).sort_index().rename("FNCC105_ZV")


def compute_cdx_esa_correlation(windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between CDX IG 5Y and the ESA Index.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    start = "2018-05-21"
    end = date.today().strftime("%Y-%m-%d")
    try:
        cdx = load_cdx_ig_5y(start, end)
        esa = load_esa_index(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([cdx, esa], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["CDX_IG_5Y"].corr(data["ESA"]),
        "chg_corr": chg["CDX_IG_5Y"].corr(chg["ESA"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["CDX_IG_5Y"].rolling(window=w, min_periods=w).corr(chg["ESA"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["CDX IG 5Y vs ESA"])


def compute_fncc105_esa_correlation(windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between FNCC current-coupon ZV spread and the ESA Index.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    start = "2018-05-21"
    end = date.today().strftime("%Y-%m-%d")
    try:
        fncc = load_fncc105_zv(start, end)
        esa = load_esa_index(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([fncc, esa], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["FNCC105_ZV"].corr(data["ESA"]),
        "chg_corr": chg["FNCC105_ZV"].corr(chg["ESA"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["FNCC105_ZV"].rolling(window=w, min_periods=w).corr(chg["ESA"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["FNCC105 ZV vs ESA"])


def compute_usd5y_cdx_correlation(df_swap: pd.DataFrame, windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between USD 5Y swap spread and CDX IG 5Y.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    if "USD_5Y" not in df_swap.columns:
        return pd.DataFrame()

    start = df_swap.index.min().strftime("%Y-%m-%d")
    end = df_swap.index.max().strftime("%Y-%m-%d")
    try:
        cdx = load_cdx_ig_5y(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([df_swap["USD_5Y"].rename("USD_5Y"), cdx], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["USD_5Y"].corr(data["CDX_IG_5Y"]),
        "chg_corr": chg["USD_5Y"].corr(chg["CDX_IG_5Y"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["USD_5Y"].rolling(window=w, min_periods=w).corr(chg["CDX_IG_5Y"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["USD 5Y Swap Spread vs CDX IG 5Y"])


def compute_ct5_usd5y_correlation(df_swap: pd.DataFrame, windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between CT5 yields and USD 5Y swap spread.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    if "USD_5Y" not in df_swap.columns:
        return pd.DataFrame()

    start = df_swap.index.min().strftime("%Y-%m-%d")
    end = df_swap.index.max().strftime("%Y-%m-%d")
    try:
        ct5 = load_ct5_yields(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([ct5, df_swap["USD_5Y"].rename("USD_5Y")], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["CT5"].corr(data["USD_5Y"]),
        "chg_corr": chg["CT5"].corr(chg["USD_5Y"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["CT5"].rolling(window=w, min_periods=w).corr(chg["USD_5Y"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["CT5 vs USD 5Y Swap Spread"])


def compute_usd5y_esa_correlation(df_swap: pd.DataFrame, windows: List[int] = [20, 60, 90]) -> pd.DataFrame:
    """
    Compute correlation between USD 5Y swap spread and the ESA Index.

    Returns full-sample level / daily-change correlations plus trailing
    correlations for each window in `windows`.
    """
    if "USD_5Y" not in df_swap.columns:
        return pd.DataFrame()

    start = df_swap.index.min().strftime("%Y-%m-%d")
    end = df_swap.index.max().strftime("%Y-%m-%d")
    try:
        esa = load_esa_index(start, end)
    except Exception:
        return pd.DataFrame()

    data = pd.concat([df_swap["USD_5Y"].rename("USD_5Y"), esa], axis=1, join="inner")
    max_window = max(windows)
    if data.empty or len(data) < max_window + 1:
        return pd.DataFrame()

    chg = data.diff().dropna()
    row = {
        "level_corr": data["USD_5Y"].corr(data["ESA"]),
        "chg_corr": chg["USD_5Y"].corr(chg["ESA"]),
    }
    for w in windows:
        row[f"rolling_{w}d_corr"] = (
            chg["USD_5Y"].rolling(window=w, min_periods=w).corr(chg["ESA"]).iloc[-1]
        )
    return pd.DataFrame([row], index=["USD 5Y Swap Spread vs ESA"])


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    df: pd.DataFrame,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate the swap spread PDF report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"SwapSpread_Report_{report_date}.pdf"

    pct_summary = compute_percentile_summary(df, window=PERCENTILE_WINDOW)
    zscore_summary = compute_zscore_summary(df, windows=ZSCORE_WINDOWS)

    pct_cols = ["latest"] + [f"{prefix}_{PERCENTILE_WINDOW}d" for prefix in ("min", "p25", "p50", "p75", "max")] + ["range_pct"]
    pct_summary = pct_summary[pct_cols]

    zscore_cols = (["latest"]
                   + [f"z_{w}d" for w in ZSCORE_WINDOWS]
                   + [f"ann_std_{w}d" for w in ZSCORE_WINDOWS]
                   + [f"sharpe_{w}d" for w in ZSCORE_WINDOWS])
    zscore_summary = zscore_summary[zscore_cols]

    def _render(path: Path) -> Path:
        with PdfPages(path) as pdf:
            # Contents page
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            ax.axis("off")
            title = "Swap Spread Report"
            subtitle = f"Latest date: {df.index[-1].date()}  |  Observations: {len(df)}"
            ax.text(0.5, 0.90, title, fontsize=24, weight="bold", ha="center", transform=ax.transAxes)
            ax.text(0.5, 0.84, subtitle, fontsize=12, ha="center", color="gray", transform=ax.transAxes)
            content_lines = ["Report contents\n"]
            content_lines.append("1. USD Swap Spreads vs ESA Index — correlation")
            content_lines.append("2. Cross-asset correlations (CT5, CDX IG 5Y, FNCC105, USD 5Y swap spread vs ESA/CDX/CT5)")
            content_lines.append("")
            for i, ccy in enumerate(CURRENCIES, start=3):
                content_lines.append(f"{i}. {ccy} swap spreads: 250-day percentiles")
                content_lines.append(f"{i+len(CURRENCIES)}. {ccy} swap spreads: 20/60/90-day z-scores & annualized std")
            contents = "\n".join(content_lines)
            ax.text(0.5, 0.55, contents, fontsize=12, family="monospace",
                    ha="center", va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#40466e"))
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            latest_date = df.index[-1].date()

            # USD swap spreads vs ESA Index correlation section
            corr_df = compute_usd_esa_correlation(df, window=90)
            if not corr_df.empty:
                _add_table_page(
                    pdf,
                    corr_df,
                    "USD Swap Spreads vs ESA Index — Correlation",
                    subtitle=f"Daily changes  |  90-day rolling correlation  |  Latest: {latest_date}",
                    fontsize=10,
                    row_height=0.12,
                    pagesize=(11.0, 5.0),
                    compact=True,
                )

            # Cross-asset correlation section
            ct5_corr = compute_ct5_esa_correlation(windows=[20, 60, 90])
            cdx_corr = compute_cdx_esa_correlation(windows=[20, 60, 90])
            fncc_corr = compute_fncc105_esa_correlation(windows=[20, 60, 90])
            usd5y_esa_corr = compute_usd5y_esa_correlation(df, windows=[20, 60, 90])
            usd5y_cdx_corr = compute_usd5y_cdx_correlation(df, windows=[20, 60, 90])
            ct5_usd5y_corr = compute_ct5_usd5y_correlation(df, windows=[20, 60, 90])
            combined_corr = pd.concat([ct5_corr, cdx_corr, fncc_corr, usd5y_esa_corr, usd5y_cdx_corr, ct5_usd5y_corr])
            if not combined_corr.empty:
                _add_table_page(
                    pdf,
                    combined_corr,
                    "Cross-Asset Correlations",
                    subtitle=f"Daily changes  |  20/60/90-day rolling correlations  |  Latest: {latest_date}",
                    fontsize=10,
                    row_height=0.12,
                    pagesize=(11.0, 5.0),
                    compact=True,
                )

            for ccy in CURRENCIES:
                ccy_pct = pct_summary[pct_summary.index.str.startswith(f"{ccy}_")]
                ccy_z = zscore_summary[zscore_summary.index.str.startswith(f"{ccy}_")]

                if not ccy_pct.empty:
                    _add_table_page(
                        pdf,
                        ccy_pct,
                        f"{ccy} Swap Spreads — {PERCENTILE_WINDOW}-Day Percentiles",
                        subtitle=f"Latest: {latest_date}  |  Values in bps",
                        fontsize=9,
                        row_height=0.08,
                        pagesize=(11.0, 5.0),
                        compact=True,
                    )

                if not ccy_z.empty:
                    _add_table_page(
                        pdf,
                        ccy_z,
                        f"{ccy} Swap Spreads — Z-Scores & Annualized Std",
                        subtitle=(f"Latest: {latest_date}  |  Rolling windows: {', '.join(map(str, ZSCORE_WINDOWS))} days  |  "
                                  f"Ann Std = std(level) * sqrt(252) in bps  |  "
                                  f"Sharpe = (-latest - 10bps) / ann_std"),
                        gradient_columns=([f"z_{w}d" for w in ZSCORE_WINDOWS]
                                          + [f"ann_std_{w}d" for w in ZSCORE_WINDOWS]
                                          + [f"sharpe_{w}d" for w in ZSCORE_WINDOWS]),
                        fontsize=9,
                        row_height=0.08,
                        pagesize=(12.0, 5.0),
                        compact=True,
                    )

        return path

    try:
        return _render(pdf_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = OUTPUT_DIR / f"SwapSpread_Report_{ts}.pdf"
        print(f"Warning: could not write to {pdf_path} (file may be open). Saving to {fallback_path}")
        return _render(fallback_path)


# ---------------------------------------------------------------------------
# Interactive HTML report
# ---------------------------------------------------------------------------
INTERACTIVE_TITLE = "Swap Spread Report"
RECENT_DAYS = 365  # trailing window for the per-country "recent" chart
ROLL_CORR_WINDOW = 90
ROLL_CORR_TAIL = 500  # observations shown on the rolling-correlation chart

CCY_COLORS = {
    "USD": "#dc2626",
    "Bund": "#d97706",
    "OAT": "#2563eb",
    "Gilt": "#059669",
    "BTP": "#7c3aed",
    "JGB": "#0891b2",
}
TENOR_COLORS = {
    "2Y": "#2563eb",
    "3Y": "#0891b2",
    "5Y": "#059669",
    "7Y": "#d97706",
    "10Y": "#dc2626",
    "20Y": "#7c3aed",
    "30Y": "#8b5cf6",
}


def _xs(index) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _ys(series) -> List[Optional[float]]:
    return [None if pd.isna(v) else float(v) for v in series]


def _apply_layout(fig: "go.Figure", title: str, height: int = 520, subtitle: Optional[str] = None) -> "go.Figure":
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


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.2f}", include_index: bool = True) -> str:
    """Render a DataFrame as an HTML table (toolbar wired up in the page JS)."""
    if include_index:
        df = df.reset_index()
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "")

    parts = ['<div class="table-wrap"><table class="data-table"><thead><tr>']
    for col in display.columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")
    for i in range(len(display)):
        parts.append("<tr>" + "".join(
            f"<td>{html.escape(str(display[col].iloc[i]))}</td>" for col in display.columns
        ) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _ccy_tenor_cols(df: pd.DataFrame, ccy: str) -> List[Tuple[str, str]]:
    """(column, tenor) pairs available for one currency, in tenor order."""
    out = []
    for tenor in TENORS:
        col = f"{ccy}_{tenor}"
        if col in df.columns and df[col].notna().any():
            out.append((col, tenor))
    return out


def _fig_overlay_tenor(df: pd.DataFrame, tenor: str) -> "go.Figure":
    fig = go.Figure()
    for ccy in CURRENCIES:
        col = f"{ccy}_{tenor}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_xs(s.index), y=_ys(s), mode="lines", name=ccy,
                line=dict(color=CCY_COLORS.get(ccy, "#475569"), width=1.3),
            )
        )
    fig.update_yaxes(title_text="Spread (bp)")
    _apply_layout(fig, f"{tenor} Swap Spreads - Full History", height=520)
    return fig


def _fig_overlay_2y_30y(df: pd.DataFrame) -> "go.Figure":
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["2Y swap spreads", "30Y swap spreads"])
    for ccy in CURRENCIES:
        color = CCY_COLORS.get(ccy, "#475569")
        for row, tenor in ((1, "2Y"), (2, "30Y")):
            col = f"{ccy}_{tenor}"
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_xs(s.index), y=_ys(s), mode="lines", name=ccy,
                    line=dict(color=color, width=1.3),
                    legendgroup=ccy, showlegend=(row == 1),
                ),
                row=row, col=1,
            )
    fig.update_yaxes(title_text="Spread (bp)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (bp)", row=2, col=1)
    _apply_layout(fig, "2Y and 30Y Swap Spreads - Full History", height=760)
    return fig


def _fig_country_history(df: pd.DataFrame, ccy: str, days: Optional[int] = None) -> "go.Figure":
    fig = go.Figure()
    for col, tenor in _ccy_tenor_cols(df, ccy):
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_xs(s.index), y=_ys(s), mode="lines", name=tenor,
                line=dict(color=TENOR_COLORS.get(tenor, "#475569"), width=1.3),
            )
        )
    title = f"{ccy} Swap Spreads by Tenor"
    if days is not None:
        end = df.index[-1]
        start = end - pd.Timedelta(days=days)
        fig.update_xaxes(range=[start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")])
        title += f" - Trailing {days} Days"
    else:
        title += " - Full History"
    fig.update_yaxes(title_text="Spread (bp)")
    _apply_layout(fig, title, height=520)
    return fig


def _fig_country_zscore(df: pd.DataFrame, ccy: str, window: int = 20, days: int = RECENT_DAYS) -> "go.Figure":
    fig = go.Figure()
    for col, tenor in _ccy_tenor_cols(df, ccy):
        s = df[col].dropna()
        if len(s) < window:
            continue
        mean = s.rolling(window=window, min_periods=window).mean()
        std = s.rolling(window=window, min_periods=window).std()
        z = ((s - mean) / std.replace(0, np.nan)).dropna()
        if z.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=_xs(z.index), y=_ys(z), mode="lines", name=tenor,
                line=dict(color=TENOR_COLORS.get(tenor, "#475569"), width=1.2),
            )
        )
    end = df.index[-1]
    start = end - pd.Timedelta(days=days)
    x0, x1 = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    fig.update_xaxes(range=[x0, x1])
    for level, color, dash in ((0, "#111827", "solid"), (2, "#b91c1c", "dash"), (-2, "#b91c1c", "dash")):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                      line=dict(color=color, width=0.9, dash=dash))
    fig.update_yaxes(title_text=f"Z-score ({window}d)")
    _apply_layout(
        fig, f"{ccy} Swap Spreads - {window}d Rolling Z-Score", height=440,
        subtitle=f"Trailing {days} days  |  z = (level - {window}d mean) / {window}d std",
    )
    return fig


def _fig_rolling_correlations(df: pd.DataFrame) -> Optional["go.Figure"]:
    """Rolling 90d correlation of daily changes: USD 10Y vs ESA, USD 5Y vs CDX IG 5Y."""
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    fig = go.Figure()
    added = 0

    if "USD_10Y" in df.columns:
        try:
            esa = load_esa_index(start, end)
            data = pd.concat([df["USD_10Y"].rename("spr"), esa], axis=1, join="inner").diff().dropna()
            rc = (data["spr"].rolling(ROLL_CORR_WINDOW, min_periods=ROLL_CORR_WINDOW)
                  .corr(data["ESA"]).dropna().tail(ROLL_CORR_TAIL))
            if not rc.empty:
                fig.add_trace(
                    go.Scatter(x=_xs(rc.index), y=_ys(rc), mode="lines",
                               name=f"USD 10Y vs ESA ({ROLL_CORR_WINDOW}d)",
                               line=dict(color="#dc2626", width=1.3))
                )
                added += 1
        except Exception as exc:
            print(f"Rolling correlation 'USD 10Y vs ESA' skipped: {exc}")

    if "USD_5Y" in df.columns:
        try:
            cdx = load_cdx_ig_5y(start, end)
            data = pd.concat([df["USD_5Y"].rename("spr"), cdx], axis=1, join="inner").diff().dropna()
            rc = (data["spr"].rolling(ROLL_CORR_WINDOW, min_periods=ROLL_CORR_WINDOW)
                  .corr(data["CDX_IG_5Y"]).dropna().tail(ROLL_CORR_TAIL))
            if not rc.empty:
                fig.add_trace(
                    go.Scatter(x=_xs(rc.index), y=_ys(rc), mode="lines",
                               name=f"USD 5Y vs CDX IG 5Y ({ROLL_CORR_WINDOW}d)",
                               line=dict(color="#2563eb", width=1.3))
                )
                added += 1
        except Exception as exc:
            print(f"Rolling correlation 'USD 5Y vs CDX IG 5Y' skipped: {exc}")

    if added == 0:
        return None
    fig.add_hline(0, line=dict(color="#111827", width=0.8))
    fig.update_yaxes(title_text="Correlation", range=[-1.05, 1.05])
    _apply_layout(
        fig, f"Rolling {ROLL_CORR_WINDOW}d Correlation of Daily Changes", height=440,
        subtitle=f"Last {ROLL_CORR_TAIL} observations",
    )
    return fig


def _safe_corr(fn, label: str, *args, **kwargs) -> pd.DataFrame:
    """Run a correlation summary, returning an empty frame on any failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"Correlation '{label}' skipped: {exc}")
        return pd.DataFrame()


class _HtmlBuilder:
    """Collects figures as JSON and emits self-contained section HTML."""

    def __init__(self) -> None:
        self._fig_count = 0

    def fig_block(self, fig: "go.Figure") -> str:
        self._fig_count += 1
        fid = f"fig-{self._fig_count:03d}"
        fig_json = pio.to_json(fig).replace("</", "<\\/")
        return (
            f'<div class="chart" id="{fid}"></div>'
            f'<script type="application/json" class="plotly-fig" data-target="{fid}">'
            f"{fig_json}</script>"
        )


_HTML_CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#d9e2ec;--panel:#fff;--bg:#eef2f6;--accent:#174a6e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}
.toggle-bar{position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:8px;flex-wrap:wrap;
background:#102b3c;color:#e7f0f5;padding:10px 18px;box-shadow:0 2px 8px rgba(15,23,42,.25)}
.tb-title{font-size:16px;font-weight:bold;margin-right:14px}
.tb-date{font-size:12px;color:#b9cad5;margin-right:14px}
.ccy-btn{border:1px solid #507084;background:#1e4b65;color:#dce9f0;padding:6px 16px;border-radius:6px;
cursor:pointer;font-size:13px;font-weight:bold}
.ccy-btn:hover{background:#2a6284}
.ccy-btn.active{background:#fff;color:#102b3c;border-color:#fff}
main{padding:20px;max-width:1500px;margin:0 auto}
.ccy-section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
box-shadow:0 3px 12px rgba(15,23,42,.06);padding:16px 18px}
.ccy-section h2{font-size:18px;margin:2px 0 12px;color:var(--accent)}
.ccy-section h3{font-size:15px;margin:16px 0 8px;color:var(--ink)}
.chart{width:100%;margin-bottom:18px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px;margin-bottom:14px}
.table-tools{display:flex;gap:8px;align-items:center;margin:2px 0 6px;font-size:12px}
.table-tools input[type=search]{padding:5px 8px;border:1px solid #b6c5cf;border-radius:6px;font-size:12px;min-width:180px}
.table-tools button{border:1px solid #b6c5cf;background:#fff;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px}
.table-count{color:var(--muted)}
.data-table{border-collapse:collapse;font-size:12px;min-width:100%}
.data-table th{background:#40466e;color:#fff;padding:5px 8px;font-weight:bold;text-align:center;white-space:nowrap;
cursor:pointer;user-select:none}
.data-table th.sort-asc::after{content:" ▲";font-size:9px}
.data-table th.sort-desc::after{content:" ▼";font-size:9px}
.data-table td{border:1px solid #e2e8f0;padding:4px 8px;text-align:center;white-space:nowrap}
.data-table tbody tr:nth-child(even){background:#f0f0f0}
h4.ovh{font-size:13px;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
@media print{.toggle-bar{position:static}.ccy-section{display:block !important;break-after:page}}
"""

_HTML_JS = """
const FIGS={};
document.querySelectorAll('script.plotly-fig').forEach(s=>{FIGS[s.dataset.target]=s.textContent;});
function renderChart(id){const el=document.getElementById(id);if(!el||el.dataset.rendered||!FIGS[id])return;const f=JSON.parse(FIGS[id]);el.dataset.rendered="1";Plotly.newPlot(el,f.data,f.layout,{responsive:true,displaylogo:false});}
function renderSection(sec){if(!sec)return;sec.querySelectorAll('.chart').forEach(el=>renderChart(el.id));}
function showCcy(c){
document.querySelectorAll('.ccy-section').forEach(s=>{s.style.display=(s.id==='ccy-'+c)?'':'none';});
document.querySelectorAll('.ccy-btn').forEach(b=>{b.classList.toggle('active',b.dataset.ccy===c);});
renderSection(document.getElementById('ccy-'+c));
window.scrollTo({top:0,behavior:'smooth'});
}
renderSection(document.getElementById('ccy-all'));
function cellValue(cell){const text=cell.textContent.trim().replace(/,/g,'');if(text==='')return {blank:true,num:false,value:''};const n=Number(text.replace(/%$/,''));return Number.isFinite(n)?{blank:false,num:true,value:n}:{blank:false,num:false,value:text.toLowerCase()};}
function csvCell(value){return '"'+value.replace(/"/g,'""')+'"';}
document.querySelectorAll('.data-table').forEach((table,tableIndex)=>{
const wrap=table.closest('.table-wrap');if(!wrap)return;const rows=[...table.tBodies[0].rows];const tools=document.createElement('div');tools.className='table-tools';
const search=document.createElement('input');search.type='search';search.placeholder='Filter this table';search.setAttribute('aria-label','Filter this table');
const reset=document.createElement('button');reset.type='button';reset.textContent='Reset sort';
const csv=document.createElement('button');csv.type='button';csv.textContent='Download CSV';
const count=document.createElement('span');count.className='table-count';tools.append(search,reset,csv,count);wrap.parentNode.insertBefore(tools,wrap);
const refresh=()=>{let shown=0;rows.forEach(row=>{if(row.style.display!=='none')shown++;});count.textContent=`${shown} of ${rows.length} rows`;};refresh();
search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();rows.forEach(row=>{row.style.display=(!q||row.textContent.toLowerCase().includes(q))?'':'none';});refresh();});
table.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{const asc=!th.classList.contains('sort-asc');table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));th.classList.add(asc?'sort-asc':'sort-desc');const sorted=[...rows].sort((a,b)=>{const av=cellValue(a.cells[col]),bv=cellValue(b.cells[col]);if(av.blank!==bv.blank)return av.blank?1:-1;let c;if(av.num&&bv.num)c=av.value-bv.value;else c=String(av.value).localeCompare(String(bv.value));return asc?c:-c;});sorted.forEach(row=>table.tBodies[0].appendChild(row));}));
reset.addEventListener('click',()=>{rows.forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));});
csv.addEventListener('click',()=>{const lines=[];lines.push([...table.tHead.rows[0].cells].map(c=>csvCell(c.textContent.trim())).join(','));[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').forEach(r=>lines.push([...r.cells].map(c=>csvCell(c.textContent.trim())).join(',')));const blob=new Blob([lines.join('\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`swap-spread-table-${String(tableIndex+1).padStart(2,'0')}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
});
"""


def _overview_section_html(
    builder: _HtmlBuilder,
    df: pd.DataFrame,
    pct_summary: pd.DataFrame,
    zscore_summary: pd.DataFrame,
) -> str:
    parts = [
        "<h2>Overview - All Markets</h2>",
        builder.fig_block(_fig_overlay_tenor(df, "10Y")),
        builder.fig_block(_fig_overlay_2y_30y(df)),
        f'<h4 class="ovh">{PERCENTILE_WINDOW}-day percentile summary (bp)</h4>',
        _df_to_html_table(pct_summary),
        '<h4 class="ovh">Z-score summary (20/60/90d)</h4>',
        _df_to_html_table(zscore_summary),
    ]
    return "".join(parts)


def _country_section_html(
    builder: _HtmlBuilder,
    df: pd.DataFrame,
    ccy: str,
    pct_summary: pd.DataFrame,
    zscore_summary: pd.DataFrame,
) -> str:
    parts = [
        f"<h2>{html.escape(ccy)} Swap Spreads</h2>",
        builder.fig_block(_fig_country_history(df, ccy)),
        builder.fig_block(_fig_country_history(df, ccy, days=RECENT_DAYS)),
        builder.fig_block(_fig_country_zscore(df, ccy)),
    ]
    ccy_pct = pct_summary[pct_summary.index.str.startswith(f"{ccy}_")]
    ccy_z = zscore_summary[zscore_summary.index.str.startswith(f"{ccy}_")]
    if not ccy_pct.empty:
        parts.append(f'<h4 class="ovh">{ccy} {PERCENTILE_WINDOW}-day percentiles (bp)</h4>')
        parts.append(_df_to_html_table(ccy_pct))
    if not ccy_z.empty:
        parts.append(f'<h4 class="ovh">{ccy} z-scores &amp; annualized std</h4>')
        parts.append(_df_to_html_table(ccy_z))
    if ccy == "USD":
        parts.append(_correlation_section_html(builder, df))
    return "".join(parts)


def _correlation_section_html(builder: _HtmlBuilder, df: pd.DataFrame) -> str:
    latest_date = df.index[-1].date()
    parts = ['<div class="corr-block"><h3>Correlations</h3>']

    usd_esa = _safe_corr(compute_usd_esa_correlation, "USD vs ESA", df, window=90)
    if not usd_esa.empty:
        parts.append('<h4 class="ovh">USD swap spreads vs ESA Index</h4>')
        parts.append(f'<p class="fib-note" style="font-size:12px;color:#64748b">'
                     f'Daily changes  |  90-day rolling correlation  |  Latest: {latest_date}</p>')
        parts.append(_df_to_html_table(usd_esa))

    combined = pd.concat([
        _safe_corr(compute_ct5_esa_correlation, "CT5 vs ESA", windows=[20, 60, 90]),
        _safe_corr(compute_cdx_esa_correlation, "CDX vs ESA", windows=[20, 60, 90]),
        _safe_corr(compute_fncc105_esa_correlation, "FNCC105 vs ESA", windows=[20, 60, 90]),
        _safe_corr(compute_usd5y_esa_correlation, "USD5Y vs ESA", df, windows=[20, 60, 90]),
        _safe_corr(compute_usd5y_cdx_correlation, "USD5Y vs CDX", df, windows=[20, 60, 90]),
        _safe_corr(compute_ct5_usd5y_correlation, "CT5 vs USD5Y", df, windows=[20, 60, 90]),
    ])
    if not combined.empty:
        parts.append('<h4 class="ovh">Cross-asset correlations</h4>')
        parts.append(f'<p class="fib-note" style="font-size:12px;color:#64748b">'
                     f'Daily changes  |  20/60/90-day rolling correlations  |  Latest: {latest_date}</p>')
        parts.append(_df_to_html_table(combined))

    rc_fig = _fig_rolling_correlations(df)
    if rc_fig is not None:
        parts.append(builder.fig_block(rc_fig))

    parts.append("</div>")
    return "".join(parts)


def build_interactive_html(
    df: pd.DataFrame,
    pct_summary: pd.DataFrame,
    zscore_summary: pd.DataFrame,
    report_path: Path,
) -> Path:
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is required for the interactive HTML report.")

    builder = _HtmlBuilder()
    as_of = df.index[-1].date()

    overview = _overview_section_html(builder, df, pct_summary, zscore_summary)
    section_parts = [
        f'<section class="ccy-section" id="ccy-all" style="display:block">{overview}</section>'
    ]
    for ccy in CURRENCIES:
        slug = ccy.lower()
        body = _country_section_html(builder, df, ccy, pct_summary, zscore_summary)
        section_parts.append(
            f'<section class="ccy-section" id="ccy-{slug}" style="display:none">{body}</section>'
        )

    buttons = [
        '<button class="ccy-btn active" data-ccy="all" onclick="showCcy(\'all\')">ALL</button>'
    ]
    for ccy in CURRENCIES:
        slug = ccy.lower()
        buttons.append(
            f'<button class="ccy-btn" data-ccy="{slug}" '
            f'onclick="showCcy(\'{slug}\')">{html.escape(ccy)}</button>'
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(INTERACTIVE_TITLE)} (Interactive)</title>
<script>{get_plotlyjs()}</script>
<style>{_HTML_CSS}</style>
</head>
<body>
<nav class="toggle-bar"><span class="tb-title">{html.escape(INTERACTIVE_TITLE)}</span>
<span class="tb-date">as of {as_of}</span>{''.join(buttons)}</nav>
<main>{''.join(section_parts)}</main>
<script>{_HTML_JS}</script>
</body></html>"""

    try:
        report_path.write_text(document, encoding="utf-8")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_path.with_name(
            f"{report_path.stem}_{ts}{report_path.suffix}"
        )
        print(f"Warning: could not write dated HTML (file may be open). Saving to {report_path}")
        report_path.write_text(document, encoding="utf-8")
    return report_path


def generate_interactive_report(
    df: pd.DataFrame,
    pct_summary: Optional[pd.DataFrame] = None,
    zscore_summary: Optional[pd.DataFrame] = None,
) -> Path:
    """Build the interactive HTML swap spread report."""
    if pct_summary is None:
        pct_summary = compute_percentile_summary(df)
    if zscore_summary is None:
        zscore_summary = compute_zscore_summary(df)
    report_date = df.index[-1].strftime("%Y%m%d")
    report_path = OUTPUT_DIR / f"SwapSpread_Report_{report_date}.html"
    return build_interactive_html(df, pct_summary, zscore_summary, report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(report_format=("interactive",)) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_swap_spread_data(FNAME)

    pct_summary = compute_percentile_summary(df)
    zscore_summary = compute_zscore_summary(df)

    print("\n250-Day Percentile Summary")
    print(pct_summary.to_string())
    print("\nZ-Score Summary")
    print(zscore_summary.to_string())

    formats = [report_format] if isinstance(report_format, str) else list(report_format)
    for fmt in formats:
        fmt = fmt.lower()
        if fmt == "interactive":
            html_path = generate_interactive_report(df, pct_summary, zscore_summary)
            print(f"\nInteractive HTML report saved to: {html_path}")
        else:
            pdf_path = generate_pdf_report(df)
            print(f"\nPDF report saved to: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the swap spread report.")
    parser.add_argument(
        "--format",
        nargs="+",
        choices=("interactive", "pdf"),
        default=["interactive"],
        help="Report output format(s): interactive and/or pdf (default: interactive).",
    )
    args = parser.parse_args()
    main(report_format=args.format)
