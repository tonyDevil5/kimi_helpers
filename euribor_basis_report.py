r"""
Swap Clearing Basis Report
===========================

Analyzes the past 1-year basis for key tenors across three clearing pairs:
    - LCH vs Eurex EURIBOR 6M basis
    - LCH vs CME SOFR basis
    - LCH vs JSCC TONAR basis

Data source: Bloomberg (xbbg)
    LCH/Eurex EURIBOR: EULCEX2 Curncy, ..., EULCEX30 Curncy
    LCH/CME SOFR:      USLCSS2 Curncy, ..., USLCSS30 Curncy
    LCH/JSCC TONAR:    JYJLO2 Curncy, ..., JYJLO30 Curncy

Output:
    ./euribor_basis_output/Basis_Report.pdf

Run:
    python euribor_basis_report.py
"""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "euribor_basis_output"

CITI_FPATH = Path(r"D:\python\notebook\data\mbs\mbs_rv_citi.xlsx")

TENORS = ["2y", "3y", "4y", "5y", "7y", "10y", "20y", "30y"]

SHEET5_TENORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]

BASIS_CONFIG = {
    "LCH / Eurex EURIBOR 6M": {
        "tickers": {t: f"EULCEX{t[:-1]} Curncy" for t in TENORS},
        "description": "LCH - Eurex EURIBOR 6M swap spread (bp)",
    },
    "LCH / CME SOFR": {
        "tickers": {t: f"USLCSS{t[:-1]} Curncy" for t in TENORS},
        "description": "LCH - CME SOFR swap spread (bp)",
    },
    "LCH / JSCC TONAR": {
        "tickers": {t: f"JYJLO{t[:-1]} Curncy" for t in TENORS},
        "description": "LCH - JSCC TONAR swap spread (bp)",
    },
}

LOOKBACK_DAYS = 365  # ~1 year of calendar days


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_sheet5_tenor(col_name: str) -> str:
    m = re.search(r"SPOT\.(\d+)Y", col_name)
    if m:
        return f"{int(m.group(1))}Y"
    return col_name


def load_sheet5_basis_data(fpath: Path = CITI_FPATH) -> Dict[str, pd.DataFrame]:
    """Load EURIBOR 3S6S and SOFR/FedFunds spot basis from Sheet5 of mbs_rv_citi.xlsx."""
    df = pd.read_excel(fpath, sheet_name="Sheet5", header=1)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    eur_cols = [c for c in df.columns if "3S6S_BASIS.EUR" in c]
    usd_cols = [c for c in df.columns if "SOFR_FEDFUND_BASIS.USD" in c]

    eur = df[eur_cols].rename(columns=_parse_sheet5_tenor)
    usd = df[usd_cols].rename(columns=_parse_sheet5_tenor)

    # Keep only expected tenor columns and sort ascending
    eur = eur[[f"{t}Y" for t in SHEET5_TENORS if f"{t}Y" in eur.columns]]
    usd = usd[[f"{t}Y" for t in SHEET5_TENORS if f"{t}Y" in usd.columns]]

    print(f"Loaded Sheet5 basis data: {df.shape[0]} rows, "
          f"EUR tenors={len(eur.columns)}, USD tenors={len(usd.columns)}")
    return {
        "EURIBOR 3S6S Basis": eur,
        "SOFR / FedFunds Basis": usd,
    }


def compute_adjacent_tenor_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Compute differences between adjacent tenors (e.g. 2Y-1Y, 3Y-2Y, ...)."""
    tenors = [int(c[:-1]) for c in df.columns if c.endswith("Y")]
    tenors = sorted(tenors)
    spreads = {}
    for i in range(1, len(tenors)):
        low, high = tenors[i - 1], tenors[i]
        spreads[f"{high}Y-{low}Y"] = df[f"{high}Y"] - df[f"{low}Y"]
    return pd.DataFrame(spreads)


def build_spread_summary(spreads: pd.DataFrame, windows: Tuple[int, ...] = (90, 250)) -> pd.DataFrame:
    """Compute latest spread, rolling z-scores, and 250-day min/max for each adjacent-tenor spread."""
    rows = []
    for col in spreads.columns:
        series = spreads[col].dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        row = {"spread": col, "latest": latest}
        for w in windows:
            recent = series.tail(w)
            mean = float(recent.mean())
            std = float(recent.std(ddof=1))
            row[f"z_{w}d"] = (latest - mean) / std if std != 0 else np.nan
        recent_250 = series.tail(250)
        row["min_250d"] = float(recent_250.min())
        row["max_250d"] = float(recent_250.max())
        rows.append(row)
    return pd.DataFrame(rows).set_index("spread")


def load_basis_data(
    tickers: Dict[str, str],
    label: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Pull basis history from Bloomberg for all tenors."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    ticker_list = list(tickers.values())
    print(f"Downloading {len(ticker_list)} {label} basis tickers from Bloomberg...")
    df = blp.bdh(tickers=ticker_list, flds=["px_last"], start_date=start_str, end_date=end_str)

    # xbbg >= 1.x returns a narwhals DataFrame in long format
    # (ticker / date / field / value); older versions return a pandas
    # DataFrame with MultiIndex columns. Normalize both to a wide frame.
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    elif isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().dropna()

    # Rename columns to tenor labels
    inv_map = {v: k for k, v in tickers.items()}
    df = df.rename(columns=inv_map)

    print(f"Loaded {label} basis data: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def build_basis_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute latest basis and 1-year statistics for each tenor."""
    rows = []
    for tenor in TENORS:
        if tenor not in df.columns:
            continue
        series = df[tenor].dropna()
        if series.empty:
            continue

        latest = float(series.iloc[-1])
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        zscore = (latest - mean) / std if std != 0 else np.nan

        q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        emp_pct = (series <= latest).mean()

        rows.append({
            "tenor": tenor,
            "latest": latest,
            "mean_1y": mean,
            "std_1y": std,
            "z_score": zscore,
            "min_1y": float(q[0.0]),
            "p25_1y": float(q[0.25]),
            "p50_1y": float(q[0.5]),
            "p75_1y": float(q[0.75]),
            "max_1y": float(q[1.0]),
            "pct_1y": emp_pct,
            "obs": len(series),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.set_index("tenor")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_basis_history(df: pd.DataFrame, title: str) -> "plt.Figure":
    """Plot 1-year basis history for all tenors in a single chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for tenor in TENORS:
        if tenor not in df.columns:
            continue
        ax.plot(df.index, df[tenor], label=tenor, linewidth=1.2)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Basis (bp)")
    ax.legend(title="Tenor", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: str = "",
    pagesize: Tuple[float, float] = (11.0, 5.5),
    fontsize: int = 8,
    heatmap_columns: Tuple[str, ...] = ("z_score",),
) -> "plt.Figure":
    """Render a DataFrame as a matplotlib table figure.

    heatmap_columns are colored with a diverging red/white/green scale centered at 0.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold", y=0.96, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.91, subtitle, fontsize=8, ha="left",
                transform=fig.transFigure, color="#555555")
    fig.subplots_adjust(top=0.80)

    display_df = df.reset_index().copy()
    numeric_df = display_df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    table_data = [display_df.columns.tolist()] + display_df.values.tolist()

    name_width = 0.10 if n_cols >= 8 else 0.14
    data_width = (1.0 - name_width) / max(1, n_cols - 1)
    if n_cols > 7:
        data_width *= 1.15
        name_width = max(0.08, 1.0 - data_width * (n_cols - 1))
    col_widths = [name_width] + [data_width] * (n_cols - 1)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")
        cell.set_height(0.10)

    # Prepare heatmap colors for requested columns
    cmap = plt.get_cmap("RdYlGn")
    heatmap_col_idx = {
        display_df.columns.get_loc(col): col
        for col in heatmap_columns
        if col in display_df.columns
    }
    heatmap_limits = {}
    for j, col in heatmap_col_idx.items():
        vals = numeric_df.iloc[:, j].dropna().astype(float)
        if not vals.empty:
            limit = max(abs(vals.min()), abs(vals.max()), 0.01)
            heatmap_limits[j] = limit

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            base_color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
            if j in heatmap_col_idx and j in heatmap_limits:
                val = numeric_df.iloc[i - 1, j]
                if pd.notna(val):
                    norm_val = float(val) / heatmap_limits[j]
                    # Clip to [-1, 1] then map to [0, 1] for RdYlGn
                    t = (np.clip(norm_val, -1.0, 1.0) + 1.0) / 2.0
                    rgba = cmap(t)
                    base_color = mcolors.to_hex(rgba)
            cell.set_facecolor(base_color)
            cell.set_height(0.06)

    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    data: Dict[str, pd.DataFrame],
    pdf_path: Path,
    sheet5_data: Dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Generate a PDF report with summary tables and history charts."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PDF report generation.") from exc

    summaries = {name: build_basis_summary(df) for name, df in data.items()}
    if all(s.empty for s in summaries.values()):
        raise ValueError("No basis data available for summary.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for name, df in data.items():
            summary = summaries[name]
            if summary.empty:
                continue

            # Summary table page
            subtitle = (
                f"Latest: {df.index[-1].date()}  |  "
                f"Observations: {len(df)}  |  "
                f"{BASIS_CONFIG[name]['description']}"
            )
            fig = _df_to_figure(
                summary,
                f"{name} Basis Summary (1-Year)",
                subtitle=subtitle,
                pagesize=(13.0, 5.0),
                fontsize=9,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # History chart
            fig = plot_basis_history(df, f"{name} Basis (1-Year History)")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Sheet5 adjacent-tenor spread section
        if sheet5_data:
            for name, df in sheet5_data.items():
                spreads = compute_adjacent_tenor_spreads(df)
                if spreads.empty:
                    continue
                summary = build_spread_summary(spreads, windows=(90, 250))
                if summary.empty:
                    continue

                subtitle = (
                    f"Latest: {df.index[-1].date()}  |  "
                    f"Spreads = higher tenor minus lower tenor  |  "
                    f"Z-scores over 90d and 250d windows"
                )
                fig = _df_to_figure(
                    summary,
                    f"{name} — Adjacent Tenor Spreads",
                    subtitle=subtitle,
                    pagesize=(11.0, 6.5),
                    fontsize=9,
                    heatmap_columns=("z_90d", "z_250d"),
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return pdf_path


def main() -> None:
    data: Dict[str, pd.DataFrame] = {}
    for name, config in BASIS_CONFIG.items():
        try:
            data[name] = load_basis_data(config["tickers"], name)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            data[name] = pd.DataFrame()

    summaries = {name: build_basis_summary(df) for name, df in data.items()}

    # Load Sheet5 adjacent-tenor spread data
    sheet5_data: Dict[str, pd.DataFrame] = {}
    try:
        sheet5_data = load_sheet5_basis_data(CITI_FPATH)
    except Exception as exc:
        print(f"Skipping Sheet5 section: {exc}")

    report_date = date.today().strftime("%Y%m%d")
    pdf_path = OUTPUT_DIR / f"Basis_Report_{report_date}.pdf"
    if pdf_path.exists():
        try:
            with open(pdf_path, "ab"):
                pass
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = OUTPUT_DIR / f"Basis_Report_{report_date}_{ts}.pdf"

    generate_pdf_report(data, pdf_path, sheet5_data=sheet5_data)

    print("\n" + "=" * 70)
    print("Swap Clearing Basis Report")
    print("=" * 70)
    latest_dates = [df.index[-1].date() for df in data.values() if not df.empty]
    if latest_dates:
        print(f"Latest date : {max(latest_dates)}")
    for name, summary in summaries.items():
        if summary.empty:
            continue
        print(f"\n{name} Summary:")
        print(summary.round(2).to_string())

    if sheet5_data:
        print("\nSheet5 Adjacent Tenor Spread Summary:")
        for name, df in sheet5_data.items():
            spreads = compute_adjacent_tenor_spreads(df)
            summary = build_spread_summary(spreads, windows=(90, 250))
            print(f"\n{name}:")
            print(summary.round(2).to_string())

    print(f"\nPDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
