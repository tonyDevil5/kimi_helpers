r"""
Month-End / Quarter-End Equity vs Bond Performance Report
===========================================================

Analyzes how UST 10Y yields behave around month-end and quarter-end,
conditional on S&P 500 (ES1 Index) performance.

Data sources (Bloomberg Python add-in via xbbg):
    - ES1 Index  : S&P 500 e-mini futures (proxy for SPX)
    - GT10 Govt  : 10Y UST yield

Output:
    ./month_end_equity_bond_output/MonthEnd_Equity_Bond_Report.pdf

Run:
    python month_end_equity_bond_report.py
"""

from __future__ import annotations

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

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "month_end_equity_bond_output"

EQUITY_TICKER = "ES1 Index"
BOND_TICKER = "GT10 Govt"
GERMAN_BOND_TICKER = "GTDEM10Y Govt"

START_DATE = "2018-01-01"
WINDOW_DAYS = 10  # +/- trading days around month/quarter end

# SPX performance window used to classify "strong" vs "weak" equity
SPX_PERF_WINDOW = 20  # 20-day return from day -20 to day 0 (month/quarter end)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data_from_bloomberg(
    tickers: List[str],
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Pull px_last from Bloomberg for the given tickers."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    print(f"Downloading {len(tickers)} tickers from Bloomberg...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)

    # xbbg >= 1.x returns a narwhals DataFrame in long format
    # (ticker / date / field / value); older versions return a pandas
    # DataFrame with MultiIndex columns. Normalize both to a wide frame.
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        df = df.pivot_table(
            index="date", columns="ticker", values="value", aggfunc="last"
        )
        df = df.reindex(columns=[t for t in tickers if t in df.columns])

    # Flatten multi-index columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.ffill().dropna()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ---------------------------------------------------------------------------
# Month-end / quarter-end helpers
# ---------------------------------------------------------------------------
def get_period_ends(df_index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return the last available observation date in each period."""
    ser = pd.Series(index=df_index, dtype=float)
    # Use pandas resample to find period-end labels, then map back to actual dates
    period_labels = ser.resample(freq).last().index

    ends = []
    for label in period_labels:
        # last date in df_index that falls in this period
        mask = (df_index.year == label.year) & (df_index.month == label.month)
        if freq == "QE":
            mask = mask & (df_index.quarter == label.quarter)
        candidates = df_index[mask]
        if len(candidates) > 0:
            ends.append(candidates[-1])
    return pd.DatetimeIndex(ends)


def build_event_windows(
    df: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """
    For each event date, build a window of +/- window_days trading days.
    Returns a long DataFrame with columns:
        event_date, days_from_event, equity_px, bond_yield, german_yield
    """
    all_dates = df.index
    records = []

    for event_date in event_dates:
        if event_date not in all_dates:
            continue
        event_pos = all_dates.get_loc(event_date)
        start_pos = max(0, event_pos - window_days)
        end_pos = min(len(all_dates) - 1, event_pos + window_days)

        window_dates = all_dates[start_pos : end_pos + 1]
        for i, d in enumerate(window_dates):
            days_from = i - (event_pos - start_pos)
            records.append({
                "event_date": event_date.date(),
                "days_from_event": days_from,
                "equity_px": df.loc[d, EQUITY_TICKER],
                "bond_yield": df.loc[d, BOND_TICKER],
                "german_yield": df.loc[d, GERMAN_BOND_TICKER],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Performance calculations
# ---------------------------------------------------------------------------
def compute_event_metrics(
    windows: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For each event, compute:
        - spx_perf: ES1 return from day -SPX_PERF_WINDOW to day 0 (%)
        - yield_chg_bps: GT10 yield change vs day -WINDOW_DAYS, in bps
        - spread_chg_bps: (GT10 - GTDEM10Y) spread change vs day -WINDOW_DAYS, in bps
          (so the plot reads left-to-right: how yields/spreads evolved into the event)

    full_df is used to look up the SPX price SPX_PERF_WINDOW trading days before the event,
    which is outside the +/-WINDOW_DAYS plot window.
    """
    results = []
    for event_date, group in windows.groupby("event_date"):
        group = group.sort_values("days_from_event").reset_index(drop=True)

        # SPX return from day -SPX_PERF_WINDOW to day 0
        t0_px = group.loc[group["days_from_event"] == 0, "equity_px"]
        if t0_px.empty:
            continue
        t0_px = float(t0_px.iloc[0])

        # Look up the SPX price SPX_PERF_WINDOW trading days before the event
        spx_perf = np.nan
        if full_df is not None:
            event_dt = pd.Timestamp(event_date)
            if event_dt in full_df.index:
                event_pos = full_df.index.get_loc(event_dt)
                start_pos = event_pos - SPX_PERF_WINDOW
                if start_pos >= 0:
                    t_minus_px = float(full_df.iloc[start_pos][EQUITY_TICKER])
                    spx_perf = (t0_px / t_minus_px - 1) * 100

        # Yield change vs day -WINDOW_DAYS yield, in bps
        start_yield = float(group.loc[group["days_from_event"] == -WINDOW_DAYS, "bond_yield"].iloc[0])
        group["yield_chg_bps"] = (group["bond_yield"] - start_yield) * 100

        # UST-DE 10Y spread change vs day -WINDOW_DAYS, in bps
        start_spread = float(
            (group.loc[group["days_from_event"] == -WINDOW_DAYS, "bond_yield"]
             - group.loc[group["days_from_event"] == -WINDOW_DAYS, "german_yield"]).iloc[0]
        ) * 100
        group["spread_bps"] = (group["bond_yield"] - group["german_yield"]) * 100
        group["spread_chg_bps"] = group["spread_bps"] - start_spread
        group["spx_perf"] = spx_perf
        results.append(group)

    return pd.concat(results, ignore_index=True)


def classify_spx_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a label column based on SPX performance into the event."""
    df = df.copy()
    df["spx_label"] = np.where(
        df["spx_perf"] >= 0,
        "SPX Up",
        "SPX Down",
    )
    return df


# ---------------------------------------------------------------------------
# Aggregation / quantiles
# ---------------------------------------------------------------------------
def aggregate_by_label(
    df: pd.DataFrame,
    value_col: str = "yield_chg_bps",
) -> Dict[str, pd.DataFrame]:
    """
    For each spx_label, compute median, q25, q75 by days_from_event.
    Also returns an 'All' aggregate.
    """
    aggregations = {}
    labels = ["All"] + sorted(df["spx_label"].dropna().unique().tolist())
    for label in labels:
        if label == "All":
            sub = df
        else:
            sub = df[df["spx_label"] == label]

        agg = (
            sub.groupby("days_from_event")[value_col]
            .agg(
                median=lambda x: x.median(),
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                mean=lambda x: x.mean(),
                n="count",
            )
            .reset_index()
        )
        aggregations[label] = agg
    return aggregations


def aggregate_by_calendar_month(
    df: pd.DataFrame,
    value_col: str = "yield_chg_bps",
) -> Dict[str, pd.DataFrame]:
    """Aggregate month-end windows by calendar month (Jan-Dec)."""
    df = df.copy()
    df["calendar_month"] = pd.to_datetime(df["event_date"]).dt.month
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    aggregations = {}
    for month_num in sorted(df["calendar_month"].unique()):
        sub = df[df["calendar_month"] == month_num]
        agg = (
            sub.groupby("days_from_event")[value_col]
            .agg(
                median=lambda x: x.median(),
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                mean=lambda x: x.mean(),
                n="count",
            )
            .reset_index()
        )
        aggregations[month_names[month_num]] = agg
    return aggregations


def aggregate_by_calendar_quarter(
    df: pd.DataFrame,
    value_col: str = "yield_chg_bps",
) -> Dict[str, pd.DataFrame]:
    """Aggregate quarter-end windows by calendar quarter (Q1-Q4)."""
    df = df.copy()
    df["calendar_quarter"] = pd.to_datetime(df["event_date"]).dt.quarter

    aggregations = {}
    for q in sorted(df["calendar_quarter"].unique()):
        sub = df[df["calendar_quarter"] == q]
        agg = (
            sub.groupby("days_from_event")[value_col]
            .agg(
                median=lambda x: x.median(),
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                mean=lambda x: x.mean(),
                n="count",
            )
            .reset_index()
        )
        aggregations[f"Q{q}"] = agg
    return aggregations


def get_latest_event_path(
    df: pd.DataFrame,
    value_col: str = "yield_chg_bps",
) -> pd.DataFrame:
    """Return the path for the single most recent event (used for dotted overlay)."""
    latest_event = df["event_date"].max()
    sub = df[df["event_date"] == latest_event][["days_from_event", value_col]].copy()
    sub = sub.sort_values("days_from_event").reset_index(drop=True)
    sub = sub.rename(columns={value_col: "median"})
    return sub


def get_prev_event_path(
    df: pd.DataFrame,
    value_col: str = "yield_chg_bps",
) -> pd.DataFrame:
    """Return the path for the most recent completed month/quarter-end (t-1 overlay)."""
    unique_events = sorted(df["event_date"].unique())
    if len(unique_events) < 2:
        return pd.DataFrame(columns=["days_from_event", "median"])

    today = date.today()
    # Walk backwards to find the most recent event that is the last day of its month.
    for ev in reversed(unique_events):
        ev_date = pd.Timestamp(ev)
        last_day = (ev_date + pd.offsets.MonthEnd(0)).date()
        if ev_date.date() == last_day and ev_date.date() <= pd.Timestamp(today).date():
            prev_event = ev
            break
    else:
        prev_event = unique_events[-2]

    sub = df[df["event_date"] == prev_event][["days_from_event", value_col]].copy()
    sub = sub.sort_values("days_from_event").reset_index(drop=True)
    sub = sub.rename(columns={value_col: "median"})
    return sub


def build_spread_month_summary_table(
    df: pd.DataFrame,
    value_col: str = "spread_chg_bps",
    key_days: List[int] = (-5, -1, 0, 1, 5),
) -> pd.DataFrame:
    """
    Build a calendar-month summary table for a given spread/value.

    Rows: Jan-Dec.  Columns: median value at each key day (vs day -10).
    Also includes n_obs (number of events per month) and d0 median.
    """
    df = df.copy()
    df["calendar_month"] = pd.to_datetime(df["event_date"]).dt.month
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    rows = []
    for month_num in sorted(month_names.keys()):
        sub = df[df["calendar_month"] == month_num]
        if sub.empty:
            continue

        n_events = sub["event_date"].nunique()
        row = {"month": month_names[month_num], "n_obs": n_events}

        for d in key_days:
            day_data = sub[sub["days_from_event"] == d][value_col]
            row[f"d{d}"] = day_data.median()

        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.set_index("month")
    return summary


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def _add_title_page(
    pdf: PdfPages,
    title: str,
    subtitle: str,
    n_obs: int,
    n_month_ends: int,
    n_quarter_ends: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.78, title, fontsize=22, weight="bold", ha="center", transform=ax.transAxes)
    # Wrap long subtitle across multiple lines to avoid overlap
    ax.text(
        0.5, 0.70, subtitle,
        fontsize=10, ha="center", color="gray", transform=ax.transAxes,
        verticalalignment="top",
    )

    summary_text = (
        "Data Summary\n"
        f"  Total trading days:     {n_obs:,}\n"
        f"  Month-ends analyzed:    {n_month_ends}\n"
        f"  Quarter-ends analyzed:  {n_quarter_ends}\n"
        f"  Event window:           +/-{WINDOW_DAYS} trading days\n"
        f"  SPX classification:     {SPX_PERF_WINDOW}-day return into event"
    )
    ax.text(
        0.5, 0.55, summary_text,
        fontsize=12, family="monospace", ha="center", transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#40466e"),
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_event_analysis(
    aggregations: Dict[str, pd.DataFrame],
    title: str,
    event_name: str,
    ylabel: str = "GT10 yield change vs day -10 (bps)",
    latest_path: Optional[pd.DataFrame] = None,
    prev_path: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Plot median yield/spread change around month/quarter end, conditional on SPX."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"All": "black", "SPX Up": "green", "SPX Down": "red"}

    for label in ["All", "SPX Up", "SPX Down"]:
        if label not in aggregations:
            continue
        agg = aggregations[label]
        ax.plot(
            agg["days_from_event"],
            agg["median"],
            label=f"{label} (median)",
            color=colors.get(label, "blue"),
            linewidth=2,
        )
        ax.fill_between(
            agg["days_from_event"],
            agg["q25"],
            agg["q75"],
            alpha=0.15,
            color=colors.get(label, "blue"),
        )

    # Overlay the latest single event as a dotted black line
    if latest_path is not None and not latest_path.empty:
        latest_label = "Latest quarter" if "quarter" in event_name.lower() else "Latest month"
        ax.plot(
            latest_path["days_from_event"],
            latest_path["median"],
            label=latest_label,
            color="black",
            linewidth=2,
            linestyle=":",
        )

    # Overlay the most recent completed month/quarter-end as a dashed grey line
    if prev_path is not None and not prev_path.empty:
        prev_label = "Prev quarter-end" if "quarter" in event_name.lower() else "Prev month-end"
        ax.plot(
            prev_path["days_from_event"],
            prev_path["median"],
            label=prev_label,
            color="gray",
            linewidth=2,
            linestyle="--",
        )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel(f"Trading days from {event_name}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_event_breakdown(
    aggregations: Dict[str, pd.DataFrame],
    title: str,
    event_name: str,
    colormap: str = "tab10",
    latest_path: Optional[pd.DataFrame] = None,
    prev_path: Optional[pd.DataFrame] = None,
    ylabel: str = "GT10 yield change vs day -10 (bps)",
) -> plt.Figure:
    """Plot median yield/spread change broken down by calendar month or quarter."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap(colormap)
    labels = list(aggregations.keys())

    for i, label in enumerate(labels):
        agg = aggregations[label]
        color = cmap(i % cmap.N)
        ax.plot(
            agg["days_from_event"],
            agg["median"],
            label=label,
            color=color,
            linewidth=1.5,
        )

    # Overlay the latest single event as a dotted black line
    if latest_path is not None and not latest_path.empty:
        latest_label = "Latest quarter" if "quarter" in event_name.lower() else "Latest month"
        ax.plot(
            latest_path["days_from_event"],
            latest_path["median"],
            label=latest_label,
            color="black",
            linewidth=2.5,
            linestyle=":",
        )

    # Overlay the most recent completed month/quarter-end as a dashed grey line
    if prev_path is not None and not prev_path.empty:
        prev_label = "Prev quarter-end" if "quarter" in event_name.lower() else "Prev month-end"
        ax.plot(
            prev_path["days_from_event"],
            prev_path["median"],
            label=prev_label,
            color="gray",
            linewidth=2,
            linestyle="--",
        )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel(f"Trading days from {event_name}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2 if len(labels) > 4 else 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_summary_table(df: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """Build a summary table of yield changes at key days, conditional on SPX."""
    key_days = [-5, -1, 0, 1, 5]
    rows = []

    for label in ["All", "SPX Up", "SPX Down"]:
        if label == "All":
            sub = df
        else:
            sub = df[df["spx_label"] == label]

        if sub.empty:
            continue

        spx_ret_mean = sub.groupby("event_date")["spx_perf"].first().mean()
        spx_ret_median = sub.groupby("event_date")["spx_perf"].first().median()
        n_events = sub["event_date"].nunique()

        row = {
            "event": event_name,
            "spx_label": label,
            "n": n_events,
            "spx_avg": spx_ret_mean,
            "spx_med": spx_ret_median,
        }
        for d in key_days:
            day_data = sub[sub["days_from_event"] == d]["yield_chg_bps"]
            row[f"d{d}_avg"] = day_data.mean()
            row[f"d{d}_med"] = day_data.median()
        rows.append(row)

    return pd.DataFrame(rows)


def _df_to_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: str = "",
    pagesize: Tuple[float, float] = (17.0, 8.5),
    fontsize: int = 7,
) -> plt.Figure:
    """Render a DataFrame as a matplotlib table figure."""
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold", y=0.98, ha="left", x=0.04)
    if subtitle:
        ax.text(0.04, 0.93, subtitle, fontsize=8, ha="left", transform=fig.transFigure, color="#555555")

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    n_rows, n_cols = display_df.shape
    table_data = [display_df.columns.tolist()] + display_df.values.tolist()

    # Landscape layout: fixed name column, remaining columns share the rest
    name_width = 0.12 if n_cols >= 10 else 0.18
    col_widths = [name_width] + [(1.0 - name_width) / (n_cols - 1)] * (n_cols - 1)

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Header style with rotated labels to avoid overlap
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white", rotation=15, ha="center", va="bottom")
        cell.set_height(0.12)

    # Row style
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")
            cell.set_height(0.05)

    return fig


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def generate_pdf_report(df: pd.DataFrame, pdf_path: Optional[Path] = None) -> Path:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"MonthEnd_Equity_Bond_Report_{report_date}.pdf"
        # If the dated file is locked (e.g., open in a PDF reader), use a timestamped name
        if pdf_path.exists():
            try:
                with open(pdf_path, "wb"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"MonthEnd_Equity_Bond_Report_{report_date}_{ts}.pdf"

    # Build month-end and quarter-end windows
    month_ends = get_period_ends(df.index, "ME")
    quarter_ends = get_period_ends(df.index, "QE")

    month_windows = build_event_windows(df, month_ends, window_days=WINDOW_DAYS)
    quarter_windows = build_event_windows(df, quarter_ends, window_days=WINDOW_DAYS)

    month_df = classify_spx_performance(compute_event_metrics(month_windows, full_df=df))
    quarter_df = classify_spx_performance(compute_event_metrics(quarter_windows, full_df=df))

    month_agg = aggregate_by_label(month_df, value_col="yield_chg_bps")
    quarter_agg = aggregate_by_label(quarter_df, value_col="yield_chg_bps")
    month_spread_agg = aggregate_by_label(month_df, value_col="spread_chg_bps")

    # Latest single-event paths (dotted overlay) — only uses data actually available
    latest_month_path = get_latest_event_path(month_df, value_col="yield_chg_bps")
    latest_quarter_path = get_latest_event_path(quarter_df, value_col="yield_chg_bps")

    # Most recent completed month/quarter-end paths (dashed overlay)
    prev_month_path = get_prev_event_path(month_df, value_col="yield_chg_bps")
    prev_quarter_path = get_prev_event_path(quarter_df, value_col="yield_chg_bps")
    prev_month_spread_path = get_prev_event_path(month_df, value_col="spread_chg_bps")

    month_summary = build_summary_table(month_df, "Month-End")
    quarter_summary = build_summary_table(quarter_df, "Quarter-End")

    data_range = f"{df.index[0].date()} to {df.index[-1].date()}"

    with PdfPages(pdf_path) as pdf:
        _add_title_page(
            pdf,
            "Equity vs Bond Performance Around Month/Quarter End",
            f"ES1 Index vs GT10 Govt vs GTDEM10Y Govt | {data_range}",
            n_obs=len(df),
            n_month_ends=len(month_ends),
            n_quarter_ends=len(quarter_ends),
        )

        # Month-end plot
        fig = plot_event_analysis(
            month_agg,
            title="GT10 Yield Change Around Month-End (conditional on SPX performance)",
            event_name="month-end",
            latest_path=latest_month_path,
            prev_path=prev_month_path,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Month-end UST-DE 10Y spread plot (unconditional)
        fig = plot_event_analysis(
            {"All": month_spread_agg["All"]},
            title="UST - Germany 10Y Spread Change Around Month-End",
            event_name="month-end",
            ylabel="UST - DE 10Y spread change vs day -10 (bps)",
            latest_path=get_latest_event_path(month_df, value_col="spread_chg_bps"),
            prev_path=prev_month_spread_path,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Quarter-end plot
        fig = plot_event_analysis(
            quarter_agg,
            title="GT10 Yield Change Around Quarter-End (conditional on SPX performance)",
            event_name="quarter-end",
            latest_path=latest_quarter_path,
            prev_path=prev_quarter_path,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Summary tables
        fig = _df_to_figure(
            month_summary,
            "Month-End Summary",
            subtitle=f"Data range: {data_range} | Yield change in bps vs day -10; SPX 20-day return (day -20 to month-end)",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = _df_to_figure(
            quarter_summary,
            "Quarter-End Summary",
            subtitle=f"Data range: {data_range} | Yield change in bps vs day -10; SPX 20-day return (day -20 to quarter-end)",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Appendix: breakdown by calendar month / quarter
        month_by_month = aggregate_by_calendar_month(month_df)
        quarter_by_quarter = aggregate_by_calendar_quarter(quarter_df)
        month_spread_by_month = aggregate_by_calendar_month(month_df, value_col="spread_chg_bps")

        fig = plot_event_breakdown(
            month_by_month,
            title="Appendix: Month-End Yield Change by Calendar Month",
            event_name="month-end",
            colormap="tab20",
            latest_path=latest_month_path,
            prev_path=prev_month_path,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_event_breakdown(
            month_spread_by_month,
            title="Appendix: UST - Germany 10Y Spread Change by Calendar Month",
            event_name="month-end",
            colormap="tab20",
            latest_path=get_latest_event_path(month_df, value_col="spread_chg_bps"),
            prev_path=prev_month_spread_path,
            ylabel="UST - DE 10Y spread change vs day -10 (bps)",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Calendar-month summary table for UST-DE 10Y spread
        spread_month_summary = build_spread_month_summary_table(
            month_df, value_col="spread_chg_bps"
        ).reset_index()
        if not spread_month_summary.empty:
            fig = _df_to_figure(
                spread_month_summary,
                "UST - Germany 10Y Spread Change by Calendar Month",
                subtitle="Median spread change (bps vs day -10) around month-end; n_obs = number of month-ends",
                pagesize=(11.0, 8.5),
                fontsize=10,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig = plot_event_breakdown(
            quarter_by_quarter,
            title="Appendix: Quarter-End Yield Change by Calendar Quarter",
            event_name="quarter-end",
            colormap="tab10",
            latest_path=latest_quarter_path,
            prev_path=prev_quarter_path,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_path


def main() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required.")

    df = load_data_from_bloomberg([EQUITY_TICKER, BOND_TICKER, GERMAN_BOND_TICKER])
    pdf_path = generate_pdf_report(df)

    print("\n" + "=" * 70)
    print("Month-End / Quarter-End Equity vs Bond Report")
    print("=" * 70)
    print(f"Latest date : {df.index[-1].date()}")
    print(f"Observations: {len(df)}")
    print(f"PDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
