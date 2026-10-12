r"""
Fed Funds Rate FOMC Hike/Cut Analysis
=====================================
Ad-hoc analysis that downloads the US effective Fed Funds rate
(Bloomberg ticker: FEDL01 Index) for the past 10 years, identifies
days when the FOMC hiked or cut rates in multiples of 25 bps, and
outputs a PDF report with a decision table and charts.

Run:
    python fed_funds_analysis.py

Output:
    ./fed_funds_output/FedFunds_FOMC_Analysis.pdf
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

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
TICKER = "FEDL01 Index"
OUTPUT_DIR = Path(__file__).parent / "fed_funds_output"

# Lookback window
YEARS_OF_HISTORY = 10

# FOMC decisions move in 25 bp steps; allow a small tolerance for the
# effective rate, which can print slightly off the target.
BP_STEP = 0.25
TOLERANCE = 0.12  # +/- 12 bps
MIN_CHANGE = 0.18  # ignore shifts smaller than 18 bps

# Window used to define the stable rate level before and after a decision.
# A 5-day median smooths day-to-day noise in the effective Fed Funds rate.
MEDIAN_WINDOW = 5

# Official FOMC target changes (in bps) for cross-checking the effective-rate signal.
# Sourced from the Federal Reserve's historical FOMC target rate data.
OFFICIAL_FOMC_DECISIONS = {
    "2016-12-15": 25,
    "2017-03-16": 25,
    "2017-06-15": 25,
    "2017-12-14": 25,
    "2018-03-22": 25,
    "2018-06-14": 25,
    "2018-09-27": 25,
    "2018-12-20": 25,
    "2019-08-01": -25,
    "2019-09-19": -25,
    "2019-10-31": -25,
    "2020-03-04": -50,
    "2020-03-16": -100,
    "2022-03-17": 25,
    "2022-05-05": 50,
    "2022-06-16": 75,
    "2022-07-28": 75,
    "2022-09-22": 75,
    "2022-11-03": 75,
    "2022-12-15": 50,
    "2023-02-02": 25,
    "2023-03-23": 25,
    "2023-05-04": 25,
    "2023-07-27": 25,
    "2024-09-19": -50,
    "2024-11-08": -25,
    "2024-12-19": -25,
    "2025-09-18": -25,
    "2025-10-30": -25,
    "2025-12-11": -25,
}

# UST yield tickers used for the event study
CT10_TICKER = "GT10 Govt"
CT2_TICKER = "GT02 Govt"
NMCMFR30_TICKER = "NMCMFR30 Index"  # 30y fixed mortgage rate
EVENT_WINDOW_DAYS = 30
EVENT_STUDY_SERIES = ["CT2", "CT10", "2s10s", "NMCMFR30"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fed_funds(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
    """Download FEDL01 Index from Bloomberg and return a clean daily series."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (date.today() - timedelta(days=YEARS_OF_HISTORY * 365)).strftime("%Y-%m-%d")

    print(f"Downloading {TICKER} from Bloomberg ({start_date} to {end_date})...")
    df = blp.bdh(tickers=[TICKER], flds=["px_last"], start_date=start_date, end_date=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    series = df[TICKER].copy() if TICKER in df.columns else df.iloc[:, 0].copy()
    series.index = pd.to_datetime(series.index)
    series = series.ffill().dropna().sort_index()
    series.name = "Fed Funds Effective Rate"
    print(f"Loaded {len(series)} rows, {series.index[0].date()} to {series.index[-1].date()}")
    return series


def load_ust_yields(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Download CT2, CT10 and NMCMFR30 from Bloomberg and compute the 2s10s spread."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (date.today() - timedelta(days=YEARS_OF_HISTORY * 365)).strftime("%Y-%m-%d")

    tickers = [CT2_TICKER, CT10_TICKER, NMCMFR30_TICKER]
    print(f"Downloading {CT2_TICKER}, {CT10_TICKER} and {NMCMFR30_TICKER} from Bloomberg ({start_date} to {end_date})...")
    df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=start_date, end_date=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("px_last", axis=1, level=-1)

    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna().sort_index()
    df = df.rename(columns={CT2_TICKER: "CT2", CT10_TICKER: "CT10", NMCMFR30_TICKER: "NMCMFR30"})
    df["2s10s"] = df["CT10"] - df["CT2"]
    print(f"Loaded yields: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ---------------------------------------------------------------------------
# FOMC decision identification
# ---------------------------------------------------------------------------
def _is_multiple_of_step(
    change: float, step: float = BP_STEP, tol: float = TOLERANCE, min_change: float = MIN_CHANGE
) -> bool:
    """Return True if |change| is approximately a positive multiple of step."""
    if abs(change) < min_change:
        return False
    ratio = round(abs(change) / step)
    return ratio > 0 and abs(abs(change) - ratio * step) <= tol


def identify_fomc_changes(series: pd.Series) -> pd.DataFrame:
    """
    Identify FOMC decisions from the effective Fed Funds rate.

    Because FEDL01 is a weighted-average effective rate, it does not always
    move by exactly the FOMC target change on the meeting day. We therefore
    compare a 5-day median before each candidate date with a 5-day median
    starting on that date, and flag shifts that are approximately a multiple
    of 25 bps.
    """
    changes = []
    n = len(series)
    values = series.values
    dates = series.index
    w = MEDIAN_WINDOW

    i = w
    while i < n - w:
        prev_level = float(np.median(values[i - w : i]))
        next_level = float(np.median(values[i : i + w]))
        change = next_level - prev_level

        if not _is_multiple_of_step(change):
            i += 1
            continue

        # The 5-day median shift is detected a couple of days before the actual
        # FOMC announcement. Pinpoint the FOMC date as the day in the window
        # with the largest absolute daily change.
        window_end = min(i + w, n - 1)
        daily_changes = np.diff(values[i - 1 : window_end + 1])
        max_idx_in_window = int(np.argmax(np.abs(daily_changes)))
        fomc_idx = i + max_idx_in_window
        fomc_date = dates[fomc_idx]

        direction = "Hike" if change > 0 else "Cut"
        steps = int(round(abs(change) / BP_STEP))
        changes.append(
            {
                "date": fomc_date,
                "rate_before": prev_level,
                "rate_after": next_level,
                "change": change,
                "change_bps": int(round(change * 100)),
                "action": direction,
                "steps_25bp": steps if direction == "Hike" else -steps,
            }
        )
        # Skip ahead to avoid duplicate detections of the same regime shift
        i += w

    df = pd.DataFrame(changes)
    if not df.empty:
        df = df.set_index("date").sort_index()
    return df


def add_official_decisions(changes: pd.DataFrame) -> pd.DataFrame:
    """Add the official FOMC target change (in bps) for each detected date."""
    if changes.empty:
        changes["official_bps"] = pd.Series(dtype="Int64")
        return changes

    official_map = {
        pd.Timestamp(d): v for d, v in OFFICIAL_FOMC_DECISIONS.items()
    }
    changes = changes.copy()
    changes["official_bps"] = changes.index.map(lambda d: official_map.get(d, pd.NA)).astype("Int64")
    return changes


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def build_summary_table(changes: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """Build a small summary table for the report."""
    latest_rate = float(series.iloc[-1])
    first_rate = float(series.iloc[0])
    total_change_bps = int(round((latest_rate - first_rate) * 100))

    if changes.empty:
        rows = [
            {"metric": "Start date", "value": str(series.index[0].date())},
            {"metric": "End date", "value": str(series.index[-1].date())},
            {"metric": "Start rate", "value": f"{first_rate:.3f}%"},
            {"metric": "Latest rate", "value": f"{latest_rate:.3f}%"},
            {"metric": "Total change", "value": f"{total_change_bps} bps"},
            {"metric": "FOMC decisions detected", "value": "0"},
        ]
        return pd.DataFrame(rows)

    hikes = changes[changes["action"] == "Hike"]
    cuts = changes[changes["action"] == "Cut"]

    official_total = (
        int(changes["official_bps"].sum())
        if "official_bps" in changes.columns and not changes["official_bps"].isna().all()
        else None
    )

    rows = [
        {"metric": "Start date", "value": str(series.index[0].date())},
        {"metric": "End date", "value": str(series.index[-1].date())},
        {"metric": "Start rate", "value": f"{first_rate:.3f}%"},
        {"metric": "Latest rate", "value": f"{latest_rate:.3f}%"},
        {"metric": "Total change (FEDL01)", "value": f"{total_change_bps} bps"},
        {"metric": "Total official FOMC change", "value": f"{official_total} bps" if official_total is not None else "N/A"},
        {"metric": "FOMC decisions detected", "value": str(len(changes))},
        {"metric": "Total hikes", "value": str(len(hikes))},
        {"metric": "Total cuts", "value": str(len(cuts))},
        {"metric": "Largest hike", "value": f"{hikes['change_bps'].max()} bps" if not hikes.empty else "N/A"},
        {"metric": "Largest cut", "value": f"{cuts['change_bps'].min()} bps" if not cuts.empty else "N/A"},
        {"metric": "Total hiking (FEDL01)", "value": f"{int(hikes['change_bps'].sum())} bps" if not hikes.empty else "0 bps"},
        {"metric": "Total easing (FEDL01)", "value": f"{int(cuts['change_bps'].sum())} bps" if not cuts.empty else "0 bps"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FOMC event study: CT10, CT2, 2s10s, NMCMFR30 around official decisions
# ---------------------------------------------------------------------------
def get_official_fomc_series() -> pd.Series:
    """Return official FOMC decisions as a pandas Series indexed by date."""
    s = pd.Series(
        OFFICIAL_FOMC_DECISIONS,
        dtype="Int64",
    )
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _classify_event(bps: int) -> str:
    """Classify an official FOMC decision into a study bucket."""
    if bps > 0:
        return "Hike 25bp" if bps == 25 else "Hike >25bp"
    if bps < 0:
        return "Cut 25bp" if bps == -25 else "Cut >25bp"
    return "Unchanged"


def build_event_study_data(
    yields_df: pd.DataFrame,
    window: int = EVENT_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build a long-form DataFrame of CT10, CT2, 2s10s and NMCMFR30 around each
    official FOMC decision, indexed by (event_date, event_day).
    """
    official = get_official_fomc_series()
    records = []

    for event_date, bps in official.items():
        if event_date not in yields_df.index:
            continue

        bucket = _classify_event(int(bps))
        start = event_date - timedelta(days=window)
        end = event_date + timedelta(days=window)
        window_df = yields_df.loc[start:end].copy()
        if window_df.empty:
            continue

        # Map each calendar date to event day (0 = decision day)
        window_df["event_day"] = (window_df.index - event_date).days
        for col in EVENT_STUDY_SERIES:
            window_df[col] = window_df[col] - window_df.loc[event_date, col]
            # Display yield changes and spread in basis points
            window_df[col] = window_df[col] * 100

        for idx, row in window_df.iterrows():
            rec = {
                "event_date": event_date,
                "event_day": int(row["event_day"]),
                "bucket": bucket,
                "official_bps": int(bps),
            }
            for col in EVENT_STUDY_SERIES:
                rec[col] = float(row[col])
            records.append(rec)

    return pd.DataFrame(records)


def _compute_quantiles(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return min/25/50/75/max quantiles for a single series by event_day."""
    q = sub.groupby("event_day")[col].quantile([0, 0.25, 0.5, 0.75, 1.0]).unstack()
    q.columns = [f"{col}_min", f"{col}_p25", f"{col}_median", f"{col}_p75", f"{col}_max"]
    return q.reset_index()


def aggregate_event_study(event_df: pd.DataFrame) -> pd.DataFrame:
    """Compute median and percentiles (min/25/50/75/max) by event day and bucket."""
    results = []
    for bucket in sorted(event_df["bucket"].unique()):
        sub = event_df[event_df["bucket"] == bucket]
        merged = sub[["event_day"]].drop_duplicates().sort_values("event_day")
        for col in EVENT_STUDY_SERIES:
            q_df = _compute_quantiles(sub, col)
            merged = merged.merge(q_df, on="event_day", how="outer")
        merged["bucket"] = bucket
        results.append(merged)

    return pd.concat(results, ignore_index=True).sort_values(["bucket", "event_day"])


def build_event_summary_table(event_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats: median change from t-30 to t+30 and number of events per bucket."""
    rows = []
    for bucket in sorted(event_df["bucket"].unique()):
        sub = event_df[event_df["bucket"] == bucket]
        n_events = sub["event_date"].nunique()
        t_minus_30 = sub[sub["event_day"] == -EVENT_WINDOW_DAYS][EVENT_STUDY_SERIES].median()
        t_plus_30 = sub[sub["event_day"] == EVENT_WINDOW_DAYS][EVENT_STUDY_SERIES].median()

        for col in EVENT_STUDY_SERIES:
            rows.append({
                "bucket": bucket,
                "n_events": n_events,
                "series": col,
                f"t-{EVENT_WINDOW_DAYS}": round(float(t_minus_30[col]), 2) if not pd.isna(t_minus_30[col]) else None,
                f"t+{EVENT_WINDOW_DAYS}": round(float(t_plus_30[col]), 2) if not pd.isna(t_plus_30[col]) else None,
                "change": round(float(t_plus_30[col] - t_minus_30[col]), 2) if not pd.isna(t_plus_30[col]) and not pd.isna(t_minus_30[col]) else None,
            })

    return pd.DataFrame(rows)


def build_event_percentile_table(event_df: pd.DataFrame, day: int = EVENT_WINDOW_DAYS) -> pd.DataFrame:
    """Build a min/25/50/75/max percentile table for each bucket and series at a given event day."""
    rows = []
    target_col = f"t+{day}" if day >= 0 else f"t{day}"
    for bucket in sorted(event_df["bucket"].unique()):
        sub = event_df[(event_df["bucket"] == bucket) & (event_df["event_day"] == day)]
        if sub.empty:
            continue
        for col in EVENT_STUDY_SERIES:
            quantiles = sub[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
            rows.append({
                "bucket": bucket,
                "series": col,
                "n": int(sub["event_date"].nunique()),
                "min": round(float(quantiles[0.0]), 2),
                "p25": round(float(quantiles[0.25]), 2),
                "median": round(float(quantiles[0.5]), 2),
                "p75": round(float(quantiles[0.75]), 2),
                "max": round(float(quantiles[1.0]), 2),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["bucket", "series"])
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fed_funds_with_decisions(series: pd.Series, changes: pd.DataFrame) -> plt.Figure:
    """Plot the Fed Funds rate with vertical markers for hikes and cuts."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, label="Effective Fed Funds Rate", color="navy", linewidth=1.5)

    if not changes.empty:
        hike_dates = changes[changes["action"] == "Hike"].index
        cut_dates = changes[changes["action"] == "Cut"].index
        ax.scatter(
            hike_dates,
            series.loc[hike_dates],
            color="red",
            marker="^",
            s=80,
            label="FOMC Hike",
            zorder=5,
        )
        ax.scatter(
            cut_dates,
            series.loc[cut_dates],
            color="green",
            marker="v",
            s=80,
            label="FOMC Cut",
            zorder=5,
        )

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=10)
    ax.set_title("Effective Fed Funds Rate and FOMC Decisions (past 10 years)", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_decision_histogram(changes: pd.DataFrame) -> plt.Figure:
    """Histogram of FOMC decision sizes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if changes.empty:
        ax.text(0.5, 0.5, "No FOMC decisions detected", fontsize=12, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    sizes = changes["change_bps"].values
    colors = ["red" if s > 0 else "green" for s in sizes]
    ax.bar(range(len(sizes)), sizes, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in changes.index], rotation=90, fontsize=7)
    ax.set_ylabel("Change (bps)", fontsize=10)
    ax.set_title("FOMC Decision Sizes", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _event_study_palette(bucket: str) -> str:
    """Color palette for event-study buckets."""
    return {
        "Hike 25bp": "#d62728",
        "Hike >25bp": "#ff7f0e",
        "Cut 25bp": "#2ca02c",
        "Cut >25bp": "#1f77b4",
    }.get(bucket, "gray")


def _plot_event_study_series(
    event_df: pd.DataFrame,
    series_name: str,
    title: str,
    buckets: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot median normalized series with 25th-75th percentile bands around FOMC decisions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if buckets is None:
        buckets = sorted(event_df["bucket"].unique())

    for bucket in buckets:
        sub = event_df[event_df["bucket"] == bucket]
        if sub.empty:
            continue
        q = sub.groupby("event_day")[series_name].quantile([0, 0.25, 0.5, 0.75, 1.0]).unstack()
        q.columns = ["min", "p25", "median", "p75", "max"]
        q = q.reset_index().sort_values("event_day")

        color = _event_study_palette(bucket)
        ax.fill_between(q["event_day"], q["p25"], q["p75"], color=color, alpha=0.30, label=f"{bucket} 25-75%")
        ax.plot(
            q["event_day"],
            q["median"],
            label=f"{bucket} median",
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="FOMC decision")
    ax.set_xlabel("Days relative to FOMC decision", fontsize=10)
    ax.set_ylabel(f"Change in {series_name} (bps)", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_event_study_overview(event_df: pd.DataFrame) -> plt.Figure:
    """Overview plot: hikes vs cuts for each event-study series with median and IQR bands."""
    series_titles = {
        "CT10": "CT10 Yield Around FOMC Decisions",
        "CT2": "CT2 Yield Around FOMC Decisions",
        "2s10s": "2s10s Spread Around FOMC Decisions",
        "NMCMFR30": "NMCMFR30 (30y Mortgage Rate) Around FOMC Decisions",
    }
    n_series = len(EVENT_STUDY_SERIES)
    fig, axes = plt.subplots(n_series, 1, figsize=(10, 4 * n_series), sharex=True)

    for ax, series_name in zip(axes, EVENT_STUDY_SERIES):
        title = series_titles.get(series_name, f"{series_name} Around FOMC Decisions")
        # Aggregate hikes vs cuts
        for direction, color in [("Hike", "red"), ("Cut", "green")]:
            sub = event_df[event_df["bucket"].str.startswith(direction)]
            if sub.empty:
                continue
            q = sub.groupby("event_day")[series_name].quantile([0, 0.25, 0.5, 0.75, 1.0]).unstack()
            q.columns = ["min", "p25", "median", "p75", "max"]
            q = q.reset_index().sort_values("event_day")
            ax.fill_between(q["event_day"], q["p25"], q["p75"], color=color, alpha=0.30)
            ax.plot(
                q["event_day"],
                q["median"],
                label=direction,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Change in {series_name} (bps)", fontsize=10)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("Days relative to FOMC decision", fontsize=10)
    fig.tight_layout()
    return fig


def _add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    fontsize: int = 9,
    pagesize: Tuple[float, float] = (11.0, 8.5),
) -> None:
    """Add a DataFrame as a single PDF page."""
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis("off")

    # Title and subtitle at the top of the figure (outside the table area)
    fig.suptitle(title, fontsize=14, weight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, fontsize=10, ha="center", color="gray")

    table_data = [df.columns.tolist()] + df.reset_index(drop=True).values.tolist()
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[1.0 / len(df.columns)] * len(df.columns),
        bbox=[0.05, 0.06, 0.9, 0.78],  # leave top ~22% for title/subtitle
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.8)

    # Header styling
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    series: pd.Series,
    changes: pd.DataFrame,
    yields_df: Optional[pd.DataFrame] = None,
    pdf_path: Optional[Path] = None,
) -> Path:
    """Generate the PDF report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for PDF report generation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        report_date = date.today().strftime("%Y%m%d")
        pdf_path = OUTPUT_DIR / f"FedFunds_FOMC_Analysis_{report_date}.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = OUTPUT_DIR / f"FedFunds_FOMC_Analysis_{report_date}_{ts}.pdf"

    summary = build_summary_table(changes, series)

    with PdfPages(pdf_path) as pdf:
        # 1. Summary table
        _add_table_page(
            pdf,
            summary,
            "Fed Funds FOMC Decision Analysis — Summary",
            subtitle=(
                f"Ticker: {TICKER} (effective Fed Funds rate)  |  "
                f"Latest: {series.index[-1].date()}"
            ),
            fontsize=10,
            pagesize=(8.0, 5.0),
        )

        # 2. Time series with hike/cut markers
        fig = plot_fed_funds_with_decisions(series, changes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3. Decision histogram
        fig = plot_decision_histogram(changes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4. Decision table
        if not changes.empty:
            col_order = [
                "date", "rate_before", "rate_after", "change_bps",
                "official_bps", "action", "steps_25bp",
            ]
            display_df = changes.reset_index()[col_order]
            display_df = display_df.rename(columns={
                "date": "Date",
                "rate_before": "Rate Before (%)",
                "rate_after": "Rate After (%)",
                "change_bps": "FEDL01 Change (bps)",
                "official_bps": "Official FOMC (bps)",
                "action": "Action",
                "steps_25bp": "25bp Steps",
            })
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            display_df["Rate Before (%)"] = display_df["Rate Before (%)"].map(lambda x: f"{x:.3f}")
            display_df["Rate After (%)"] = display_df["Rate After (%)"].map(lambda x: f"{x:.3f}")
            display_df["Official FOMC (bps)"] = display_df["Official FOMC (bps)"].apply(
                lambda x: str(int(x)) if pd.notna(x) else "—"
            )
            _add_table_page(
                pdf,
                display_df,
                "FOMC Decision Table",
                subtitle=(
                    f"Detected from {TICKER} using {MEDIAN_WINDOW}-day median levels; "
                    f"official FOMC target changes shown for comparison"
                ),
                fontsize=8,
                pagesize=(11.0, 8.5),
            )

        # 5. Event study: CT10, CT2, 2s10s, NMCMFR30 around official FOMC decisions
        if yields_df is not None and not yields_df.empty:
            event_df = build_event_study_data(yields_df)
            if not event_df.empty:
                # Overview: hikes vs cuts
                fig = plot_event_study_overview(event_df)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Detailed buckets
                for series_name in EVENT_STUDY_SERIES:
                    fig = _plot_event_study_series(
                        event_df,
                        series_name,
                        f"{series_name} Around FOMC Decisions by Size",
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                # Summary table
                summary_df = build_event_summary_table(event_df)
                if not summary_df.empty:
                    pivot = summary_df.pivot(index="bucket", columns="series", values="change").reset_index()
                    pivot.columns.name = None
                    pivot = pivot.rename(columns={
                        "CT10": "CT10 Δ (bps)",
                        "CT2": "CT2 Δ (bps)",
                        "2s10s": "2s10s Δ (bps)",
                        "NMCMFR30": "NMCMFR30 Δ (bps)",
                    })
                    _add_table_page(
                        pdf,
                        pivot,
                        "Yield Event-Study Summary",
                        subtitle=(
                            f"Median change from t-{EVENT_WINDOW_DAYS} to t+{EVENT_WINDOW_DAYS} around official FOMC decisions; "
                            f"all yield changes shown in basis points"
                        ),
                        fontsize=9,
                        pagesize=(9.0, 5.0),
                    )

                # Percentile breakdown table at t+30
                pct_df = build_event_percentile_table(event_df, day=EVENT_WINDOW_DAYS)
                if not pct_df.empty:
                    display_pct = pct_df.rename(columns={
                        "bucket": "Bucket",
                        "series": "Series",
                        "n": "N",
                        "min": "Min",
                        "p25": "25%",
                        "median": "50% (Median)",
                        "p75": "75%",
                        "max": "Max",
                    })
                    _add_table_page(
                        pdf,
                        display_pct,
                        f"Yield Distribution at t+{EVENT_WINDOW_DAYS}",
                        subtitle=(
                            f"Min / 25th / 50th / 75th / Max percentile of yield changes {EVENT_WINDOW_DAYS} days "
                            f"after official FOMC decisions; basis points"
                        ),
                        fontsize=8,
                        pagesize=(10.0, 7.0),
                    )

    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    series = load_fed_funds()
    changes = identify_fomc_changes(series)
    changes = add_official_decisions(changes)
    yields_df = load_ust_yields()
    pdf_path = generate_pdf_report(series, changes, yields_df=yields_df)

    print("\n" + "=" * 70)
    print("Fed Funds FOMC Hike/Cut Analysis")
    print("=" * 70)
    print(f"Latest date : {series.index[-1].date()}")
    print(f"Observations: {len(series)}")
    print(f"FOMC decisions detected: {len(changes)}")
    if not changes.empty:
        print("\nDetected decisions:")
        print(changes.to_string())
    print(f"\nPDF report saved to: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
