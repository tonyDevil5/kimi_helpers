"""Identify FN basis widening shocks and their subsequent recovery.

By default download standalone FN Bloomberg history from 2022-01-01 onward.
Alternatively pass --input-csv to reproduce results from a saved snapshot.
The first observation will be the first Bloomberg data date on or after the
requested start date; the end date defaults to today.

Example: python fn_basis_shocks.py --threshold 1.5 --horizon 5 --vol-window 60
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).parent / "mbs_rv_output" / "fn_shocks"
FN_TICKER = ".FNCC105 G Index"
DEFAULT_START_DATE = "2022-01-01"


def load_fn_basis_from_bloomberg(start_date=DEFAULT_START_DATE, end_date=None):
    """Download standalone FN current-coupon basis history from Bloomberg."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise ImportError("xbbg is required for Bloomberg data download.") from exc
    end_date = end_date or date.today().isoformat()
    if pd.Timestamp(end_date) < pd.Timestamp(start_date):
        raise ValueError("end_date must be on or after start_date.")
    raw = blp.bdh(
        tickers=[FN_TICKER], flds=["px_last"],
        start_date=start_date, end_date=end_date,
    )
    if raw.empty:
        raise ValueError(f"Bloomberg returned no {FN_TICKER} data for the requested dates.")
    series = pd.to_numeric(raw.iloc[:, 0], errors="coerce") * 100.0
    series.index = pd.to_datetime(series.index)
    series = series.dropna().sort_index()
    series.name = "FNCC"
    if series.empty:
        raise ValueError(f"Bloomberg returned no usable {FN_TICKER} observations.")
    print(
        f"Loaded standalone FN basis: {len(series)} observations, "
        f"{series.index[0].date()} to {series.index[-1].date()}"
    )
    return series.to_frame()


def score_widening(series, horizon=5, vol_window=60):
    """Scale changes by daily volatility known at the START of each change.

    For a signal at t, the volatility window ends at t-horizon, so none of
    the shock's own changes enters its denominator. sqrt(horizon) is a scale
    convention, not a claim of independent or normally distributed changes.
    """
    if horizon < 1 or vol_window < 2:
        raise ValueError("Require horizon >= 1 and vol_window >= 2.")
    series = pd.to_numeric(series, errors="raise").dropna().sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Basis data must have a DatetimeIndex.")
    if series.index.has_duplicates or not np.isfinite(series).all():
        raise ValueError("Basis data must have unique dates and finite values.")
    if len(series) <= horizon + vol_window:
        raise ValueError("Insufficient observations for the volatility warm-up.")
    vol = series.diff().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    scale = vol.shift(horizon) * np.sqrt(horizon)
    change = series.diff(horizon)
    return pd.DataFrame({
        "basis_bps": series,
        "change_bps": change,
        "pre_shock_daily_vol_bps": vol.shift(horizon),
        "horizon_scale_bps": scale,
        "shock_score": change / scale.replace(0, np.nan),
    })


def identify_episodes(scored, horizon=5, threshold=1.5):
    """Group overlapping threshold signals, then measure each cluster separately.

    Signals no more than ``horizon`` observations apart have overlapping shock
    windows and belong to one cluster. A later non-overlapping cluster becomes
    a new event even when an earlier event has not recovered. Consequently,
    recovery periods can overlap; this preserves every distinct signal cluster.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")
    s = scored["basis_bps"]
    values = s.to_numpy()
    signals = np.flatnonzero(scored["shock_score"].to_numpy() >= threshold)
    clusters = []
    for signal in signals:
        if clusters and signal - clusters[-1][-1] <= horizon:
            clusters[-1].append(int(signal))
        else:
            clusters.append([int(signal)])
    starts = [cluster[0] - horizon for cluster in clusters]
    records = []
    for cluster_number, cluster in enumerate(clusters):
        trigger = cluster[0]
        last_signal = cluster[-1]
        start = starts[cluster_number]
        baseline = float(values[start])
        initial_hits = np.flatnonzero(values[trigger + 1:] <= baseline)
        initial_recovery = int(trigger + 1 + initial_hits[0]) if len(initial_hits) else None
        next_start = starts[cluster_number + 1] if cluster_number + 1 < len(starts) else len(s)
        peak_window_end = next_start - 1
        if initial_recovery is not None:
            peak_window_end = min(peak_window_end, initial_recovery)
        peak = trigger + int(np.argmax(values[trigger:peak_window_end + 1]))
        peak_value = float(values[peak])
        recovery_hits = np.flatnonzero(values[peak + 1:] <= baseline)
        recovery = int(peak + 1 + recovery_hits[0]) if len(recovery_hits) else None
        stop = recovery if recovery is not None else len(s) - 1
        halfway_level = (baseline + peak_value) / 2
        half_hits = np.flatnonzero(values[peak + 1:stop + 1] <= halfway_level)
        half = int(peak + 1 + half_hits[0]) if len(half_hits) else None
        records.append({
            "episode": len(records) + 1,
            "start_date": s.index[start],
            "trigger_date": s.index[trigger],
            "last_signal_date": s.index[last_signal],
            "peak_date": s.index[peak],
            "half_recovery_date": s.index[half] if half is not None else pd.NaT,
            "full_recovery_date": s.index[recovery] if recovery is not None else pd.NaT,
            "baseline_bps": baseline,
            "trigger_basis_bps": float(values[trigger]),
            "trigger_change_bps": float(scored["change_bps"].iloc[trigger]),
            "trigger_score": float(scored["shock_score"].iloc[trigger]),
            "pre_shock_daily_vol_bps": float(scored["pre_shock_daily_vol_bps"].iloc[trigger]),
            "peak_bps": peak_value,
            "peak_widening_bps": peak_value - baseline,
            "signal_count": len(cluster),
            "max_signal_score": float(scored["shock_score"].iloc[cluster].max()),
            "last_signal_to_peak_observations": peak - last_signal,
            "widening_observations": peak - start,
            "half_recovery_observations": half - peak if half is not None else np.nan,
            "last_signal_to_half_recovery_observations": half - last_signal if half is not None else np.nan,
            "full_recovery_observations": recovery - peak if recovery is not None else np.nan,
            "full_recovery_calendar_days": (s.index[recovery] - s.index[peak]).days if recovery is not None else np.nan,
            "trigger_to_full_recovery_observations": recovery - trigger if recovery is not None else np.nan,
            "last_signal_to_full_recovery_observations": recovery - last_signal if recovery is not None else np.nan,
            "episode_observations": recovery - start if recovery is not None else np.nan,
            "observations_since_peak": len(s) - 1 - peak,
            "asof_excess_over_baseline_bps": float(values[-1]) - baseline,
            "status": "Recovered" if recovery is not None else "Open",
        })
    return pd.DataFrame(records)


def threshold_sensitivity(scored, horizon, thresholds):
    records = []
    for threshold in sorted(set(thresholds)):
        events = identify_episodes(scored, horizon, threshold)
        closed = events[events["status"] == "Recovered"] if not events.empty else events
        records.append({
            "threshold": threshold,
            "signal_observations": int((scored["shock_score"] >= threshold).sum()),
            "episodes": len(events),
            "recovered": len(closed),
            "open": len(events) - len(closed),
            "median_peak_to_recovery_obs": float(closed["full_recovery_observations"].median()) if not closed.empty else np.nan,
        })
    return pd.DataFrame(records)


def render_report(scored, events, sensitivity, output_dir, horizon, vol_window, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages

    s = scored["basis_bps"]
    first_score = scored["shock_score"].first_valid_index()
    date_fmt = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d") if pd.notna(d) else "Open"
    count_fmt = lambda n: str(int(n)) if pd.notna(n) else "Open"
    colors = ["#b45309", "#2563eb", "#0f766e", "#9333ea", "#be185d"]
    source = (
        "Source: Bloomberg .FNCC105 G Index, PX_LAST x 100; standalone FN history "
        "without an Excel-date join."
    )
    pdf_path = output_dir / "FN_Basis_Shocks.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, (ax, score_ax) = plt.subplots(
            2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("FN basis widening shocks and recovery", fontsize=18, weight="bold", y=.97)
        ax.set_title(
            f"{date_fmt(s.index[0])} to {date_fmt(s.index[-1])}   |   {len(s):,} observations\n"
            f"Shock: {horizon}-observation widening >= {threshold:g} x pre-shock volatility scale",
            fontsize=11, pad=12,
        )
        ax.plot(s.index, s, color="#243447", lw=1.3, label="FN basis")
        ax.axvspan(s.index[0], first_score, color="#d1d5db", alpha=.4, label="Volatility warm-up")
        for row in events.itertuples():
            color = colors[(row.episode - 1) % len(colors)]
            ax.axvspan(
                row.start_date, row.last_signal_date, color=color, alpha=.10,
                label="Shock cluster window" if row.episode == 1 else None,
            )
            ax.scatter(
                [row.trigger_date], [row.trigger_basis_bps], color=color, s=18,
                zorder=5, label="First signal" if row.episode == 1 else None,
            )
            ax.annotate(
                f"E{row.episode}", (row.trigger_date, row.trigger_basis_bps),
                xytext=(0, 7 + 7 * (row.episode % 2)), textcoords="offset points",
                ha="center", color=color, fontsize=7, weight="bold",
            )
        ax.set_ylabel("FN basis (bps)")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.margins(y=.16)
        score_ax.plot(scored.index, scored["shock_score"], color="#4f46e5", lw=1)
        score_ax.axhline(threshold, color="#b91c1c", ls="--", label=f"Widening threshold: {threshold:g}")
        score_ax.axhline(0, color="#6b7280", lw=.7)
        signals = scored["shock_score"] >= threshold
        score_ax.scatter(scored.index[signals], scored.loc[signals, "shock_score"], color="#b91c1c", s=15, zorder=3)
        score_ax.set_ylabel("Widening / volatility scale")
        score_ax.set_xlabel("Date")
        score_ax.legend(loc="upper left", frameon=False, fontsize=9)
        for axis in (ax, score_ax):
            axis.grid(alpha=.2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        fig.text(.07, .025, source, fontsize=8, color="#4b5563")
        fig.tight_layout(rect=[.02, .06, .99, .92])
        pdf.savefig(fig)
        fig.savefig(output_dir / "FN_Basis_Shocks.png", dpi=150)
        plt.close(fig)

        # Page 2: every non-overlapping signal cluster, including overlapping recoveries.
        fig = plt.figure(figsize=(17, 11))
        fig.text(.03, .955, "All FN widening shock clusters and recovery", fontsize=18, weight="bold")
        fig.text(
            .03, .918,
            "Nearby threshold breaches with overlapping 5-observation windows are one cluster. Later clusters remain separate even when prior recovery is incomplete.",
            fontsize=10,
        )
        headers = [
            "Event", "First\nsignal", "Last\nsignal", "Peak\ndate", "Base\nbps",
            "Peak\nbps", "Widening\nbps", "Signal\ndates", "Max\nscore",
            "50% back\nobs", "Peak to\nrecovery\nobs",
            "Last signal to\nrecovery\nobs", "Recovered\non",
        ]
        cells = []
        for row in events.itertuples():
            cells.append([
                f"E{row.episode}", date_fmt(row.trigger_date), date_fmt(row.last_signal_date),
                date_fmt(row.peak_date), f"{row.baseline_bps:.2f}", f"{row.peak_bps:.2f}",
                f"+{row.peak_widening_bps:.2f}", str(int(row.signal_count)),
                f"{row.max_signal_score:.2f}", count_fmt(row.half_recovery_observations),
                count_fmt(row.full_recovery_observations),
                count_fmt(row.last_signal_to_full_recovery_observations),
                date_fmt(row.full_recovery_date),
            ])
        table_ax = fig.add_axes([.025, .10, .95, .78])
        table_ax.axis("off")
        if cells:
            table = table_ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                                   colWidths=[.045, .08, .08, .08, .06, .06, .07, .055, .06, .07, .075, .085, .10])
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1, 1.55)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor("#e5e7eb")
                if r == 0:
                    cell.set_facecolor("#243447")
                    cell.set_text_props(color="white", weight="bold")
                elif r % 2:
                    cell.set_facecolor("#f1f5f9")
        else:
            table_ax.text(.5, .5, "No qualifying widening episodes.", ha="center")
        fig.text(.03, .035, source, fontsize=8, color="#4b5563")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: distributions measured from the last signal date.
        fig, axes = plt.subplots(2, 3, figsize=(17, 11))
        fig.suptitle("FN shock timing distributions", fontsize=18, weight="bold", y=.96)
        metrics = [
            ("last_signal_to_peak_observations", "Last signal to peak", True),
            ("last_signal_to_half_recovery_observations", "Last signal to 50% recovery", False),
            ("last_signal_to_full_recovery_observations", "Last signal to full recovery", False),
            ("half_recovery_observations", "Peak to 50% recovery", False),
            ("full_recovery_observations", "Peak to full recovery", False),
        ]
        for ax, (column, title, include_open) in zip(axes.flat, metrics):
            values = events[column].dropna().astype(float)
            if not include_open:
                values = events.loc[events["status"] == "Recovered", column].dropna().astype(float)
            if len(values):
                ordered = np.sort(values.to_numpy())
                cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
                median = float(values.median())
                q25, q75 = values.quantile([.25, .75])
                ax.step(ordered, cumulative, where="post", color="#2563eb", lw=2)
                ax.scatter(ordered, cumulative, color="#2563eb", s=18, zorder=3)
                ax.scatter(ordered, np.full(len(ordered), .025), marker="|", color="#1e3a8a", s=65)
                ax.axvline(median, color="#b91c1c", lw=1.6, label=f"Median: {median:g}")
                ax.axvspan(q25, q75, color="#f59e0b", alpha=.15, label=f"IQR: {q25:g} to {q75:g}")
                ax.text(.02, .96, f"n={len(values)}\nmin={values.min():g}  max={values.max():g}",
                        transform=ax.transAxes, va="top", fontsize=9)
                if not include_open:
                    ax.set_xscale("symlog", linthresh=10)
            ax.set_title(title, fontsize=12, weight="bold")
            ax.set_xlabel("Trading observations")
            ax.set_ylabel("Cumulative share of events")
            ax.set_ylim(0, 1.04)
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=9, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes.flat[-1].axis("off")
        fig.text(
            .04, .075,
            "Recovery distributions include recovered events only. Last-signal-to-peak includes all events; negative values mean the basis peak preceded the final qualifying signal.",
            fontsize=9,
        )
        fig.text(.04, .04, source, fontsize=8, color="#4b5563")
        fig.tight_layout(rect=[.03, .12, .98, .90])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: complete, explicit description of the revised event rule.
        fig = plt.figure(figsize=(14, 8.5))
        fig.text(.04, .94, "Shock-cluster definitions and threshold sensitivity", fontsize=17, weight="bold")
        notes = [
            f"Signal = (basis[t] - basis[t-{horizon}]) / (std of {vol_window} daily changes ending at t-{horizon} x sqrt({horizon})).",
            f"Use sample standard deviation. No changes inside the shock window enter its volatility estimate. First eligible date: {date_fmt(first_score)}.",
            f"Every score >= {threshold:g} is retained. Signals no more than {horizon} observations apart are one cluster because their shock windows overlap.",
            f"A cluster's baseline is the basis {horizon} observations before its first signal. First and last signal dates appear in the page 2 table.",
            "A later non-overlapping cluster is a new event even if an earlier event has not recovered. Recovery periods can therefore overlap.",
            "Peak is the maximum after the first signal and before the earlier of initial baseline recovery or the next cluster's baseline date.",
            "Half recovery is the first close at or below (baseline + peak) / 2 after the peak. Full recovery is the first close at or below baseline.",
            "Page 2 reports full recovery from both the peak date and the last signal date; both counts use trading observations.",
            "Peak and recovery dates are retrospective. Open events have no completed recovery time; their peak can change with future observations.",
            "These are actual-basis episodes, without adjusting for rates or volatility drivers. The score is descriptive, not a probability or OLS residual.",
            "The initial warm-up is not classified. Daily sampling misses intraday extremes; square-root scaling does not account for serial correlation.",
        ]
        fig.text(.04, .865, "Definitions", fontsize=12, weight="bold")
        for i, note in enumerate(notes):
            fig.text(.04, .825 - i * .043, note, fontsize=9)
        fig.text(.04, .31, "Threshold sensitivity (each cutoff redefines the signal clusters)", fontsize=11, weight="bold")
        for i, row in enumerate(sensitivity.itertuples()):
            med = f"{row.median_peak_to_recovery_obs:g}" if pd.notna(row.median_peak_to_recovery_obs) else "n/a"
            fig.text(.04, .27 - i * .042,
                     f"{row.threshold:g} x scale: {row.signal_observations} signal dates; {row.episodes} episodes; "
                     f"{row.recovered} recovered / {row.open} open; median full recovery {med} observations (completed only).", fontsize=10)
        fig.text(.04, .035, source, fontsize=8, color="#4b5563")
        pdf.savefig(fig)
        plt.close(fig)

        # Event paths are measured relative to their own pre-shock level.
        for offset in range(0, len(events), 3):
            chunk = events.iloc[offset:offset + 3]
            fig, axes = plt.subplots(len(chunk), 1, figsize=(13, 3.2 * len(chunk) + 1), squeeze=False)
            fig.suptitle("Each shock's widening and unwind", fontsize=17, weight="bold", y=.98)
            for ax, row in zip(axes[:, 0], chunk.itertuples()):
                end = row.full_recovery_date if pd.notna(row.full_recovery_date) else s.index[-1]
                path = s.loc[row.start_date:end] - row.baseline_bps
                color = colors[(row.episode - 1) % len(colors)]
                ax.plot(path.index, path, color=color, lw=1.5)
                ax.fill_between(path.index, path.values, 0, color=color, alpha=.1)
                ax.axhline(0, color="#374151", lw=1)
                ax.axhline(row.peak_widening_bps / 2, color="#6b7280", ls="--", lw=.8)
                ax.axvspan(
                    row.trigger_date, row.last_signal_date, color=color, alpha=.08,
                    label="Signal cluster",
                )
                ax.axvline(row.trigger_date, color="#6b7280", ls=":", label="First signal")
                ax.scatter([row.peak_date], [row.peak_widening_bps], color=color, zorder=4, label="Peak")
                if pd.notna(row.half_recovery_date):
                    ax.scatter([row.half_recovery_date], [float(s.loc[row.half_recovery_date]) - row.baseline_bps], marker="s", color="#6b7280", zorder=4, label="Half recovered")
                if pd.notna(row.full_recovery_date):
                    ax.scatter([row.full_recovery_date], [float(s.loc[row.full_recovery_date]) - row.baseline_bps], marker="D", color="#166534", zorder=4, label="Fully recovered")
                recovery_text = (
                    f"Full recovery: {int(row.full_recovery_observations)} observations after peak"
                    if pd.notna(row.full_recovery_observations)
                    else f"Recovery open; {int(row.observations_since_peak)} observations since peak"
                )
                ax.set_title(f"E{row.episode}   {date_fmt(row.start_date)} to {date_fmt(end)}   |   "
                             f"Peak +{row.peak_widening_bps:.2f} bps   |   "
                             f"{recovery_text}", loc="left", fontsize=10)
                ax.set_ylabel("Above baseline (bps)")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False)
                ax.margins(y=.3, x=.03)
                date_format = "%b %Y" if (pd.Timestamp(end) - row.start_date).days > 370 else "%b %d"
                ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.tight_layout(rect=[.02, .02, .98, .93])
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, help="Optional saved dataset; otherwise download standalone FN Bloomberg history.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Bloomberg request start date (default: 2022-01-01).")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="Bloomberg request end date (default: today).")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.input_csv:
        basis = pd.read_csv(args.input_csv, index_col=0, parse_dates=True)
    else:
        basis = load_fn_basis_from_bloomberg(args.start_date, args.end_date)
    scored = score_widening(basis["FNCC"], args.horizon, args.vol_window)
    events = identify_episodes(scored, args.horizon, args.threshold)
    sensitivity = threshold_sensitivity(scored, args.horizon, [1.5, 2, 2.5, 3, args.threshold])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "FN_Basis_Shock_Scores.csv", index_label="date")
    events.to_csv(args.output_dir / "FN_Basis_Shock_Episodes.csv", index=False, date_format="%Y-%m-%d")
    path = render_report(scored, events, sensitivity, args.output_dir, args.horizon, args.vol_window, args.threshold)
    print(events.to_string(index=False))
    print("\nTHRESHOLD SENSITIVITY\n", sensitivity.to_string(index=False))
    print("\nPDF:", path)


if __name__ == "__main__":
    main()
