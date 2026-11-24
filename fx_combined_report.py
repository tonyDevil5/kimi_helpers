"""Build one portable, interactive FX forward report from the five pair monitors.

Pairs: USD/HKD, USD/CNH, USD/JPY, USD/SGD, EUR/USD (quoted EURUSD per market convention).
Sequential refresh (pairs share the Bloomberg session); a pair that fails is
packaged from its existing output with a warning.

Run:
    python fx_combined_report.py
Use --from-existing to package existing HTML without refreshing Bloomberg.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from swap_combined_report import build_report, latest_report

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "fx_combined_output" / "FX_Combined_Report.html"

PAIR_MONITORS = [
    ("usdhkd", "USD/HKD forwards", "fx_usdhkd_monitor.py", "fx_usdhkd_output", "USDHKD_Forward_Monitor_????????.html"),
    ("usdcnh", "USD/CNH forwards", "fx_usdcnh_monitor.py", "fx_usdcnh_output", "USDCNH_Forward_Monitor_????????.html"),
    ("usdjpy", "USD/JPY forwards", "fx_usdjpy_monitor.py", "fx_usdjpy_output", "USDJPY_Forward_Monitor_????????.html"),
    ("usdsgd", "USD/SGD forwards", "fx_usdsgd_monitor.py", "fx_usdsgd_output", "USDSGD_Forward_Monitor_????????.html"),
    ("eurusd", "EUR/USD forwards", "fx_eurusd_monitor.py", "fx_eurusd_output", "EURUSD_Forward_Monitor_????????.html"),
]


def refresh_reports() -> None:
    # Sequential: the scripts share the Bloomberg session.
    for _, title, script, _, _ in PAIR_MONITORS:
        print(f"Refreshing {script}...", flush=True)
        result = subprocess.run([sys.executable, "-u", script], cwd=ROOT)
        if result.returncode != 0:
            print(f"WARNING: {script} failed (exit {result.returncode}); "
                  f"packaging existing {title} output instead.", flush=True)


def report_sources() -> list[tuple[str, str, Path]]:
    sources = []
    for key, title, _, folder, pattern in PAIR_MONITORS:
        try:
            sources.append((key, title, latest_report(ROOT / folder, pattern)))
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {title}: {exc}", flush=True)
    if not sources:
        raise SystemExit("No source reports found. Run without --from-existing first.")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-existing", action="store_true",
                        help="Package existing reports without querying Bloomberg")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.from_existing:
        refresh_reports()
    build_report(report_sources(), args.output, title="FX Forward Monitor")


if __name__ == "__main__":
    main()
