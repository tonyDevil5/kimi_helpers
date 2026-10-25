"""One MBS report: RV, TBA, JPM mean reversion, MBS/credit and 120d fair value.

Run in mercury_env: python mbs_combined_report.py
Use --from-existing to combine saved reports without refreshing the analyses.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from ust_combined_report import build_report

ROOT = Path(__file__).resolve().parent
DEFAULT_BACKTEST_ROOT = ROOT.parent/'BackTestingTool'
DEFAULT_OUTPUT = ROOT/'mbs_combined_output'/'MBS_Combined_Report.html'


def latest_mbs_rv(folder: Path) -> Path:
    candidates = []
    for path in folder.glob('MBS_RV_Report*.html'):
        match = re.search(r'_(\d{8})(?:_\d{6})?$', path.stem)
        if match:
            candidates.append((match.group(1), 'Interactive' in path.stem, path.name, path))
    if not candidates:
        raise FileNotFoundError(f'No MBS RV HTML in {folder}; run without --from-existing first.')
    return max(candidates)[-1]


def report_sources(backtest_root: Path):
    examples = backtest_root/'strategy_example'
    return [
        ('mbs-rv', 'MBS RV', latest_mbs_rv(ROOT/'mbs_rv_output')),
        ('tba-strategies', 'TBA strategies', examples/'tba_strategy_output'/'tba_strategy_analysis.html'),
        ('jpm-mean-reversion', 'JPM performance mean reversion', examples/'jpm_coupon_mean_reversion_output'/'jpm_coupon_mean_reversion.html'),
        ('mbs-credit', 'MBS vs credit', examples/'mbs_credit_output'/'mbs_credit_basis_report_interactive.html'),
        ('fair-value-120', '120d fair value', examples/'fair_value_120_output'/'fair_value_120_backtest.html'),
    ]


def refresh_reports(backtest_root: Path):
    # Keep imports tied to this checkout when the backtesting scripts refer to a
    # differently named monitor checkout on another computer.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(backtest_root), env.get('PYTHONPATH', '')])
    jobs = [
        (ROOT, ['mbs_rv_report.py', '--format', 'interactive']),
        (backtest_root, ['strategy_example/tba_strategy_backtest.py', '--update']),
        (backtest_root, ['strategy_example/jpm_coupon_mean_reversion.py']),
        (backtest_root, ['strategy_example/mbs_credit_basis_backtest.py', '--update']),
        (backtest_root, ['strategy_example/fair_value_120_backtest.py', '--update']),
    ]
    for cwd, args in jobs:
        print(f'Refreshing {args[0]}...', flush=True)
        subprocess.run([sys.executable, '-u', *args], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-existing', action='store_true', help='Combine existing HTML without running the underlying analyses')
    parser.add_argument('--backtest-root', type=Path, default=DEFAULT_BACKTEST_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.from_existing:
        refresh_reports(args.backtest_root)
    from plotly.offline import get_plotlyjs
    return build_report(report_sources(args.backtest_root), args.output,
        title='MBS Research & Strategies', inline_plotly_js=get_plotlyjs())


if __name__ == '__main__':
    main()
