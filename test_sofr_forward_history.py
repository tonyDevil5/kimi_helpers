import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import sofr_forward_curve as core
from sofr_forward_history import calculate


class HistoryTests(unittest.TestCase):
    def test_cache_reuse_revision_and_missing_pillar(self):
        args = SimpleNamespace(basis_sign=1, settlement_days=2, payment_lag=2, rebuild_history=False)
        quotes = pd.DataFrame([dict(date='2026-11-13', tenor_years=n,
            lch_rate_pct=3.3+.025*n, basis_bps=1+.03*n) for n in core.TENORS])
        first, failures, reused = calculate(core, quotes, pd.DataFrame(), args)
        self.assertEqual(len(first), 66)
        self.assertTrue(failures.empty)
        with patch.object(core, 'build_curve', side_effect=AssertionError('Unexpected recalculation')):
            cached, failures, reused = calculate(core, quotes, first, args)
        self.assertEqual(reused, 1)
        self.assertEqual(len(cached), 66)
        revised = quotes.copy()
        revised.loc[0, 'basis_bps'] += .1
        changed, failures, reused = calculate(core, revised, first, args)
        self.assertEqual(reused, 0)
        self.assertNotEqual(changed.input_sha256.iloc[0], first.input_sha256.iloc[0])
        missing, failures, reused = calculate(core, quotes.iloc[1:], first, args)
        self.assertTrue(missing.empty)
        self.assertEqual(len(failures), 1)


if __name__ == '__main__':
    unittest.main()
