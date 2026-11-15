import unittest
import numpy as np
import pandas as pd
from sofr_forward_monitor import basis_zscores


class MonitorTests(unittest.TestCase):
    def test_trailing_window_includes_latest_and_excludes_future(self):
        days = pd.bdate_range('2025-01-01', periods=252).strftime('%Y-%m-%d')
        history = [dict(date=d, forward='3y1y', cme_minus_lch_bps=float(i)) for i,d in enumerate(days)]
        value = basis_zscores([dict(forward='3y1y',cme_minus_lch_bps=250.)],history,days[250])['3y1y']
        expected = (250-np.mean(np.arange(1,251)))/np.std(np.arange(1,251),ddof=1)
        self.assertAlmostEqual(value['z'], expected)
        self.assertEqual(value['start'], days[1])
        self.assertEqual(value['end'], days[250])

    def test_insufficient_or_constant_history_is_unavailable(self):
        dates = pd.bdate_range('2025-01-01',periods=250).strftime('%Y-%m-%d')
        history = [dict(date=d,forward='3y1y',cme_minus_lch_bps=1.) for d in dates]
        forward = [dict(forward='3y1y',cme_minus_lch_bps=1.)]
        self.assertIsNone(basis_zscores(forward,history,dates[-1])['3y1y']['z'])
        self.assertIsNone(basis_zscores(forward,history[:-1],dates[-1])['3y1y']['z'])
