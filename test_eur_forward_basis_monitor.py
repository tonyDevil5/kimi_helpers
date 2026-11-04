"""Quote reproduction and forward-basis unit/sign checks."""
import unittest
import QuantLib as ql
import pandas as pd

import eur_forward_basis_monitor as monitor


class EurForwardBasisTests(unittest.TestCase):
    def setUp(self):
        self.asof = ql.Date(3, 11, 2026)
        # Synthetic upward curves with positive credit/reset premia.
        self.rows = [dict(tenor_years=n, estr_pct=2.8+.015*n,
                          eur3m_pct=3.0+.012*n, eur6m_pct=3.1+.009*n) for n in monitor.TENORS]
        self.curves, self.calibration = monitor.build_curves(self.rows, self.asof)

    def test_independent_spot_swap_repricing(self):
        for row in self.rows:
            rates = monitor.forward_rates(self.curves, self.asof, 0, row['tenor_years'])
            for curve in monitor.TEMPLATES:
                self.assertAlmostEqual(rates[curve+'_pct'], row[curve+'_pct'], places=7)

    def test_grid_units_and_credit_premium_direction(self):
        forwards = [monitor.forward_rates(self.curves, self.asof, s, t)
                    for s in monitor.STARTS for t in monitor.LENGTHS]
        self.assertEqual(len(forwards), 70)
        for f in forwards:
            self.assertAlmostEqual(f['eur3m_minus_estr_bps'], (f['eur3m_pct']-f['estr_pct'])*100)
            self.assertAlmostEqual(f['eur6m_minus_eur3m_bps'], (f['eur6m_pct']-f['eur3m_pct'])*100)
            self.assertAlmostEqual(f['eur6m_minus_estr_bps'], f['eur3m_minus_estr_bps']+f['eur6m_minus_eur3m_bps'])
            self.assertAlmostEqual(f['eur6m_minus_estr_bps'], (f['eur6m_pct']-f['estr_pct'])*100)
        raised = [dict(r, eur6m_pct=r['eur6m_pct']+.1) for r in self.rows]
        curves, _ = monitor.build_curves(raised, self.asof)
        before = monitor.forward_rates(self.curves, self.asof, 5, 10)
        after = monitor.forward_rates(curves, self.asof, 5, 10)
        self.assertGreater(after['eur6m_minus_eur3m_bps'], before['eur6m_minus_eur3m_bps'])
        self.assertAlmostEqual(after['eur3m_minus_estr_bps'], before['eur3m_minus_estr_bps'])

    def test_missing_duplicate_and_nonfinite_quotes_rejected(self):
        for bad in (self.rows[:-1], self.rows+[self.rows[0]],
                    [dict(r, estr_pct=float('nan')) if i==0 else r for i,r in enumerate(self.rows)]):
            with self.assertRaises(ValueError):
                monitor.validate_quotes(bad)

    def test_zscores_use_last_250_sample_std_and_ignore_future(self):
        dates=pd.bdate_range('2025-01-01',periods=301).strftime('%Y-%m-%d').tolist()
        metrics=('eur3m_minus_estr_bps','eur6m_minus_eur3m_bps','eur6m_minus_estr_bps')
        rows=[dict(date=day,**{metric:float(i) for metric in metrics}) for i,day in enumerate(dates)]
        rows[-1][metrics[0]]=1000000.
        stats=monitor.latest_zscores({'1Y1Y':rows},dates[-2])['1Y1Y']
        expected=pd.Series(range(50,300),dtype=float)
        self.assertAlmostEqual(stats[metrics[0]]['z'],(299-expected.mean())/expected.std(ddof=1))
        self.assertEqual(stats[metrics[0]]['obs'],250)
        for series in (rows[:249],[dict(r,**{m:1. for m in metrics}) for r in rows[:250]]):
            result=monitor.latest_zscores({'1Y1Y':series},series[-1]['date'])['1Y1Y']
            self.assertIsNone(result[metrics[0]]['z'])


if __name__ == '__main__':
    unittest.main()
