"""SOFR fly arithmetic, par repricing, input validation and incremental reuse."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import sofr_butterfly_monitor as monitor


class SofrButterflyTests(unittest.TestCase):
    def inputs(self, day='2026-11-05'):
        return pd.DataFrame([dict(date=day,tenor_years=n,rate_pct=3+.01*n,ticker=f'USOSFR{n} Curncy')
                             for n in monitor.PILLARS])

    def test_adjacent_triples_spot_repricing_and_fly_sign_units(self):
        digest,rates,flies,checks=monitor.calculate_day(self.inputs())
        self.assertEqual(len(monitor.FLIES),12)
        self.assertEqual(monitor.FLIES[-1],(15,20,30))
        self.assertEqual(len(rates),154)
        self.assertEqual(len(flies),132)
        self.assertLess(max(abs(r['error_bps']) for r in checks),1e-5)
        for r in rates:
            if r['start_years']==0:
                self.assertAlmostEqual(r['rate_pct'],3+.01*r['tenor_years'],places=7)
        for f in flies:
            self.assertAlmostEqual(f['fly_bps'],100*(2*f['belly_rate_pct']-f['short_rate_pct']-f['long_rate_pct']))
        spot={f['structure']:f['fly_bps'] for f in flies if f['start_years']==0}
        self.assertAlmostEqual(spot['1/2/3'],0.,places=6)
        self.assertAlmostEqual(spot['10/12/15'],-1.,places=6)
        self.assertAlmostEqual(spot['15/20/30'],-5.,places=6)

    def test_offline_cache_reuses_and_recalculates_revised_date_only(self):
        with tempfile.TemporaryDirectory() as name, patch.object(monitor,'OUTPUT',Path(name)):
            folder=Path(name)/'history';folder.mkdir()
            quotes=pd.concat([self.inputs('2026-11-04'),self.inputs()],ignore_index=True)
            monitor.atomic_csv(quotes,folder/'quotes.csv')
            _,frames,audit=monitor.update('2026-11-05',offline=True)
            self.assertEqual(audit['calculated_dates'],2)
            with patch.object(monitor,'calculate_day',wraps=monitor.calculate_day) as calculate:
                _,_,audit=monitor.update('2026-11-05',offline=True)
                self.assertEqual(audit['reused_dates'],2)
                self.assertEqual(calculate.call_count,0)
                quotes.loc[(quotes.date=='2026-11-05')&(quotes.tenor_years==5),'rate_pct']+=.01
                monitor.atomic_csv(quotes,folder/'quotes.csv')
                _,frames,audit=monitor.update('2026-11-05',offline=True)
                self.assertEqual(audit['calculated_dates'],1)
                self.assertEqual(audit['reused_dates'],1)
                self.assertEqual(calculate.call_count,1)
                self.assertEqual(len(frames['butterflies']),264)

    def test_incomplete_inputs_rejected(self):
        with self.assertRaises(ValueError):
            monitor.quote_rows(self.inputs().iloc[:-1])


if __name__=='__main__':
    unittest.main()
