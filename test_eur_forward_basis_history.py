"""Incremental history checks: reuse, revisions and upsert preservation."""
import unittest
from unittest.mock import patch

import pandas as pd

import eur_forward_basis_monitor as core
import eur_forward_basis_history as history


class HistoryTests(unittest.TestCase):
    def test_incremental_reuses_cache_and_recalculates_only_revision(self):
        quotes=pd.DataFrame([dict(date=day,tenor_years=n,estr_pct=2.8+.01*n,
                                  eur3m_pct=3+.01*n,eur6m_pct=3.1+.01*n)
                             for day in ('2026-11-02','2026-11-03') for n in core.TENORS])
        forwards,calibration,audit=history.compute_missing(core,quotes,pd.DataFrame(),pd.DataFrame())
        self.assertEqual(audit['calculated_dates'],2)
        with patch.object(core,'build_curves',wraps=core.build_curves) as build:
            f,c,a=history.compute_missing(core,quotes,forwards,calibration)
            self.assertEqual(a['reused_dates'],2)
            self.assertEqual(build.call_count,0)
            revised=quotes.copy()
            revised.loc[(revised.date=='2026-11-03')&(revised.tenor_years==5),'eur6m_pct']+=.01
            f,c,a=history.compute_missing(core,revised,f,c)
            self.assertEqual(a['calculated_dates'],1)
            self.assertEqual(a['reused_dates'],1)
            self.assertEqual(build.call_count,1)
            self.assertEqual(len(f),140)

    def test_nonmissing_upsert_retains_old_history_and_no_duplicates(self):
        old=pd.DataFrame([dict(date='2026-11-02',tenor_years=1,estr_pct=3.),
                          dict(date='2026-11-03',tenor_years=1,estr_pct=3.1)])
        fresh=pd.DataFrame([dict(date='2026-11-03',tenor_years=1,estr_pct=float('nan')),
                            dict(date='2026-11-04',tenor_years=1,estr_pct=3.2)])
        result=history.merge_inputs(old,fresh)
        self.assertEqual(len(result),3)
        self.assertEqual(result.estr_pct.tolist(),[3.,3.1,3.2])
        pd.testing.assert_frame_equal(result,history.merge_inputs(result,fresh))


if __name__=='__main__':
    unittest.main()
