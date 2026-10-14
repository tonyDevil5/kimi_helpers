import unittest
import numpy as np
import pandas as pd
from fund_flows_monitor import prepare


class FundFlowTests(unittest.TestCase):
    def inputs(self):
        universe=pd.DataFrame({'ticker':['A','P','M'],'group':['Active','Passive','Active'],
                               'name':['a','p','m'],'fund_type':['Fund']*3,'workbook_aum':['n/a']*3})
        ref=pd.DataFrame({'ticker':['A','P','M'],'NAV_CRNCY':['USD','EUR','USD'],
                          'NAME':['a','p','m'],'ID_BB_GLOBAL':['1','2','3']})
        dates=pd.bdate_range('2026-07-01','2026-10-09')
        raw=pd.concat([pd.DataFrame({'ticker':t,'date':dates,'FUND_FLOW':v}) for t,v in [('A',2),('P',-1)]],ignore_index=True)
        raw=pd.concat([raw,pd.DataFrame({'ticker':['M']*3,'date':pd.to_datetime(['2026-07-31','2026-08-31','2026-09-30']),'FUND_FLOW':[500]*3})],ignore_index=True)
        fx=pd.DataFrame({'currency':'EUR','date':dates,'PX_LAST':1.2})
        return universe,ref,raw,fx

    def test_currency_weekly_sum_and_monthly_exclusion(self):
        meta,d,weekly,agg,end=prepare(*self.inputs(),'2026-10-12')
        self.assertEqual(str(end.date()),'2026-10-09')
        self.assertEqual(meta.weekly_reporter.tolist(),[True,True,False])
        last=agg[agg.week_ending==end].set_index('group')
        self.assertAlmostEqual(last.loc['Active','net_usd_mm'],10)
        self.assertAlmostEqual(last.loc['Passive','net_usd_mm'],-6)
        self.assertAlmostEqual(last.loc['All','net_usd_mm'],4)
        self.assertEqual(last.loc['All','universe_funds'],3)
        self.assertEqual(last.loc['All','eligible_weekly_funds'],2)
        self.assertTrue(last.loc['All','complete_eligible_coverage'])
        # Sparse September flow remains raw, and does not enter weekly aggregates.
        self.assertEqual(d.loc[d.ticker=='M','flow_usd_mm'].sum(),1500)
        by_week=agg.pivot(index='week_ending',columns='group',values='net_usd_mm')
        np.testing.assert_allclose(by_week.All,by_week.Active+by_week.Passive)

    def test_missing_is_not_zero(self):
        u,r,d,fx=self.inputs()
        d=d[~((d.ticker=='P')&(d.date>=pd.Timestamp('2026-10-05')))]
        meta,d,w,a,end=prepare(u,r,d,fx,'2026-10-12')
        self.assertTrue(pd.isna(w.loc[end,'P']))
        last=a[(a.week_ending==end)&(a.group=='All')].iloc[0]
        self.assertFalse(last.complete_eligible_coverage)
        self.assertEqual(last.reporting_funds,1)

    def test_no_future_or_stale_fx(self):
        u,r,d,fx=self.inputs()
        fx=fx[fx.date==pd.Timestamp('2026-10-09')]
        _,converted,_,_,_=prepare(u,r,d,fx,'2026-10-12')
        p=converted[converted.ticker=='P']
        self.assertTrue(p.loc[p.date<pd.Timestamp('2026-10-09'),'flow_usd_mm'].isna().all())
        self.assertAlmostEqual(p.loc[p.date==pd.Timestamp('2026-10-09'),'flow_usd_mm'].iloc[0],-1.2)

    def test_duplicate_share_class_is_rejected(self):
        u,r,d,fx=self.inputs()
        r.loc[1,'ID_BB_GLOBAL']='1'
        with self.assertRaises(ValueError):prepare(u,r,d,fx,'2026-10-12')


if __name__=='__main__':unittest.main()
