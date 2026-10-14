import unittest
import numpy as np
import pandas as pd
from fund_sensitivity_monitor import FACTORS,TRI,fit_model,weekly_endpoints,make_weekly,group_returns,ytd_total_return


class SensitivityTests(unittest.TestCase):
    def test_ytd_uses_exact_year_end_not_first_week_return(self):
        s=pd.Series([95,100,101,110],index=pd.to_datetime(['2025-12-26','2025-12-31','2026-01-02','2026-10-09']))
        ytd=ytd_total_return(s,'2026-10-09')
        self.assertAlmostEqual(ytd['ytd_return_pct'],10)
        self.assertEqual(ytd['ytd_start_observation'],'2025-12-31')
        self.assertTrue(pd.isna(ytd_total_return(s.drop(pd.Timestamp('2025-12-31')),'2026-10-09')['ytd_return_pct']))

    def test_regression_recovers_return_bps_per_factor_bp(self):
        rng=np.random.default_rng(7)
        idx=pd.date_range('2024-01-05',periods=100,freq='W-FRI')
        x=pd.DataFrame(rng.normal(size=(100,5)),columns=FACTORS,index=idx)
        beta=np.array([-5,-2,-3,.5,-.1])
        y=pd.Series(4+x.to_numpy()@beta,index=idx)
        m,d=fit_model(y,x)
        np.testing.assert_allclose(m.params.iloc[1:],beta,atol=1e-10)
        self.assertAlmostEqual(m.params['const'],4)
        self.assertAlmostEqual(m.rsquared,1)

    def test_stale_weekly_endpoint_not_filled(self):
        s=pd.Series([100,101,102],index=pd.to_datetime(['2026-09-25','2026-10-05','2026-10-09']))
        w=weekly_endpoints(s,pd.Timestamp('2026-10-09'))
        self.assertTrue(pd.isna(w.loc['2026-10-02']))
        self.assertTrue(pd.isna(w.pct_change(fill_method=None).loc['2026-10-09']))
        monday=pd.Series([100],index=pd.to_datetime(['2026-10-05']))
        self.assertTrue(pd.isna(weekly_endpoints(monday,pd.Timestamp('2026-10-09')).iloc[0]))

    def test_usd_returns_include_currency_and_keep_missing_basket_missing(self):
        dates=pd.to_datetime(['2026-09-18','2026-09-25','2026-10-02'])
        u=pd.DataFrame({'ticker':['A','P'],'group':['Active','Passive'],'name':['a','p']})
        ref=pd.DataFrame({'ticker':['A','P'],'CRNCY':['EUR','USD'],'ID_BB_GLOBAL':['1','2']})
        raw=pd.DataFrame({'ticker':['A']*3+['P']*3,'date':list(dates)*2,TRI:[100,101,102,100,100,100]})
        fx=pd.DataFrame({'currency':['EUR']*3,'date':dates,'PX_LAST':[1,1.1,1.1]})
        f=pd.DataFrame({c:[1,2,3] for c in FACTORS},index=dates)
        meta,r,x=make_weekly(u,ref,raw,fx,f,pd.Timestamp('2026-10-02'))
        self.assertAlmostEqual(r.loc['2026-09-25','A'],1110)
        self.assertAlmostEqual(r.loc['2026-10-02','A'],(102/101-1)*10000)
        basket=group_returns(meta,r)
        self.assertAlmostEqual(basket.loc['2026-09-25','All equal weight'],555)
        r.loc['2026-10-02','P']=np.nan
        self.assertTrue(pd.isna(group_returns(meta,r).loc['2026-10-02','All equal weight']))

    def test_insufficient_and_singular_inputs_not_estimated(self):
        x=pd.DataFrame(np.ones((50,5)),columns=FACTORS)
        m,d=fit_model(pd.Series(np.arange(50)),x)
        self.assertIsNone(m)
        m,d=fit_model(pd.Series(np.arange(10)),x.iloc[:10])
        self.assertIsNone(m)


if __name__=='__main__':unittest.main()
