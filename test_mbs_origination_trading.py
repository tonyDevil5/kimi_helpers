import unittest
import numpy as np
import pandas as pd
from mbs_origination_trading import prepare,PerformancePositionBacktester


class TradingTests(unittest.TestCase):
    def test_engine_lag_cost_and_dollar_scale(self):
        d=pd.DataFrame(dict(daily_ticks=[0,1,2,3,4],target=[1,1,-1,-1,0]),index=pd.bdate_range('2026-01-01',periods=5))
        r=PerformancePositionBacktester(d,10_000_000,1,.5).run()
        np.testing.assert_allclose(r.gross_pnl_usd,[0,0,6250,9375,-12500])
        np.testing.assert_allclose(r.cost_usd,[0,1562.5,0,3125,0])

    def test_future_invariance_and_missing_origination(self):
        rng=np.random.default_rng(8);idx=pd.bdate_range('2020-01-01',periods=600)
        orig=pd.Series(rng.lognormal(4,1,len(idx)),index=idx);perf=pd.Series(rng.normal(size=len(idx)),index=idx)
        z,raw,target=prepare(orig,perf)
        altered=orig.copy();altered.iloc[500:]*=100
        zz,rr,tt=prepare(altered,perf)
        np.testing.assert_allclose(z.iloc[:500],zz.iloc[:500],equal_nan=True)
        np.testing.assert_allclose(target.iloc[:500],tt.iloc[:500])
        orig.iloc[-1]=np.nan
        z,_,target=prepare(orig,perf)
        self.assertTrue(pd.isna(z.iloc[-1]));self.assertEqual(target.iloc[-1],0)
        self.assertLessEqual(target.abs().max(),1)
