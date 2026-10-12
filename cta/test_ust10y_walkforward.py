import unittest
import numpy as np
import pandas as pd
from ust10y_walkforward import breakout, candidates, daily_pnl, metrics, walkforward


class CausalTests(unittest.TestCase):
    def test_breakout_exits_without_same_day_reversal(self):
        s=pd.Series([1.,2.,3.,4.,0.,-1.])
        np.testing.assert_array_equal(breakout(s,2),[0,0,-1,-1,0,1])

    def test_lag_fractional_dv01(self):
        y=pd.Series([4.,4.1,4.2,4.1])
        target=pd.Series([1.,.5,-1.,0.])
        result=daily_pnl(target,y)
        np.testing.assert_allclose(result.pnl_bps,[0,-10,-5,-10],atol=1e-10)
        np.testing.assert_allclose(result.pnl_usd,result.pnl_bps*1000)

    def test_zero_anchored_drawdown(self):
        frame=pd.DataFrame({'pnl_bps':[-5.,-3.,1.],'held_position':[1.,1.,1.]})
        self.assertEqual(metrics(frame)['Max DD bp'],8)

    def test_prefix_invariance_and_training_cutoff(self):
        rng=np.random.default_rng(17)
        idx=pd.bdate_range('2020-01-01',periods=160)
        data=pd.DataFrame({t:4+np.cumsum(rng.normal(0,.03,len(idx)))
                           for t in ['2Y','5Y','10Y','30Y']},index=idx)
        for family in ['EWMA','KAMA']:
            full,names=candidates(data,family)
            short,_=candidates(data.iloc[:120],family)
            np.testing.assert_array_equal(full[:120],short)
            a,audit=walkforward(full,names,data['10Y'],initial=70,step=10)
            b,_=walkforward(short,names,data['10Y'].iloc[:120],initial=70,step=10)
            np.testing.assert_array_equal(a.iloc[:120],b)
            self.assertTrue((pd.to_datetime(audit.training_end)<pd.to_datetime(audit.decision_date)).all())
            self.assertTrue(set(a.unique()).issubset({-1,-.5,0,.5,1}))


if __name__=='__main__': unittest.main()
