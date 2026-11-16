import unittest
import numpy as np
import pandas as pd
from mbs_origination_signal import features, simulate


class SignalTests(unittest.TestCase):
    def test_future_changes_do_not_change_earlier_forecasts_or_positions(self):
        rng=np.random.default_rng(71)
        dates=pd.bdate_range('2020-01-01',periods=700)
        orig=pd.Series(rng.lognormal(4,1,700),index=dates)
        perf=pd.Series(rng.normal(0,2,700),index=dates)
        _,daily,full=simulate(features(orig,perf),5)
        changed=perf.copy();changed.iloc[600:]+=50
        revised_orig=orig.copy();revised_orig.iloc[600:]*=10
        _,later,revised=simulate(features(revised_orig,changed),5)
        for col in ['orig_model','perf_baseline','combined','orig_z']:
            np.testing.assert_allclose(full[col].iloc[:600],revised[col].iloc[:600],equal_nan=True)
        pd.testing.assert_frame_equal(daily[daily.date<str(dates[600].date())].reset_index(drop=True),
            later[later.date<str(dates[600].date())].reset_index(drop=True))

    def test_forward_label_excludes_signal_day(self):
        dates=pd.bdate_range('2020-01-01',periods=400)
        d=features(pd.Series(1.,index=dates),pd.Series(np.arange(400.),index=dates),20,5)
        self.assertEqual(d.future.iloc[0],sum(range(1,6)))
        self.assertTrue(d.future.iloc[-5:].isna().all())
