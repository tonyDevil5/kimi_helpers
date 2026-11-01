"""Numerical checks for coupon rolling OLS and selectable JPM performance."""
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

import mbs_rv_report as rv
from mbs_rv_coupon_views import performance_payload


class CouponViewsTests(unittest.TestCase):
    def test_coupon_order_is_agency_then_numeric(self):
        names = ['G2 6.5', 'FN 6', 'G2 2', 'FN 2.5', 'FN 2']
        self.assertEqual(sorted(names, key=rv.coupon_sort_key),
                         ['FN 2', 'FN 2.5', 'FN 6', 'G2 2', 'G2 6.5'])

    def test_rolling_matches_direct_fit_and_has_no_future_inputs(self):
        rng = np.random.default_rng(82)
        dates = pd.bdate_range("2025-01-01", periods=150)
        frame = pd.DataFrame(rng.normal(size=(150, 3)), index=dates, columns=rv.COUPON_BASIS_PREDICTORS)
        frame["FN 5"] = 90 + frame["10y"]*4 - frame["2s10s"]*2 + rng.normal(size=150)
        frame["G2 5"] = frame["FN 5"] + 12
        result = rv.run_rolling_coupon_basis(frame)
        self.assertEqual(len(result), 2*31)
        sub = result[result.structure == "FN 5"]
        self.assertEqual(sub.index[0], dates[119])
        direct = sm.OLS(frame["FN 5"].iloc[:120], sm.add_constant(frame[rv.COUPON_BASIS_PREDICTORS].iloc[:120])).fit()
        self.assertAlmostEqual(sub.iloc[0].mdl_val, direct.fittedvalues.iloc[-1])
        self.assertAlmostEqual(sub.iloc[0].resid_z, direct.resid.iloc[-1]/np.std(direct.resid))
        changed = frame.copy()
        changed.loc[dates[120]:, "FN 5"] += 1000
        after = rv.run_rolling_coupon_basis(changed)
        self.assertAlmostEqual(after[after.structure == "FN 5"].iloc[0].mdl_val, sub.iloc[0].mdl_val)

    def test_performance_uses_latest_data_year_and_preserves_missing(self):
        frame = pd.DataFrame({"FN 5 Perf": [100, 1, np.nan, 3]},
                             index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-10-01", "2026-10-02"]))
        payload = performance_payload(frame)
        self.assertEqual(payload["minimum"], "2026-01-01")
        self.assertEqual(payload["default"], "2026-10-01")
        self.assertEqual(payload["series"]["FN 5"], [1, None, 3])


if __name__ == "__main__":
    unittest.main()
