"""Numerical correctness and no-current-price-leakage checks."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adhoc_cmo_caps import analyze, regress
import adhoc_cmo_caps as caps


class CmoModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(321)
        self.x = pd.DataFrame(rng.normal(size=(150, 4)),
                              columns=["CT10", "cc", "2s10s", "vol"],
                              index=pd.bdate_range("2025-01-01", periods=150))
        self.y = 98 + self.x @ np.array([-2., .05, .3, -.8]) + rng.normal(0, .01, 150)

    def test_ols_optimality_and_units(self):
        model = regress(self.y, self.x)
        residual = model["resid"].to_numpy()
        design = np.column_stack([np.ones(len(self.x)), self.x])
        np.testing.assert_allclose(design.T @ residual, 0, atol=1e-8)
        scaled = self.x.copy()
        scaled["CT10"] *= 100
        converted = regress(self.y, scaled)
        np.testing.assert_allclose(model["fit"], converted["fit"], atol=1e-10)
        self.assertAlmostEqual(model["b"]["CT10"], converted["b"]["CT10"]*100)

    def test_current_price_excluded_and_added_factors_help(self):
        data = self.x.rename(columns={"cc": "FN CC"}).assign(Bond=self.y)
        original = analyze(data, "Bond", "FN")
        changed = data.copy()
        changed.iloc[-1, changed.columns.get_loc("Bond")] += 10
        updated = analyze(changed, "Bond", "FN")
        self.assertAlmostEqual(original["models"]["rolling"]["fit"].iloc[-1],
                               updated["models"]["rolling"]["fit"].iloc[-1])
        self.assertEqual(original["models"]["rolling"]["end"], data.index[-2])
        self.assertEqual(original["models"]["rolling"]["n"], 120)
        self.assertGreater(original["validation"]["improvement"], 90)

    def test_rank_deficiency_does_not_abort_report(self):
        data = self.x.rename(columns={"cc": "FN CC"}).assign(Bond=self.y)
        data["vol"] = data["2s10s"] * 2
        result = analyze(data, "Bond", "FN")
        self.assertIsNone(result["reg"])
        self.assertIn("collinear", result["error"])

    def test_reference_partial_response_and_cache_fallback(self):
        response = pd.DataFrame({"FLT_SPREAD": [-65., np.nan],
                                 "MTG_COLLAT_TYP": ["G2SF 5.5", "FNCL   6"]}, index=["A", "B"])
        with TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="cmo_reference_test_") as directory:
            with patch.object(caps, "BONDS", [("A", 6., "FN"), ("B", 6., "FN")]), \
                 patch.object(caps, "REFERENCE_CACHE", Path(directory)/"reference.csv"), \
                 patch("xbbg.blp.bdp", return_value=response):
                live = caps.load_reference()
                self.assertEqual(live["A"]["flt_spread"], -65.)
                self.assertIsNone(live["B"]["flt_spread"])
                self.assertEqual(live["A"]["underlying_cpn"], 5.5)
                self.assertEqual(live["B"]["underlying_cpn"], 6.)
                self.assertTrue(live["A"]["asof"])
                with patch("xbbg.blp.bdp", return_value=pd.DataFrame()), \
                     patch.object(caps, "_reference_direct", side_effect=RuntimeError("Offline")):
                    fallback = caps.load_reference()
                self.assertEqual(fallback["A"]["flt_spread"], -65.)
                self.assertEqual(fallback["A"]["asof"], live["A"]["asof"])
                self.assertIn("Cached", fallback["A"]["status"])
                self.assertEqual(fallback["B"]["mtg_collat_typ"], "FNCL   6")
                self.assertEqual(fallback["B"]["underlying_cpn"], 6.)

    def test_coupon_parser_rejects_ambiguous_collateral(self):
        self.assertEqual(caps._collateral_coupon("G2SF 5.5"), 5.5)
        self.assertEqual(caps._collateral_coupon("FNCL   6"), 6.)
        for description in [None, "Mixed", "FNCL 5.5/6", "FNCL 30YR", "5.5", "FNCL 99"]:
            self.assertIsNone(caps._collateral_coupon(description))


if __name__ == "__main__":
    unittest.main()
