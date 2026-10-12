import unittest
from unittest.mock import patch
import pandas as pd
from cmo_price_comparison import build_comparison, START, END, BENCHMARKS


class PriceComparisonTests(unittest.TestCase):
    def test_exact_endpoints_and_units(self):
        dates = pd.to_datetime([START, "2026-09-02", END])
        prices = pd.DataFrame({"Bond A": [100, 99, 98], "Bond B": [None, 99, 98]}, index=dates)
        tba = pd.DataFrame({k+" Price": [100, 98, 97] for k in BENCHMARKS}, index=dates)
        with patch("cmo_price_comparison.pd.read_excel", return_value=tba):
            p = build_comparison(prices, [("Bond A", 6, "FN"), ("Bond B", 7, "G2")], "unused")
        self.assertEqual(p["bonds"][0]["change"], -2)
        self.assertEqual(p["bonds"][0]["label"], "Bond A · 6% cap")
        self.assertEqual(p["bonds"][1]["label"], "Bond B · 7% cap")
        self.assertEqual(p["bonds"][0]["excess"]["FN 5.5"], 1)
        self.assertEqual(p["bonds"][0]["change"]*32, -64)
        self.assertIsNone(p["bonds"][1]["change"])
        self.assertIn(START, p["bonds"][1]["status"])
        self.assertIsNone(p["bonds"][1]["prices"][0])  # no forward-fill
        self.assertEqual(p["dates"], dates.strftime('%Y-%m-%d').tolist())


if __name__ == "__main__":
    unittest.main()
