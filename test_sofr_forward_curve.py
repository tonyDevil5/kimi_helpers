import unittest

import QuantLib as ql

from sofr_forward_curve import TENORS, FORWARDS, build_curve, forward_rate, validate_quotes


class SofrCurveTests(unittest.TestCase):
    def setUp(self):
        self.previous_date = ql.Settings.instance().evaluationDate
        self.asof = ql.Date(2, 11, 2026)
        self.rows = [dict(tenor_years=n, lch_rate_pct=3.3 + .025*n,
                          basis_bps=1 + .03*n) for n in TENORS]

    def tearDown(self):
        ql.Settings.instance().evaluationDate = self.previous_date

    def test_independent_spot_swaps_reprice_both_venues(self):
        for venue in ("LCH", "CME"):
            curve, _ = build_curve(self.rows, venue, self.asof)
            for row in self.rows:
                fair, _, _ = forward_rate(curve, self.asof, 0, row["tenor_years"])
                expected = row["lch_rate_pct"] / 100
                if venue == "CME":
                    expected += row["basis_bps"] / 10000
                self.assertAlmostEqual(fair, expected, places=10)

    def test_zero_basis_identical_forwards(self):
        rows = [dict(row, basis_bps=0) for row in self.rows]
        lch, _ = build_curve(rows, "LCH", self.asof)
        cme, _ = build_curve(rows, "CME", self.asof)
        for start, length in FORWARDS:
            self.assertEqual(forward_rate(lch, self.asof, start, length),
                             forward_rate(cme, self.asof, start, length))

    def test_basis_sign_changes_cme_direction(self):
        lch, _ = build_curve(self.rows, "LCH", self.asof)
        plus, _ = build_curve(self.rows, "CME", self.asof, 1)
        minus, _ = build_curve(self.rows, "CME", self.asof, -1)
        for start, length in FORWARDS:
            base = forward_rate(lch, self.asof, start, length)[0]
            self.assertGreater(forward_rate(plus, self.asof, start, length)[0], base)
            self.assertLess(forward_rate(minus, self.asof, start, length)[0], base)

    def test_missing_duplicate_and_nan_quotes_rejected(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1],
                     [dict(self.rows[0], basis_bps=float("nan"))] + self.rows[1:]):
            with self.assertRaises(ValueError):
                validate_quotes(rows)


if __name__ == "__main__":
    unittest.main()
