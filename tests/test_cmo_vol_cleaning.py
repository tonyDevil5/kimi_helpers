import unittest
from unittest.mock import patch
import pandas as pd
from adhoc_cmo_caps import clean_vol


class VolCleaningTests(unittest.TestCase):
    @patch("pandas.DataFrame.to_csv")
    def test_reviewed_quotes_only_and_idempotent(self, save):
        dates = pd.to_datetime(["2023-06-21", "2023-06-22", "2023-06-23",
                                "2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"])
        raw = pd.DataFrame({"vol": [101.506, 57.348, 102.266, 101.964, 62, 101.746, None]}, index=dates)
        cleaned = clean_vol(raw)
        self.assertEqual(cleaned.loc["2023-06-22", "vol"], 101.506)
        self.assertEqual(cleaned.loc["2024-06-10", "vol"], 101.964)
        self.assertEqual(cleaned.loc["2023-06-22", "vol_raw"], 57.348)
        self.assertEqual(cleaned.loc["2024-06-10", "vol_raw"], 62)
        self.assertTrue(pd.isna(cleaned.loc["2024-06-12", "vol"]))
        self.assertEqual(cleaned.vol_replaced.sum(), 2)
        pd.testing.assert_frame_equal(cleaned, clean_vol(cleaned))
        self.assertEqual(raw.loc["2023-06-22", "vol"], 57.348)

    @patch("pandas.DataFrame.to_csv")
    def test_no_future_value_used(self, save):
        raw = pd.DataFrame({"vol": [57.348, 102.266]},
                           index=pd.to_datetime(["2023-06-22", "2023-06-23"]))
        result = clean_vol(raw)
        self.assertTrue(pd.isna(result.vol.iloc[0]))
        self.assertFalse(result.vol_replaced.iloc[0])


if __name__ == "__main__":
    unittest.main()
