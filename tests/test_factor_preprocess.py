from __future__ import annotations

import unittest

import pandas as pd

from factors.preprocess import preprocess_factor_panel


class FactorPreprocessTestCase(unittest.TestCase):
    def test_empty_input_returns_empty_result(self) -> None:
        data = pd.DataFrame(columns=["trade_date", "ts_code"])

        result, warnings = preprocess_factor_panel(data, factor_cols=["factor_a"])

        self.assertTrue(result.empty)
        self.assertIn("factor_a", result.columns)
        self.assertEqual(warnings, [])

    def test_fill_clip_standardize_and_direction_align(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "industry": ["bank", "bank", "broker", "broker"],
                "factor_a": [1.0, None, 3.0, 100.0],
                "factor_b": [2.0, 4.0, 6.0, 8.0],
            }
        )

        result, warnings = preprocess_factor_panel(
            data,
            factor_cols=["factor_a", "factor_b"],
            industry_col="industry",
            directions={"factor_a": 1, "factor_b": -1},
        )

        self.assertEqual(warnings, [])
        self.assertFalse(result["factor_a"].isna().any())
        self.assertFalse(result["factor_b"].isna().any())
        self.assertAlmostEqual(float(result["factor_a"].mean()), 0.0, places=7)
        self.assertAlmostEqual(float(result["factor_b"].mean()), 0.0, places=7)
        self.assertLess(float(result.loc[result["ts_code"] == "000004.SZ", "factor_a"].iloc[0]), 2.0)
        ordered = result.sort_values("factor_b", ascending=False)["ts_code"].tolist()
        self.assertEqual(ordered[0], "000001.SZ")

    def test_zero_std_and_neutralization_warnings(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 3,
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "factor_a": [5.0, 5.0, 5.0],
            }
        )

        result, warnings = preprocess_factor_panel(
            data,
            factor_cols=["factor_a"],
            neutralize_industry=True,
            neutralize_market_cap=True,
        )

        self.assertTrue((result["factor_a"] == 0.0).all())
        self.assertTrue(any("std is zero" in warning for warning in warnings))
        self.assertTrue(any("industry neutralization requested" in warning for warning in warnings))
        self.assertTrue(any("market_cap neutralization requested" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
