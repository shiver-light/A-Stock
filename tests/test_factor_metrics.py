from __future__ import annotations

import unittest

import pandas as pd

from analysis.factor_metrics import (
    calc_factor_coverage,
    calc_forward_returns,
    calc_ic,
    calc_quantile_groups,
    calc_quantile_returns,
    calc_rank_ic,
)


class FactorMetricsTestCase(unittest.TestCase):
    def test_calc_forward_returns_empty_input(self) -> None:
        price_data = pd.DataFrame(columns=["trade_date", "ts_code", "close"])

        result = calc_forward_returns(price_data, [1, 5])

        self.assertEqual(
            list(result.columns),
            ["trade_date", "ts_code", "forward_return_1d", "forward_return_5d"],
        )
        self.assertTrue(result.empty)

    def test_calc_forward_returns_basic_values(self) -> None:
        price_data = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "close": [10.0, 11.0, 12.0],
            }
        )

        result = calc_forward_returns(price_data, [1])

        self.assertAlmostEqual(result.loc[0, "forward_return_1d"], 0.1)
        self.assertAlmostEqual(result.loc[1, "forward_return_1d"], 12.0 / 11.0 - 1.0)
        self.assertTrue(pd.isna(result.loc[2, "forward_return_1d"]))

    def test_calc_ic_returns_empty_for_empty_input(self) -> None:
        factor_data = pd.DataFrame(columns=["trade_date", "ts_code", "factor_value"])
        forward_returns = pd.DataFrame(columns=["trade_date", "ts_code", "forward_return_1d"])

        result = calc_ic(factor_data, forward_returns)

        self.assertEqual(list(result.columns), ["trade_date", "horizon", "ic", "n_obs", "method"])
        self.assertTrue(result.empty)

    def test_calc_ic_positive_and_negative_samples(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 3 + ["20240103"] * 3,
                "ts_code": ["A", "B", "C"] * 2,
                "factor_value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )
        forward_returns = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 3 + ["20240103"] * 3,
                "ts_code": ["A", "B", "C"] * 2,
                "forward_return_1d": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
            }
        )

        result = calc_ic(factor_data, forward_returns)

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc[0, "ic"], 1.0)
        self.assertAlmostEqual(result.loc[1, "ic"], -1.0)

    def test_calc_ic_skips_insufficient_cross_section(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "ts_code": ["A"],
                "factor_value": [1.0],
            }
        )
        forward_returns = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "ts_code": ["A"],
                "forward_return_1d": [0.02],
            }
        )

        result = calc_ic(factor_data, forward_returns)

        self.assertTrue(result.empty)

    def test_calc_rank_ic_negative_sample(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["A", "B", "C", "D"],
                "factor_value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        forward_returns = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["A", "B", "C", "D"],
                "forward_return_1d": [0.04, 0.03, 0.02, 0.01],
            }
        )

        result = calc_rank_ic(factor_data, forward_returns)

        self.assertEqual(result.loc[0, "method"], "spearman")
        self.assertAlmostEqual(result.loc[0, "ic"], -1.0)

    def test_calc_quantile_groups_handles_empty_input(self) -> None:
        factor_data = pd.DataFrame(columns=["trade_date", "ts_code", "factor_value"])

        result = calc_quantile_groups(factor_data, n_quantiles=5)

        self.assertEqual(list(result.columns), ["trade_date", "ts_code", "factor_value", "quantile"])
        self.assertTrue(result.empty)

    def test_calc_quantile_groups_assigns_boundary_quantiles(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 5,
                "ts_code": ["A", "B", "C", "D", "E"],
                "factor_value": [1, 2, 3, 4, 5],
            }
        )

        result = calc_quantile_groups(factor_data, n_quantiles=5)

        self.assertEqual(result["quantile"].tolist(), [1, 2, 3, 4, 5])

    def test_calc_quantile_groups_preserves_missing_values(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 3,
                "ts_code": ["A", "B", "C"],
                "factor_value": [1.0, None, 3.0],
            }
        )

        result = calc_quantile_groups(factor_data, n_quantiles=3)

        self.assertTrue(pd.isna(result.loc[result["ts_code"] == "B", "quantile"]).all())

    def test_calc_quantile_returns_basic_output(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["A", "B", "C", "D"],
                "factor_value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        forward_returns = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["A", "B", "C", "D"],
                "forward_return_1d": [0.01, 0.02, 0.03, 0.04],
            }
        )

        result = calc_quantile_returns(factor_data, forward_returns, n_quantiles=2)

        self.assertEqual(result["quantile"].tolist(), [1, 2])
        self.assertAlmostEqual(result.loc[0, "forward_return"], 0.015)
        self.assertAlmostEqual(result.loc[1, "forward_return"], 0.035)

    def test_calc_factor_coverage_handles_duplicates_and_missing(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240102", "20240102"],
                "ts_code": ["A", "A", "B", "C"],
                "factor_value": [1.0, None, 2.0, None],
            }
        )

        result = calc_factor_coverage(factor_data)

        self.assertEqual(result.loc[0, "total_count"], 3)
        self.assertEqual(result.loc[0, "non_null_count"], 1)
        self.assertAlmostEqual(result.loc[0, "coverage"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
