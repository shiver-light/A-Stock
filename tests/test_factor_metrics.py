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
from analysis.factor_report import build_factor_diagnostics_report, render_factor_report_text


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

    def test_build_factor_diagnostics_report_basic_summary(self) -> None:
        ic_data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "horizon": [5, 5],
                "ic": [0.1, 0.3],
                "n_obs": [100, 100],
                "method": ["pearson", "pearson"],
            }
        )
        rank_ic_data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "horizon": [5, 5],
                "ic": [0.2, 0.4],
                "n_obs": [100, 100],
                "method": ["spearman", "spearman"],
            }
        )
        coverage_data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "coverage": [0.8, 0.6],
                "non_null_count": [80, 60],
                "total_count": [100, 100],
            }
        )
        quantile_returns = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "quantile": [1, 5, 1, 5],
                "horizon": [5, 5, 5, 5],
                "forward_return": [0.01, 0.05, 0.02, 0.06],
                "n_obs": [20, 20, 20, 20],
            }
        )

        report = build_factor_diagnostics_report(
            factor_name="return_20d",
            horizon=5,
            ic_data=ic_data,
            rank_ic_data=rank_ic_data,
            coverage_data=coverage_data,
            quantile_returns=quantile_returns,
        )

        self.assertEqual(report["factor_name"], "return_20d")
        self.assertEqual(report["horizon"], 5)
        self.assertAlmostEqual(report["ic_mean"], 0.2)
        self.assertAlmostEqual(report["rank_ic_mean"], 0.3)
        self.assertAlmostEqual(report["coverage"], 0.7)
        self.assertEqual(len(report["quantile_return_summary"]), 2)

    def test_build_factor_diagnostics_report_handles_empty_metrics(self) -> None:
        report = build_factor_diagnostics_report(
            factor_name="return_20d",
            horizon=5,
            ic_data=pd.DataFrame(columns=["trade_date", "horizon", "ic", "n_obs", "method"]),
            rank_ic_data=pd.DataFrame(columns=["trade_date", "horizon", "ic", "n_obs", "method"]),
            coverage_data=pd.DataFrame(columns=["trade_date", "coverage", "non_null_count", "total_count"]),
            quantile_returns=pd.DataFrame(columns=["trade_date", "quantile", "horizon", "forward_return", "n_obs"]),
        )

        self.assertIsNone(report["ic_mean"])
        self.assertIsNone(report["icir"])
        self.assertEqual(report["quantile_return_summary"], [])

    def test_render_factor_report_text_contains_key_fields(self) -> None:
        report = {
            "factor_name": "return_20d",
            "horizon": 5,
            "ic_mean": 0.1,
            "ic_std": 0.05,
            "icir": 2.0,
            "rank_ic_mean": 0.12,
            "coverage": 0.8,
            "observation_summary": {"ic_dates": 10, "rank_ic_dates": 10, "coverage_dates": 12},
            "quantile_return_summary": [
                {"quantile": 1, "mean_return": -0.01, "mean_count": 30.0, "date_count": 10},
                {"quantile": 5, "mean_return": 0.02, "mean_count": 30.0, "date_count": 10},
            ],
            "risk": ["基于历史样本统计，不保证未来有效。"],
        }

        text = render_factor_report_text(report)

        self.assertIn("factor name: return_20d", text)
        self.assertIn("horizon: 5d", text)
        self.assertIn("ic mean: 0.100000", text)
        self.assertIn("q5: mean_return=0.020000", text)


if __name__ == "__main__":
    unittest.main()
