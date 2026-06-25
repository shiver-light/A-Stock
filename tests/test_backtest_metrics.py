from __future__ import annotations

import math
import unittest

import pandas as pd

from backtest.metrics import calc_performance, calc_relative_performance
from reports.formatter import format_strategy_report, render_strategy_report_text


class BacktestMetricsTestCase(unittest.TestCase):
    def _strategy_returns(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["20240131", "20240229", "20240329", "20240430", "20240531", "20240628"],
                "strategy_return": [0.10, -0.05, 0.02, 0.03, 0.04, -0.01],
                "turnover": [1.00, 0.00, 0.50, 0.00, 0.75, 0.25],
            }
        )

    def _returns_with_benchmark(self) -> pd.DataFrame:
        data = self._strategy_returns().copy()
        data["benchmark_return"] = [0.08, -0.04, 0.01, 0.03, 0.01, 0.01]
        data["excess_return"] = data["strategy_return"] - data["benchmark_return"]
        return data

    def test_calc_performance_includes_turnover_and_positive_month_ratio(self) -> None:
        result = calc_performance(self._strategy_returns())

        self.assertAlmostEqual(result["mean_daily_turnover"], (1.0 + 0.0 + 0.5 + 0.0 + 0.75 + 0.25) / 6.0)
        self.assertAlmostEqual(result["mean_rebalance_turnover"], (1.0 + 0.5 + 0.75 + 0.25) / 4.0)
        self.assertAlmostEqual(result["median_rebalance_turnover"], 0.625)
        self.assertAlmostEqual(result["max_rebalance_turnover"], 1.0)
        self.assertAlmostEqual(result["positive_month_ratio"], 4.0 / 6.0)

    def test_calc_performance_includes_forward_max_gain_diagnostics(self) -> None:
        returns = pd.DataFrame(
            {
                "trade_date": [
                    "20240102",
                    "20240103",
                    "20240104",
                    "20240105",
                    "20240108",
                    "20240109",
                    "20240110",
                    "20240111",
                    "20240112",
                    "20240115",
                ],
                "strategy_return": [0.00, 0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, 0.01],
            }
        )

        result = calc_performance(returns)
        forward_metrics = result["forward_max_gain"]

        expected_first_3d = max(
            0.02,
            (1.02 * 0.99) - 1.0,
            (1.02 * 0.99 * 1.03) - 1.0,
        )
        expected_latest_1d = 0.01
        expected_latest_1w = max(
            0.01,
            (1.01 * 1.04) - 1.0,
            (1.01 * 1.04 * 0.99) - 1.0,
            (1.01 * 1.04 * 0.99 * 1.02) - 1.0,
            (1.01 * 1.04 * 0.99 * 1.02 * 1.01) - 1.0,
        )

        self.assertEqual(len(forward_metrics), 10)
        self.assertAlmostEqual(forward_metrics[0]["forward_3d_max_gain"], expected_first_3d)
        self.assertAlmostEqual(result["latest_forward_1d_max_gain"], expected_latest_1d)
        self.assertAlmostEqual(result["latest_forward_1w_max_gain"], expected_latest_1w)
        self.assertGreater(result["best_forward_7d_max_gain"], 0.0)

    def test_calc_relative_performance_includes_monthly_and_rolling_metrics(self) -> None:
        returns = self._returns_with_benchmark()
        result = calc_relative_performance(returns)

        monthly_excess = pd.Series([0.02, -0.01, 0.01, 0.00, 0.03, -0.02], dtype="float64")
        first_window = monthly_excess.iloc[:5]
        second_window = monthly_excess.iloc[1:]
        expected_latest_rolling_excess = float((1.0 + second_window).prod() - 1.0)
        expected_mean_rolling_excess = float(
            (
                ((1.0 + first_window).prod() - 1.0)
                + ((1.0 + second_window).prod() - 1.0)
            )
            / 2.0
        )
        first_window_sharpe = float((first_window.mean() / first_window.std(ddof=0)) * math.sqrt(12))
        second_window_sharpe = float((second_window.mean() / second_window.std(ddof=0)) * math.sqrt(12))

        self.assertAlmostEqual(result["positive_excess_month_ratio"], 3.0 / 6.0)
        self.assertEqual(len(result["rolling_5m_excess_return"]), 2)
        self.assertEqual(len(result["rolling_5m_sharpe"]), 2)
        self.assertAlmostEqual(result["latest_rolling_5m_excess_return"], expected_latest_rolling_excess)
        self.assertAlmostEqual(result["mean_rolling_5m_excess_return"], expected_mean_rolling_excess)
        self.assertAlmostEqual(
            result["worst_rolling_5m_excess_return"],
            min(
                float((1.0 + first_window).prod() - 1.0),
                float((1.0 + second_window).prod() - 1.0),
            ),
        )
        self.assertAlmostEqual(result["latest_rolling_5m_sharpe"], second_window_sharpe)
        self.assertAlmostEqual(result["mean_rolling_5m_sharpe"], (first_window_sharpe + second_window_sharpe) / 2.0)
        self.assertAlmostEqual(result["worst_rolling_5m_sharpe"], min(first_window_sharpe, second_window_sharpe))

    def test_report_formatter_includes_new_metric_sections(self) -> None:
        performance = calc_performance(self._strategy_returns())
        performance.update(calc_relative_performance(self._returns_with_benchmark()))

        report = format_strategy_report(
            {"top_stocks": []},
            performance,
            {"universe": ["000300.SH"], "benchmark_code": "000300.SH", "factor_config": {}},
        )
        report_text = render_strategy_report_text(report)

        self.assertIn("mean_daily_turnover", report["backtest_summary"])
        self.assertIn("positive_excess_month_ratio", report["benchmark"])
        self.assertIn("rolling_5m_excess_return", report["robustness"])
        self.assertIn("latest_forward_3d_max_gain", report["forward_max_gain"])
        self.assertIn("mean daily turnover", report_text)
        self.assertIn("positive excess month ratio", report_text)
        self.assertIn("latest rolling 5m excess return", report_text)
        self.assertIn("latest forward 3d max gain", report_text)


if __name__ == "__main__":
    unittest.main()
