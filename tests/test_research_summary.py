from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.storage import write_json
from research.summary import build_research_summary, rebuild_summary_from_disk


class ResearchSummaryTestCase(unittest.TestCase):
    def test_build_research_summary_includes_extended_metrics(self) -> None:
        summary = build_research_summary(
            [
                {
                    "name": "demo_exp",
                    "config": {
                        "universe_name": "hs300",
                        "top_n": 10,
                        "benchmark_code": "000300.SH",
                        "factor_config": {"turnover_mean_20d": 1.0},
                        "market_regime_filter": {"min_return_20d": 0.0},
                    },
                    "performance": {
                        "cumulative_return": 0.1,
                        "sharpe": 0.5,
                        "excess_cumulative_return": 0.08,
                        "positive_excess_month_ratio": 0.6,
                        "mean_rebalance_turnover": 0.7,
                        "worst_rolling_5m_excess_return": -0.1,
                    },
                }
            ]
        )

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["name"], "demo_exp")
        self.assertEqual(row["positive_excess_month_ratio"], 0.6)
        self.assertEqual(row["mean_rebalance_turnover"], 0.7)
        self.assertEqual(row["worst_rolling_5m_excess_return"], -0.1)
        self.assertEqual(row["signal_filters"], [])
        self.assertEqual(row["market_regime_filter"], '{"min_return_20d": 0.0}')

    def test_rebuild_summary_from_disk_reads_extended_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "demo_run"
            exp_dir = run_dir / "experiments" / "demo_exp"
            exp_dir.mkdir(parents=True)

            write_json(exp_dir / "status.json", {"status": "completed"})
            write_json(
                exp_dir / "config.json",
                {
                    "name": "demo_exp",
                    "universe_name": "zz500",
                    "top_n": 20,
                    "benchmark_code": "000300.SH",
                    "factor_config": {"turnover_mean_20d": 0.7, "return_60d": 0.3},
                },
            )
            write_json(
                exp_dir / "metrics.json",
                {
                    "cumulative_return": 0.12,
                    "sharpe": 0.4,
                    "excess_cumulative_return": 0.09,
                    "positive_excess_month_ratio": 0.55,
                    "mean_rebalance_turnover": 1.1,
                    "worst_rolling_5m_sharpe": -1.3,
                },
            )

            summary = rebuild_summary_from_disk(run_dir)
            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(row["universe_name"], "zz500")
            self.assertEqual(row["positive_excess_month_ratio"], 0.55)
            self.assertEqual(row["mean_rebalance_turnover"], 1.1)
            self.assertEqual(row["worst_rolling_5m_sharpe"], -1.3)


if __name__ == "__main__":
    unittest.main()
