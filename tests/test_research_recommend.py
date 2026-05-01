from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.recommend import (
    build_daily_consensus_output,
    generate_daily_consensus_recommendations,
    generate_daily_recommendations_from_run,
    render_consensus_recommendation_text,
    render_recommendation_text,
    select_recommendation_models,
)


class ResearchRecommendTestCase(unittest.TestCase):
    def _write_experiment(
        self,
        base_dir: Path,
        *,
        name: str,
        config: dict[str, object],
        metrics: dict[str, object],
        status: str = "completed",
    ) -> None:
        experiment_dir = base_dir / "experiments" / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        (experiment_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (experiment_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def test_select_recommendation_models_filters_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "first_batch"
            (run_dir / "experiments").mkdir(parents=True, exist_ok=True)

            shared_config = {
                "start_date": "20240101",
                "end_date": "20240131",
                "universe_name": "hs300",
                "top_n": 10,
                "factor_config": {"turnover_mean_20d": 1.0},
            }
            self._write_experiment(
                run_dir,
                name="exp_a",
                config=shared_config,
                metrics={
                    "sharpe": 0.5,
                    "excess_cumulative_return": 0.1,
                    "max_drawdown": -0.2,
                    "positive_excess_month_ratio": 0.6,
                },
            )
            self._write_experiment(
                run_dir,
                name="exp_b_duplicate",
                config=shared_config,
                metrics={
                    "sharpe": 0.4,
                    "excess_cumulative_return": 0.08,
                    "max_drawdown": -0.2,
                    "positive_excess_month_ratio": 0.55,
                },
            )
            self._write_experiment(
                run_dir,
                name="exp_c_bad",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "zz500",
                    "top_n": 20,
                    "factor_config": {"return_20d": 1.0},
                },
                metrics={
                    "sharpe": -0.1,
                    "excess_cumulative_return": -0.05,
                    "max_drawdown": -0.4,
                    "positive_excess_month_ratio": 0.3,
                },
            )

            selected = select_recommendation_models(run_dir, top_k_models=5)

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["name"], "exp_a")

    @patch("research.recommend.run_recommendation_pipeline")
    def test_generate_daily_recommendations_from_run_builds_consensus(self, mock_pipeline) -> None:
        mock_pipeline.side_effect = [
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.9, "rank": 1, "selected": True},
                        {"ts_code": "000002.SZ", "score": 0.8, "rank": 2, "selected": True},
                    ]
                }
            },
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.85, "rank": 2, "selected": True},
                        {"ts_code": "000003.SZ", "score": 0.82, "rank": 1, "selected": True},
                    ]
                }
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "first_batch"
            (run_dir / "experiments").mkdir(parents=True, exist_ok=True)

            self._write_experiment(
                run_dir,
                name="model_a",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "hs300",
                    "top_n": 10,
                    "factor_config": {"turnover_mean_20d": 1.0},
                },
                metrics={
                    "sharpe": 0.6,
                    "excess_cumulative_return": 0.12,
                    "max_drawdown": -0.2,
                    "positive_excess_month_ratio": 0.6,
                },
            )
            self._write_experiment(
                run_dir,
                name="model_b",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "zz500",
                    "top_n": 20,
                    "factor_config": {"turnover_mean_20d": 1.0, "volatility_20d": -0.25},
                },
                metrics={
                    "sharpe": 0.4,
                    "excess_cumulative_return": 0.08,
                    "max_drawdown": -0.25,
                    "positive_excess_month_ratio": 0.55,
                },
            )

            result = generate_daily_recommendations_from_run(
                run_dir,
                as_of_date="20240228",
                top_k_models=5,
                top_k_stocks=10,
            )

            self.assertEqual(len(result["selected_models"]), 2)
            self.assertEqual(result["recommendations"][0]["ts_code"], "000001.SZ")
            self.assertEqual(result["recommendations"][0]["consensus_count"], 2)
            self.assertIn("model_a", result["recommendations"][0]["source_models"])
            self.assertIn("model_b", result["recommendations"][0]["source_models"])

            report_text = render_recommendation_text(result)
            self.assertIn("selected models:", report_text)
            self.assertIn("recommendations:", report_text)
            self.assertIn("000001.SZ", report_text)

    def test_build_daily_consensus_output_assigns_abc_levels(self) -> None:
        buckets = build_daily_consensus_output(
            [
                {
                    "name": "c01_hs300_turnover_top10",
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.95, "rank": 1, "selected": True},
                        {"ts_code": "000002.SZ", "score": 0.90, "rank": 2, "selected": True},
                    ],
                },
                {
                    "name": "c03_hs300_turnover_ret60_70_30_top20",
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.85, "rank": 3, "selected": True},
                        {"ts_code": "000003.SZ", "score": 0.80, "rank": 4, "selected": True},
                    ],
                },
                {
                    "name": "c04_zz500_turnover_top10",
                    "top_stocks": [
                        {"ts_code": "000004.SZ", "score": 0.88, "rank": 1, "selected": True},
                    ],
                },
            ],
            core_model_names=["c01_hs300_turnover_top10"],
            confirm_model_names=["c03_hs300_turnover_ret60_70_30_top20"],
            watch_model_names=["c04_zz500_turnover_top10"],
        )

        self.assertEqual(buckets["trade_consensus"][0]["ts_code"], "000001.SZ")
        self.assertEqual(buckets["trade_consensus"][0]["recommend_level"], "A")
        self.assertEqual(buckets["trade_core"][1]["ts_code"], "000002.SZ")
        self.assertEqual(buckets["trade_core"][1]["recommend_level"], "B")
        self.assertEqual(buckets["watch_list"][0]["ts_code"], "000003.SZ")
        self.assertEqual(buckets["watch_list"][0]["recommend_level"], "C")

    @patch("research.recommend.run_recommendation_pipeline")
    def test_generate_daily_consensus_recommendations_uses_named_model_roles(self, mock_pipeline) -> None:
        mock_pipeline.side_effect = [
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.95, "rank": 1, "selected": True},
                        {"ts_code": "000002.SZ", "score": 0.90, "rank": 2, "selected": True},
                    ]
                }
            },
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.87, "rank": 2, "selected": True},
                        {"ts_code": "000003.SZ", "score": 0.84, "rank": 3, "selected": True},
                    ]
                }
            },
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000004.SZ", "score": 0.82, "rank": 1, "selected": True},
                    ]
                }
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "stage2_candidates"
            (run_dir / "experiments").mkdir(parents=True, exist_ok=True)

            self._write_experiment(
                run_dir,
                name="c01_hs300_turnover_top10",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "hs300",
                    "top_n": 10,
                    "factor_config": {"turnover_mean_20d": 1.0},
                },
                metrics={
                    "sharpe": 0.6,
                    "excess_cumulative_return": 0.12,
                    "max_drawdown": -0.2,
                    "positive_excess_month_ratio": 0.6,
                },
            )
            self._write_experiment(
                run_dir,
                name="c03_hs300_turnover_ret60_70_30_top20",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "hs300",
                    "top_n": 20,
                    "factor_config": {"turnover_mean_20d": 0.7, "return_60d": 0.3},
                },
                metrics={
                    "sharpe": 0.4,
                    "excess_cumulative_return": 0.08,
                    "max_drawdown": -0.25,
                    "positive_excess_month_ratio": 0.55,
                },
            )
            self._write_experiment(
                run_dir,
                name="c04_zz500_turnover_top10",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "zz500",
                    "top_n": 10,
                    "factor_config": {"turnover_mean_20d": 1.0},
                },
                metrics={
                    "sharpe": 0.3,
                    "excess_cumulative_return": 0.05,
                    "max_drawdown": -0.28,
                    "positive_excess_month_ratio": 0.52,
                },
            )

            result = generate_daily_consensus_recommendations(
                run_dir,
                as_of_date="20240228",
                watch_model_names=["c04_zz500_turnover_top10"],
            )

            self.assertEqual(result["trade_consensus"][0]["ts_code"], "000001.SZ")
            self.assertEqual(result["trade_consensus"][0]["recommend_level"], "A")
            self.assertEqual(result["trade_core"][1]["ts_code"], "000002.SZ")
            self.assertEqual(result["watch_list"][0]["ts_code"], "000003.SZ")
            self.assertEqual(result["watch_list"][1]["ts_code"], "000004.SZ")

            report_text = render_consensus_recommendation_text(result)
            self.assertIn("trade consensus:", report_text)
            self.assertIn("trade core:", report_text)
            self.assertIn("watch list:", report_text)
            self.assertIn("000001.SZ", report_text)

    @patch("research.recommend.run_recommendation_pipeline")
    def test_generate_daily_consensus_recommendations_allows_empty_watch_models(self, mock_pipeline) -> None:
        mock_pipeline.side_effect = [
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.95, "rank": 1, "selected": True},
                    ]
                }
            },
            {
                "latest_selection": {
                    "top_stocks": [
                        {"ts_code": "000001.SZ", "score": 0.87, "rank": 2, "selected": True},
                        {"ts_code": "000003.SZ", "score": 0.84, "rank": 3, "selected": True},
                    ]
                }
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "zz500_stage2"
            (run_dir / "experiments").mkdir(parents=True, exist_ok=True)

            self._write_experiment(
                run_dir,
                name="s2_m01_zz500_turnover_top10",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "zz500",
                    "top_n": 10,
                    "factor_config": {"turnover_mean_20d": 1.0},
                },
                metrics={
                    "sharpe": 0.3,
                    "excess_cumulative_return": 0.02,
                    "max_drawdown": -0.15,
                    "positive_excess_month_ratio": 0.6,
                },
            )
            self._write_experiment(
                run_dir,
                name="s2_m04_zz500_ep_ttm_top10",
                config={
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "universe_name": "zz500",
                    "top_n": 10,
                    "factor_config": {"ep_ttm": 1.0},
                },
                metrics={
                    "sharpe": 1.2,
                    "excess_cumulative_return": 1.0,
                    "max_drawdown": -0.25,
                    "positive_excess_month_ratio": 0.64,
                },
            )

            result = generate_daily_consensus_recommendations(
                run_dir,
                as_of_date="20240228",
                core_model_names=["s2_m01_zz500_turnover_top10"],
                confirm_model_names=["s2_m04_zz500_ep_ttm_top10"],
                watch_model_names=[],
            )

            self.assertEqual(result["model_roles"]["watch_models"], [])
            self.assertEqual(result["trade_consensus"][0]["ts_code"], "000001.SZ")
            self.assertEqual(result["watch_list"][0]["ts_code"], "000003.SZ")


if __name__ == "__main__":
    unittest.main()
