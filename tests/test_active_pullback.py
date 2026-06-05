from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from research.active_pullback import (
    DEFAULT_ACTIVE_PULLBACK_FACTOR_CONFIG,
    _exclude_chinext_codes,
    build_active_pullback_signal_filters,
    generate_active_pullback_recommendations,
    render_active_pullback_text,
)


class ActivePullbackTestCase(unittest.TestCase):
    def test_build_active_pullback_signal_filters_uses_active_pool_rules(self) -> None:
        filters = build_active_pullback_signal_filters(
            return_quantile=0.65,
            turnover_quantile=0.70,
            amount_quantile=0.60,
            money_flow_quantile=0.55,
        )

        self.assertEqual(
            filters,
            [
                {"factor": "return_20d", "op": "quantile_gte", "value": 0.65},
                {"factor": "turnover_mean_20d", "op": "quantile_gte", "value": 0.70},
                {"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.60},
                {"factor": "money_flow_strength_20d", "op": "quantile_gte", "value": 0.55},
            ],
        )

    def test_build_active_pullback_signal_filters_rejects_invalid_quantile(self) -> None:
        with self.assertRaisesRegex(ValueError, "return_quantile"):
            build_active_pullback_signal_filters(return_quantile=1.2)

    def test_exclude_chinext_codes_removes_300_and_301_prefixes(self) -> None:
        result = _exclude_chinext_codes(["300001.SZ", "301001.SZ", "002001.SZ", "688001.SH", "600001.SH"])

        self.assertEqual(result, ["002001.SZ", "600001.SH", "688001.SH"])

    @patch("research.active_pullback.run_recommendation_pipeline")
    def test_generate_active_pullback_recommendations_reuses_recommendation_pipeline(self, mock_pipeline) -> None:
        mock_pipeline.return_value = {
            "latest_selection": {
                "top_stocks": [
                    {"ts_code": "000001.SZ", "score": 0.9, "rank": 1, "selected": True},
                ]
            }
        }

        result = generate_active_pullback_recommendations(
            as_of_date="20260430",
            start_date="20250101",
            universe_name="zz1000",
            top_n=30,
            benchmark_code="000852.SH",
            amount_quantile=0.60,
        )

        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["start_date"], "20250101")
        self.assertEqual(kwargs["end_date"], "20260430")
        self.assertEqual(kwargs["universe_name"], "zz1000")
        self.assertEqual(kwargs["top_n"], 30)
        self.assertEqual(kwargs["benchmark_code"], "000852.SH")
        self.assertEqual(kwargs["factor_config"], DEFAULT_ACTIVE_PULLBACK_FACTOR_CONFIG)
        self.assertIn({"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.60}, kwargs["signal_filters"])
        self.assertEqual(result["top_stocks"][0]["ts_code"], "000001.SZ")
        self.assertEqual(result["research_experiment"]["signal_filters"], kwargs["signal_filters"])

    @patch("research.active_pullback.get_universe")
    @patch("research.active_pullback.run_recommendation_pipeline")
    def test_generate_active_pullback_recommendations_can_exclude_chinext(self, mock_pipeline, mock_get_universe) -> None:
        mock_get_universe.return_value = pd.DataFrame(
            {
                "as_of_date": ["20260430"] * 4,
                "ts_code": ["300001.SZ", "301001.SZ", "002001.SZ", "688001.SH"],
                "universe_name": ["zz1000"] * 4,
                "in_universe": [True] * 4,
            }
        )
        mock_pipeline.return_value = {"latest_selection": {"top_stocks": []}}

        result = generate_active_pullback_recommendations(
            as_of_date="20260430",
            start_date="20250101",
            universe_name="zz1000",
            exclude_chinext=True,
        )

        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["ts_codes"], ["002001.SZ", "688001.SH"])
        self.assertTrue(result["exclude_chinext"])
        self.assertEqual(result["ts_code_count"], 2)
        self.assertEqual(result["research_experiment"]["ts_code_count"], 2)

    @patch("research.active_pullback.run_recommendation_pipeline")
    def test_generate_active_pullback_recommendations_defaults_start_date(self, mock_pipeline) -> None:
        mock_pipeline.return_value = {"latest_selection": {"top_stocks": []}}

        generate_active_pullback_recommendations(as_of_date="20260430")

        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["start_date"], "20250902")

    def test_render_active_pullback_text_outputs_selected_stocks(self) -> None:
        text = render_active_pullback_text(
            {
                "universe_name": "zz1000",
                "benchmark_code": "000852.SH",
                "as_of_date": "20260430",
                "start_date": "20250101",
                "top_n": 30,
                "signal_filters": [{"factor": "return_20d", "op": "quantile_gte", "value": 0.7}],
                "factor_config": {"pullback_after_trend_60d": 1.0},
                "top_stocks": [
                    {"ts_code": "000001.SZ", "score": 0.9, "rank": 1, "selected": True},
                    {"ts_code": "000002.SZ", "score": 0.8, "rank": 2, "selected": False},
                ],
                "research_experiment": {"name": "active_pullback_zz1000_top30"},
            }
        )

        self.assertIn("active pullback candidates:", text)
        self.assertIn("return_20d quantile_gte 0.7", text)
        self.assertIn("000001.SZ", text)
        self.assertNotIn("000002.SZ", text)


if __name__ == "__main__":
    unittest.main()
