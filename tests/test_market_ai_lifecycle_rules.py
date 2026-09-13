from __future__ import annotations

import unittest

import pandas as pd

from market_ai.lifecycle import classify_theme_lifecycle
from market_ai.models import ThemeScoreResult


class ThemeLifecycleRulesTestCase(unittest.TestCase):
    def test_start_stage_for_first_active_day(self) -> None:
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=55.0, rank=1)]

        results = classify_theme_lifecycle(themes, pd.DataFrame(), trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 1)
        self.assertTrue(any("启动" in reason for reason in results[0].reasons))

    def test_main_rise_stage_for_persistent_high_score(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260908", "20260909", "20260910"],
                "theme": ["AI算力", "AI算力", "AI算力"],
                "score": [50.0, 60.0, 70.0],
                "rank": [3, 2, 2],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=75.0, rank=2)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 3)

    def test_divergence_stage_for_falling_but_active_theme(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260908", "20260909", "20260910"],
                "theme": ["AI算力", "AI算力", "AI算力"],
                "score": [80.0, 70.0, 60.0],
                "rank": [1, 1, 2],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=45.0, rank=4)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 5)

    def test_climax_stage_requires_persistent_high_average(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260907", "20260908", "20260909", "20260910"],
                "theme": ["AI算力", "AI算力", "AI算力", "AI算力"],
                "score": [70.0, 76.0, 82.0, 88.0],
                "rank": [2, 2, 1, 1],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=90.0, rank=1)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 4)
        self.assertTrue(any("高潮" in reason for reason in results[0].reasons))

    def test_return_stage_for_rebound_after_previous_peak(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260907", "20260908", "20260909", "20260910"],
                "theme": ["AI算力", "AI算力", "AI算力", "AI算力"],
                "score": [70.0, 60.0, 30.0, 35.0],
                "rank": [1, 2, 6, 5],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=50.0, rank=3)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 6)
        self.assertTrue(any("回流" in reason for reason in results[0].reasons))

    def test_retreat_stage_after_previous_peak_and_cooling(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260907", "20260908", "20260909", "20260910"],
                "theme": ["AI算力", "AI算力", "AI算力", "AI算力"],
                "score": [70.0, 60.0, 25.0, 20.0],
                "rank": [1, 2, 8, 9],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=25.0, rank=9)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 7)
        self.assertTrue(any("退潮" in reason for reason in results[0].reasons))

    def test_future_history_is_ignored(self) -> None:
        history = pd.DataFrame(
            {
                "trade_date": ["20260912"],
                "theme": ["AI算力"],
                "score": [90.0],
                "rank": [1],
            }
        )
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=55.0, rank=1)]

        results = classify_theme_lifecycle(themes, history, trade_date="20260911")

        self.assertEqual(results[0].lifecycle_stage, 1)


if __name__ == "__main__":
    unittest.main()
