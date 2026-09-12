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
