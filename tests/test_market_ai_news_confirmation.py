from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, ThemeScoreResult
from market_ai.scoring import CONFIRMED_CATALYST, UNCONFIRMED_CATALYST, classify_news_market_confirmation


class MarketAiNewsConfirmationTestCase(unittest.TestCase):
    def test_event_matching_strong_core_theme_is_confirmed(self) -> None:
        events = [
            NewsEventAnalysis(
                event="A股PCB板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["PCB服务器液冷电源"],
            )
        ]
        themes = [ThemeScoreResult(trade_date="20260911", theme="PCB服务器液冷电源", score=70.0)]

        result = classify_news_market_confirmation(events, core_themes=themes)

        self.assertEqual(result[0].validation_state, CONFIRMED_CATALYST)
        self.assertEqual(result[0].market_confirm_score, 70.0)
        self.assertIn("命中核心题材", result[0].validation_reason[0])

    def test_event_without_core_theme_match_is_unconfirmed(self) -> None:
        events = [
            NewsEventAnalysis(
                event="低空经济政策发布",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["低空经济"],
            )
        ]
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=80.0)]

        result = classify_news_market_confirmation(events, core_themes=themes)

        self.assertEqual(result[0].validation_state, UNCONFIRMED_CATALYST)
        self.assertEqual(result[0].market_confirm_score, 0.0)

    def test_event_matching_weak_core_theme_is_unconfirmed(self) -> None:
        events = [
            NewsEventAnalysis(
                event="AI算力消息",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["AI算力"],
            )
        ]
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=30.0)]

        result = classify_news_market_confirmation(events, core_themes=themes)

        self.assertEqual(result[0].validation_state, UNCONFIRMED_CATALYST)
        self.assertEqual(result[0].market_confirm_score, 30.0)


if __name__ == "__main__":
    unittest.main()
