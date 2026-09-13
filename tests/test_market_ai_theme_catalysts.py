from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, ThemeScoreResult
from market_ai.scoring import aggregate_theme_catalysts


class MarketAiThemeCatalystsTestCase(unittest.TestCase):
    def test_aggregate_theme_catalysts_prefers_confirmed_event(self) -> None:
        themes = [ThemeScoreResult(trade_date="20260911", theme="PCB服务器液冷电源", score=80.0, rank=1)]
        events = [
            NewsEventAnalysis(
                event="相关但未确认消息",
                event_type="news",
                event_time="2026-09-11T09:00:00+08:00",
                themes=["PCB服务器液冷电源"],
                validation_state="unconfirmed_catalyst",
                market_confirm_score=20.0,
            ),
            NewsEventAnalysis(
                event="A股PCB板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["PCB服务器液冷电源"],
                validation_state="confirmed_catalyst",
                market_confirm_score=80.0,
            ),
        ]

        result = aggregate_theme_catalysts(trade_date="20260911", core_themes=themes, news_events=events)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].primary_event, "A股PCB板块多股涨停")
        self.assertEqual(result[0].confirmed_event_count, 1)
        self.assertEqual(result[0].unconfirmed_event_count, 1)

    def test_aggregate_theme_catalysts_keeps_theme_without_news(self) -> None:
        themes = [ThemeScoreResult(trade_date="20260911", theme="AI算力", score=60.0, rank=1)]

        result = aggregate_theme_catalysts(trade_date="20260911", core_themes=themes, news_events=[])

        self.assertEqual(result[0].theme, "AI算力")
        self.assertEqual(result[0].related_event_count, 0)
        self.assertIn("暂无核心消息", result[0].conclusion)


if __name__ == "__main__":
    unittest.main()
