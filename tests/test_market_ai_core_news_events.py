from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.scoring import rank_core_news_events, select_core_news_events


class MarketAiCoreNewsEventsTestCase(unittest.TestCase):
    def test_select_core_news_events_prefers_authoritative_core_theme_news(self) -> None:
        news_items = [
            NewsItem(news_id="n1", source="新华社", title="算力政策发布", published_at="2026-09-11T15:00:00+08:00"),
            NewsItem(news_id="n2", source="未知来源", title="机器人消息", published_at="2026-09-11T15:01:00+08:00"),
        ]
        events = [
            NewsEventAnalysis(
                event="算力政策发布",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["AI算力"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="机器人消息",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["机器人具身智能"],
                source_news_ids=["n2"],
            ),
        ]

        selected = select_core_news_events(
            events,
            news_items=news_items,
            core_themes=["AI算力"],
            authority_scores={"新华社": 95.0},
            default_authority_score=40.0,
            min_score=60.0,
            top_n=10,
        )

        self.assertEqual([event.event for event in selected], ["算力政策发布"])

    def test_rank_core_news_events_can_keep_non_core_when_configured(self) -> None:
        news_items = [
            NewsItem(news_id="n1", source="财联社", title="非核心题材", published_at="2026-09-11T15:00:00+08:00")
        ]
        events = [
            NewsEventAnalysis(
                event="非核心题材",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["低空经济"],
                source_news_ids=["n1"],
            )
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["AI算力"],
            authority_scores={"财联社": 80.0},
            require_core_theme_match=False,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].core_theme_score, 0.0)


if __name__ == "__main__":
    unittest.main()
