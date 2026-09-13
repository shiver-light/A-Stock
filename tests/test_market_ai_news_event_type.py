from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.scoring import apply_news_event_types, classify_news_event_type, event_type_score


class MarketAiNewsEventTypeTestCase(unittest.TestCase):
    def test_classify_policy_event(self) -> None:
        event = NewsEventAnalysis(
            event="工信部发布算力产业链支持政策",
            event_type="news",
            event_time="2026-09-11T10:00:00+08:00",
            themes=["AI算力"],
        )

        result = classify_news_event_type(event)

        self.assertEqual(result.primary_type, "policy")
        self.assertIn("policy", result.tags)
        self.assertGreater(result.score, event_type_score("other"))

    def test_classify_price_hike_from_news_item_content(self) -> None:
        event = NewsEventAnalysis(
            event="电子布供需偏紧",
            event_type="news",
            event_time="2026-09-11T10:00:00+08:00",
            themes=["PCB"],
            source_news_ids=["n1"],
        )
        news_item = NewsItem(
            news_id="n1",
            source="财联社",
            title="电子布涨价落地",
            published_at="2026-09-11T10:00:00+08:00",
            content="头部企业报价上调。",
        )

        result = classify_news_event_type(event, news_item)

        self.assertEqual(result.primary_type, "price_hike")
        self.assertIn("price_hike", result.tags)

    def test_single_stock_event_is_low_quality_type(self) -> None:
        event = NewsEventAnalysis(
            event="某公司控制权变更收到问询函",
            event_type="news",
            event_time="2026-09-11T10:00:00+08:00",
            themes=["机器人具身智能"],
        )

        result = classify_news_event_type(event)

        self.assertEqual(result.primary_type, "single_stock")
        self.assertLess(result.score, event_type_score("other"))

    def test_apply_news_event_types_returns_cloned_events(self) -> None:
        event = NewsEventAnalysis(
            event="PCB订单增长",
            event_type="news",
            event_time="2026-09-11T10:00:00+08:00",
            themes=["PCB"],
            source_news_ids=["n1"],
        )
        news_item = NewsItem(news_id="n1", source="财联社", title="PCB订单增长", published_at=event.event_time)

        result = apply_news_event_types([event], news_items=[news_item])

        self.assertEqual(result[0].event_type, "order")
        self.assertEqual(event.event_type, "news")


if __name__ == "__main__":
    unittest.main()
