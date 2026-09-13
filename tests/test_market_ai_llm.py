from __future__ import annotations

import unittest

from market_ai.ai.provider import _coerce_news_event, _extract_json_object, analyze_news_items_with_llm
from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.themes import ThemeDefinition, ThemeTaxonomy


class FakeLLMProvider:
    def __init__(self, event: NewsEventAnalysis | None = None, should_fail: bool = False) -> None:
        self.event = event
        self.should_fail = should_fail

    def analyze_news_item(self, news: NewsItem, taxonomy: ThemeTaxonomy) -> NewsEventAnalysis | None:
        if self.should_fail:
            raise RuntimeError("llm failed")
        return self.event


class MarketAiLLMTestCase(unittest.TestCase):
    def test_extract_json_object_handles_think_and_code_block(self) -> None:
        payload = _extract_json_object(
            '<think>推理内容</think>\n```json\n{"themes": ["AI算力"], "importance": 4}\n```'
        )

        self.assertEqual(payload["themes"], ["AI算力"])
        self.assertEqual(payload["importance"], 4)

    def test_coerce_news_event_keeps_only_known_themes(self) -> None:
        taxonomy = ThemeTaxonomy([ThemeDefinition(theme="AI算力", aliases=("算力",))])
        news = NewsItem(news_id="n1", source="local", title="算力政策", published_at="2026-09-11T16:00:00+08:00")

        event = _coerce_news_event(
            {
                "event": "算力政策",
                "event_type": "policy",
                "event_time": news.published_at,
                "themes": ["AI算力", "未知题材"],
                "importance": 8,
                "novelty": 4,
                "confidence": 1.2,
            },
            news,
            taxonomy,
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.themes, ["AI算力"])
        self.assertEqual(event.importance, 5)
        self.assertEqual(event.confidence, 1.0)
        self.assertEqual(event.source_news_ids, ["n1"])

    def test_analyze_news_items_falls_back_to_rule_when_llm_fails(self) -> None:
        taxonomy = ThemeTaxonomy([ThemeDefinition(theme="AI算力", aliases=("算力",))])
        news = NewsItem(
            news_id="n1",
            source="local",
            title="算力产业链多股涨停",
            published_at="2026-09-11T16:00:00+08:00",
        )

        events = analyze_news_items_with_llm([news], taxonomy, provider=FakeLLMProvider(should_fail=True))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].themes, ["AI算力"])
        self.assertLessEqual(events[0].confidence, 0.8)

    def test_analyze_news_items_uses_llm_event_when_valid(self) -> None:
        taxonomy = ThemeTaxonomy([ThemeDefinition(theme="AI算力", aliases=("算力",))])
        news = NewsItem(news_id="n1", source="local", title="数据中心订单", published_at="2026-09-11T16:00:00+08:00")
        llm_event = NewsEventAnalysis(
            event="数据中心订单",
            event_type="order",
            event_time=news.published_at,
            themes=["AI算力"],
            importance=4,
            novelty=3,
            confidence=0.88,
            source_news_ids=[news.news_id],
        )

        events = analyze_news_items_with_llm([news], taxonomy, provider=FakeLLMProvider(event=llm_event))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "order")
        self.assertEqual(events[0].importance, 4)
        self.assertEqual(events[0].confidence, 0.88)


if __name__ == "__main__":
    unittest.main()
