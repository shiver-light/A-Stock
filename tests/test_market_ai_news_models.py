from __future__ import annotations

import unittest

from market_ai.models import EventCluster, MarketEvent, NormalizedNews, RawNews


class MarketAiNewsModelsTestCase(unittest.TestCase):
    def test_raw_news_validates_required_fields_and_time(self) -> None:
        item = RawNews(
            news_id="n1",
            source="财联社",
            title="PCB板块走强",
            published_at="2026-09-11T15:20:00+08:00",
        )

        self.assertEqual(item.source, "财联社")

        with self.assertRaisesRegex(ValueError, "published_at must be an ISO datetime"):
            RawNews(news_id="n1", source="财联社", title="PCB板块走强", published_at="20260911")

    def test_normalized_news_defaults_audit_fields(self) -> None:
        item = NormalizedNews(
            news_id="n1",
            source="财联社",
            title="PCB板块走强",
            published_at="2026-09-11T15:20:00+08:00",
            normalized_title="pcb板块走强",
            content_hash="abc",
        )

        self.assertEqual(item.source_news_ids, ["n1"])
        self.assertEqual(item.sources, ["财联社"])
        self.assertFalse(item.is_filtered)

    def test_event_cluster_requires_source_news_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_news_ids"):
            EventCluster(
                cluster_id="c1",
                event_time="2026-09-11T15:20:00+08:00",
                title="PCB板块走强",
                source_news_ids=[],
            )

    def test_market_event_validates_confidence(self) -> None:
        event = MarketEvent(
            event_id="e1",
            event_time="2026-09-11T15:20:00+08:00",
            event_summary="PCB订单催化",
            event_type="industry_news",
            themes=["PCB服务器液冷电源"],
            confidence=0.8,
        )

        self.assertEqual(event.themes, ["PCB服务器液冷电源"])
        with self.assertRaisesRegex(ValueError, "confidence"):
            MarketEvent(
                event_id="e2",
                event_time="2026-09-11T15:20:00+08:00",
                event_summary="PCB订单催化",
                event_type="industry_news",
                confidence=1.5,
            )


if __name__ == "__main__":
    unittest.main()
