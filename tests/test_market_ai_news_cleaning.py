from __future__ import annotations

import unittest

from market_ai.models import RawNews
from market_ai.providers.news.cleaning import clean_news_item, clean_news_items, normalize_news_text


class MarketAiNewsCleaningTestCase(unittest.TestCase):
    def test_clean_news_item_keeps_policy_news(self) -> None:
        item = RawNews(
            news_id="n1",
            source="财联社",
            title="工信部推动算力基础设施建设",
            published_at="2026-09-11T15:20:00+08:00",
            content="政策提出加快智算中心建设。",
        )

        cleaned = clean_news_item(item)

        self.assertFalse(cleaned.is_filtered)
        self.assertEqual(cleaned.filter_reason, "")
        self.assertTrue(cleaned.content_hash)

    def test_clean_news_item_filters_pure_quote_broadcast(self) -> None:
        item = RawNews(
            news_id="n1",
            source="财联社",
            title="某股涨超5%",
            published_at="2026-09-11T15:20:00+08:00",
        )

        cleaned = clean_news_item(item)

        self.assertTrue(cleaned.is_filtered)
        self.assertEqual(cleaned.filter_reason, "quote_broadcast")

    def test_clean_news_items_does_not_drop_filtered_rows(self) -> None:
        items = [
            RawNews(news_id="n1", source="财联社", title="某股涨超5%", published_at="2026-09-11T15:20:00+08:00"),
            RawNews(
                news_id="n2",
                source="新华社",
                title="政策支持机器人产业发展",
                published_at="2026-09-11T15:30:00+08:00",
            ),
        ]

        cleaned = clean_news_items(items)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual([item.is_filtered for item in cleaned], [True, False])

    def test_normalize_news_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_news_text("  PCB\u3000 板块\n走强  "), "PCB 板块 走强")


if __name__ == "__main__":
    unittest.main()
