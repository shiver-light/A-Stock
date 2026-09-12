from __future__ import annotations

import unittest
from datetime import datetime, timezone

from market_ai.models import NewsItem
from market_ai.providers.news import NewsProvider


class FakeNewsProvider(NewsProvider):
    def fetch_news(self, *, start_time: datetime, end_time: datetime, sources=None) -> list[NewsItem]:
        allowed_sources = set(sources or ["fake"])
        if "fake" not in allowed_sources:
            return []
        return [
            NewsItem(
                news_id="fake-1",
                source="fake",
                title="工信部发布算力基础设施政策",
                published_at=start_time.isoformat(),
                content="政策支持算力基础设施建设。",
            )
        ]


class NewsProviderTestCase(unittest.TestCase):
    def test_news_provider_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            NewsProvider()

    def test_fake_provider_returns_normalized_news_items(self) -> None:
        provider = FakeNewsProvider()
        start_time = datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc)
        end_time = datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)

        items = provider.fetch_news(start_time=start_time, end_time=end_time)

        self.assertEqual(items[0].news_id, "fake-1")
        self.assertEqual(items[0].source, "fake")
        self.assertIn("算力", items[0].title)

    def test_fake_provider_respects_source_filter(self) -> None:
        provider = FakeNewsProvider()
        start_time = datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc)
        end_time = datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)

        items = provider.fetch_news(start_time=start_time, end_time=end_time, sources=["other"])

        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
