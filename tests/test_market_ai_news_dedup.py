from __future__ import annotations

import unittest

from market_ai.models import NormalizedNews
from market_ai.providers.news.dedup import deduplicate_news_items


def news(
    news_id: str,
    *,
    source: str = "财联社",
    title: str = "PCB板块走强",
    published_at: str = "2026-09-11T15:20:00+08:00",
    normalized_title: str = "PCB板块走强",
    content_hash: str = "hash1",
    url: str = "",
    content: str = "",
) -> NormalizedNews:
    return NormalizedNews(
        news_id=news_id,
        source=source,
        title=title,
        published_at=published_at,
        normalized_title=normalized_title,
        content_hash=content_hash,
        url=url,
        content=content,
    )


class MarketAiNewsDedupTestCase(unittest.TestCase):
    def test_same_url_is_deduplicated_and_sources_are_kept(self) -> None:
        result = deduplicate_news_items(
            [
                news("n1", source="财联社", url="https://example.com/a"),
                news("n2", source="东方财富", url="https://example.com/a"),
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_news_ids, ["n1", "n2"])
        self.assertEqual(result[0].sources, ["财联社", "东方财富"])

    def test_same_content_hash_is_deduplicated_without_url(self) -> None:
        result = deduplicate_news_items(
            [
                news("n1", content_hash="abc", normalized_title="标题A"),
                news("n2", content_hash="abc", normalized_title="标题B"),
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_news_ids, ["n1", "n2"])

    def test_same_title_without_hash_is_deduplicated(self) -> None:
        result = deduplicate_news_items(
            [
                news("n1", content_hash="", normalized_title="同一标题"),
                news("n2", content_hash="", normalized_title="同一标题"),
            ]
        )

        self.assertEqual(len(result), 1)

    def test_different_news_are_kept(self) -> None:
        result = deduplicate_news_items(
            [
                news("n1", content_hash="abc", normalized_title="标题A"),
                news("n2", content_hash="def", normalized_title="标题B"),
            ]
        )

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
