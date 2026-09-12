from __future__ import annotations

import unittest

from market_ai.models import LimitStock, NewsItem, StrongStock
from market_ai.themes import (
    RuleBasedThemeNormalizer,
    ThemeDefinition,
    ThemeTaxonomy,
    normalize_limit_stocks,
    normalize_news_items,
)


class ThemeNormalizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = ThemeTaxonomy(
            [
                ThemeDefinition(theme="AI算力", aliases=("AIDC", "AI服务器", "液冷服务器"), parent="科技"),
                ThemeDefinition(theme="CPO光通信", aliases=("CPO", "光模块"), parent="科技"),
            ]
        )
        self.normalizer = RuleBasedThemeNormalizer(self.taxonomy)

    def test_normalize_limit_stock_uses_reason_and_concepts(self) -> None:
        stock = LimitStock(
            trade_date="20260912",
            stock_code="000001.SZ",
            stock_name="样本股份",
            limit_reason="AI服务器订单增长",
            concepts=["CPO"],
            source="fixture",
        )

        result = self.normalizer.normalize_limit_stock(stock)

        self.assertIsNotNone(result)
        self.assertEqual(result.stock_code, "000001.SZ")
        self.assertEqual(result.primary_theme, "AI算力")
        self.assertEqual(result.secondary_themes, ["CPO光通信"])
        self.assertTrue(result.event_driven)
        self.assertEqual(result.source, "fixture")

    def test_normalize_strong_stock_drops_unmatched_text(self) -> None:
        stock = StrongStock(
            trade_date="20260912",
            stock_code="000002.SZ",
            stock_name="样本地产",
            pct_chg=8.0,
            industry="房地产",
        )

        result = self.normalizer.normalize_strong_stock(stock)

        self.assertIsNone(result)

    def test_normalize_news_item_outputs_structured_event(self) -> None:
        news = NewsItem(
            news_id="n1",
            source="local",
            title="液冷服务器需求提升",
            published_at="2026-09-12T16:30:00+08:00",
            content="CPO 光模块订单同步增长。",
        )

        result = self.normalizer.normalize_news_item(news)

        self.assertIsNotNone(result)
        self.assertEqual(result.event, "液冷服务器需求提升")
        self.assertEqual(result.event_type, "news")
        self.assertEqual(result.themes, ["AI算力", "CPO光通信"])
        self.assertEqual(result.source_news_ids, ["n1"])

    def test_batch_helpers_drop_unmatched_rows(self) -> None:
        matched = LimitStock(
            trade_date="20260912",
            stock_code="000001.SZ",
            stock_name="样本股份",
            limit_reason="AIDC",
        )
        unmatched = LimitStock(
            trade_date="20260912",
            stock_code="000002.SZ",
            stock_name="样本地产",
            limit_reason="地产",
        )
        news = NewsItem(
            news_id="n1",
            source="local",
            title="CPO 订单增长",
            published_at="2026-09-12T16:30:00+08:00",
        )

        stock_results = normalize_limit_stocks([matched, unmatched], self.taxonomy)
        news_results = normalize_news_items([news], self.taxonomy)

        self.assertEqual([item.stock_code for item in stock_results], ["000001.SZ"])
        self.assertEqual(news_results[0].themes, ["CPO光通信"])


if __name__ == "__main__":
    unittest.main()
