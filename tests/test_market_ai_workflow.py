from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path

import pandas as pd

from market_ai.config import NewsScoringConfig, RadarConfig
from market_ai.models import LimitStock, NewsItem, StrongStock
from market_ai.providers.market import MarketProvider
from market_ai.providers.news import NewsProvider
from market_ai.themes import ThemeDefinition, ThemeTaxonomy
from market_ai.workflow import build_daily_radar_report


CN_TZ = timezone(timedelta(hours=8))


class FakeMarketProvider(MarketProvider):
    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        return [
            LimitStock(
                trade_date=trade_date,
                stock_code="000001.SZ",
                stock_name="样本一",
                pct_chg=10.0,
                amount=1000.0,
                limit_reason="AI服务器",
                consecutive_limit_count=2,
            )
        ]

    def get_strong_stocks(self, *, trade_date: str, min_pct_chg: float = 7.0) -> list[StrongStock]:
        return [
            StrongStock(
                trade_date=trade_date,
                stock_code="000002.SZ",
                stock_name="样本二",
                pct_chg=8.0,
                amount=500.0,
                concepts=["光模块"],
            )
        ]

    def get_daily_quotes(self, *, trade_date: str, stock_codes=None):
        raise NotImplementedError


class FakeNewsProvider(NewsProvider):
    def fetch_news(self, *, start_time: datetime, end_time: datetime, sources=None) -> list[NewsItem]:
        return [
            NewsItem(
                news_id="n1",
                source="local",
                title="A股AI服务器板块政策催化",
                published_at=end_time.isoformat(),
                content="政策推动算力产业链订单增长，多股涨停。",
            )
        ]


class MarketAiWorkflowTestCase(unittest.TestCase):
    def test_build_daily_radar_report(self) -> None:
        taxonomy = ThemeTaxonomy(
            [
                ThemeDefinition(theme="AI算力", aliases=("AI服务器",)),
                ThemeDefinition(theme="CPO光通信", aliases=("光模块",)),
            ]
        )

        report = build_daily_radar_report(
            trade_date="20260912",
            market_provider=FakeMarketProvider(),
            taxonomy=taxonomy,
            config=RadarConfig(news_scoring=NewsScoringConfig(authority_scores={"local": 80.0})),
            news_provider=FakeNewsProvider(),
            news_start_time=datetime(2026, 9, 12, 9, 0, tzinfo=CN_TZ),
            news_end_time=datetime(2026, 9, 12, 16, 30, tzinfo=CN_TZ),
        )

        self.assertEqual(report.trade_date, "20260912")
        self.assertGreaterEqual(len(report.core_themes), 1)
        self.assertEqual(report.news_events[0].themes, ["AI算力"])
        self.assertEqual(report.news_events[0].validation_state, "confirmed_catalyst")
        self.assertGreaterEqual(report.news_events[0].market_confirm_score or 0.0, 50.0)
        self.assertEqual(report.theme_catalysts[0].theme, "AI算力")
        self.assertEqual(report.metadata["theme_catalyst_count"], len(report.theme_catalysts))
        self.assertEqual(report.metadata["limit_stock_count"], 1)
        self.assertEqual(report.metadata["strong_stock_count"], 1)
        self.assertTrue(report.next_day_observations)

    def test_stock_theme_labels_feed_theme_scores(self) -> None:
        taxonomy = ThemeTaxonomy([ThemeDefinition(theme="AI算力", aliases=("AI服务器",))])
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            pd.DataFrame(
                {
                    "trade_date": ["20260912"],
                    "stock_code": ["000002.SZ"],
                    "primary_theme": ["CPO光通信"],
                    "confidence": [0.9],
                    "source": ["manual"],
                }
            ).to_csv(labels_path, index=False)

            report = build_daily_radar_report(
                trade_date="20260912",
                market_provider=FakeMarketProvider(),
                taxonomy=taxonomy,
                config=RadarConfig(),
                stock_theme_labels_path=str(labels_path),
            )

        themes = {theme.theme for theme in report.core_themes}
        self.assertIn("CPO光通信", themes)
        self.assertEqual(report.metadata["stock_theme_label_count"], 1)


if __name__ == "__main__":
    unittest.main()
