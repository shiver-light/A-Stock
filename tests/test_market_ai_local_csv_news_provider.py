from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_ai.providers.news import LocalCsvNewsProvider


CN_TZ = timezone(timedelta(hours=8))


class LocalCsvNewsProviderTestCase(unittest.TestCase):
    def test_missing_file_returns_empty_list(self) -> None:
        provider = LocalCsvNewsProvider("missing_news.csv")

        items = provider.fetch_news(
            start_time=datetime(2026, 9, 12, 9, 0, tzinfo=CN_TZ),
            end_time=datetime(2026, 9, 12, 16, 0, tzinfo=CN_TZ),
        )

        self.assertEqual(items, [])

    def test_fetch_news_filters_by_time_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "news.csv"
            pd.DataFrame(
                {
                    "news_id": ["n2", "n1", "n3", "n4", ""],
                    "source": ["cls", "cls", "eastmoney", "cls", "cls"],
                    "title": ["较晚新闻", "窗口内新闻", "其他来源", "窗口前新闻", "无效新闻"],
                    "published_at": [
                        "2026-09-12T11:00:00+08:00",
                        "2026-09-12T10:00:00+08:00",
                        "2026-09-12T10:30:00+08:00",
                        "2026-09-11T23:00:00+08:00",
                        "2026-09-12T10:00:00+08:00",
                    ],
                    "url": ["", "https://example.com/news/1", "", "", ""],
                    "content": ["较晚内容", "正文内容", "其他来源内容", "窗口前内容", "无效内容"],
                }
            ).to_csv(csv_path, index=False)
            provider = LocalCsvNewsProvider(csv_path)

            items = provider.fetch_news(
                start_time=datetime(2026, 9, 12, 9, 30, tzinfo=CN_TZ),
                end_time=datetime(2026, 9, 12, 11, 0, tzinfo=CN_TZ),
                sources=["cls"],
            )

            self.assertEqual([item.news_id for item in items], ["n1", "n2"])
            self.assertEqual(items[0].source, "cls")
            self.assertEqual(items[0].url, "https://example.com/news/1")
            self.assertEqual(items[1].url, None)

    def test_optional_columns_are_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "news.csv"
            pd.DataFrame(
                {
                    "news_id": ["n1"],
                    "source": ["policy"],
                    "title": ["政策新闻"],
                    "published_at": ["2026-09-12T10:00:00+08:00"],
                }
            ).to_csv(csv_path, index=False)
            provider = LocalCsvNewsProvider(csv_path)

            items = provider.fetch_news(
                start_time=datetime(2026, 9, 12, 9, 30, tzinfo=CN_TZ),
                end_time=datetime(2026, 9, 12, 10, 30, tzinfo=CN_TZ),
            )

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].url, None)
            self.assertEqual(items[0].content, None)

    def test_missing_required_columns_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "news.csv"
            pd.DataFrame({"news_id": ["n1"], "source": ["cls"]}).to_csv(csv_path, index=False)
            provider = LocalCsvNewsProvider(csv_path)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                provider.fetch_news(
                    start_time=datetime(2026, 9, 12, 9, 30, tzinfo=CN_TZ),
                    end_time=datetime(2026, 9, 12, 10, 30, tzinfo=CN_TZ),
                )


if __name__ == "__main__":
    unittest.main()
