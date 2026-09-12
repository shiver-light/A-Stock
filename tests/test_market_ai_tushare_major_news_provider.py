from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from market_ai.providers.news import TushareMajorNewsProvider


CN_TZ = timezone(timedelta(hours=8))


class FakeMajorNewsClient:
    def __init__(self) -> None:
        self.calls = 0

    def major_news(self, *, src=None, start_date=None, end_date=None, fields=None) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame(
            {
                "pub_time": ["2026-09-11 15:20:00", "2026-09-10 08:30:00", "bad-time"],
                "title": ["PCB板块走强", "机器人订单增长", "无效时间"],
                "content": ["AI服务器和液冷方向活跃", "减速器方向受关注", "无效"],
                "src": ["财联社", "同花顺", "新浪财经"],
            }
        )


class TushareMajorNewsProviderTestCase(unittest.TestCase):
    def test_fetch_news_frame_normalizes_fields_and_filters_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = TushareMajorNewsProvider(
                client=FakeMajorNewsClient(),
                cache_dir=Path(directory),
                slice_hours=48,
            )

            data = provider.fetch_news_frame(
                start_time=datetime(2026, 9, 10, 4, 30, tzinfo=CN_TZ),
                end_time=datetime(2026, 9, 11, 16, 30, tzinfo=CN_TZ),
            )

        self.assertEqual(len(data), 2)
        self.assertEqual(list(data.columns), ["news_id", "source", "title", "published_at", "url", "content"])
        self.assertEqual(set(data["source"]), {"财联社", "同花顺"})
        self.assertTrue(data["news_id"].str.startswith("tushare_major_").all())
        self.assertTrue(data["published_at"].str.endswith("+08:00").all())

    def test_fetch_news_uses_cache_without_refresh(self) -> None:
        client = FakeMajorNewsClient()
        with tempfile.TemporaryDirectory() as directory:
            provider = TushareMajorNewsProvider(client=client, cache_dir=Path(directory), slice_hours=48)
            kwargs = {
                "start_time": datetime(2026, 9, 10, 4, 30, tzinfo=CN_TZ),
                "end_time": datetime(2026, 9, 11, 16, 30, tzinfo=CN_TZ),
            }

            first = provider.fetch_news_frame(**kwargs)
            second = provider.fetch_news_frame(**kwargs)

        self.assertEqual(client.calls, 1)
        self.assertEqual(len(first), len(second))


if __name__ == "__main__":
    unittest.main()
