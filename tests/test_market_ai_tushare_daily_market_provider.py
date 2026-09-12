from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_ai.providers.market import TushareDailyMarketProvider


class FakeTushareClient:
    def __init__(self) -> None:
        self.daily_calls = 0

    def daily(self, *, trade_date=None, fields=None, **kwargs) -> pd.DataFrame:
        self.daily_calls += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "trade_date": [trade_date, trade_date, trade_date],
                "open": [10.0, 20.0, 30.0],
                "high": [11.0, 21.0, 31.0],
                "low": [9.8, 19.8, 29.8],
                "close": [11.0, 21.6, 32.94],
                "pre_close": [10.0, 20.0, 30.0],
                "pct_chg": [10.0, 8.0, 9.8],
                "vol": [1000.0, 2000.0, 3000.0],
                "amount": [100000.0, 200000.0, 300000.0],
            }
        )

    def daily_basic(self, *, trade_date=None, fields=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "trade_date": [trade_date, trade_date, trade_date],
                "turnover_rate": [3.0, 4.0, 5.0],
                "volume_ratio": [1.2, 1.5, 2.0],
            }
        )

    def stock_basic(self, *, list_status=None, fields=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["样本一", "样本二", "样本三"],
                "industry": ["银行", "AI算力", "光通信"],
            }
        )


class TushareDailyMarketProviderTestCase(unittest.TestCase):
    def test_get_limit_and_strong_stocks_from_daily_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = TushareDailyMarketProvider(
                client=FakeTushareClient(),
                cache_dir=Path(directory),
                limit_up_pct_threshold=9.8,
            )

            limit_stocks = provider.get_limit_stocks(trade_date="20260912")
            strong_stocks = provider.get_strong_stocks(trade_date="20260912", min_pct_chg=7.0)

        self.assertEqual([stock.stock_code for stock in limit_stocks], ["000001.SZ", "000003.SZ"])
        self.assertEqual(limit_stocks[0].stock_name, "样本一")
        self.assertEqual(limit_stocks[0].turnover_rate, 3.0)
        self.assertEqual([stock.stock_code for stock in strong_stocks], ["000002.SZ"])
        self.assertEqual(strong_stocks[0].industry, "AI算力")

    def test_snapshot_is_cached_by_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeTushareClient()
            provider = TushareDailyMarketProvider(client=client, cache_dir=Path(directory))

            provider.get_daily_quotes(trade_date="20260912")
            provider.get_daily_quotes(trade_date="20260912")

        self.assertEqual(client.daily_calls, 1)


if __name__ == "__main__":
    unittest.main()
