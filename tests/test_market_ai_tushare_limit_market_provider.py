from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_ai.providers.market import TushareLimitMarketProvider


class FakeTushareLimitClient:
    def __init__(self) -> None:
        self.limit_calls = 0
        self.daily_calls = 0

    def limit_list_d(self, *, trade_date=None, fields=None, **kwargs) -> pd.DataFrame:
        self.limit_calls += 1
        return pd.DataFrame(
            {
                "trade_date": [trade_date, trade_date, trade_date],
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["样本一", "样本二", "样本三"],
                "close": [11.0, 9.0, 21.0],
                "pct_chg": [10.0, -10.0, 9.5],
                "amount": [100000.0, 200000.0, 300000.0],
                "turnover_ratio": [3.0, 4.0, 5.0],
                "first_time": ["093000", "100000", "101000"],
                "last_time": ["145700", "143000", "144000"],
                "open_times": [0, 1, 3],
                "limit_times": [2, 1, 1],
                "fd_amount": [50000.0, 10000.0, 0.0],
                "industry": ["AI算力", "银行", "光通信"],
                "limit": ["U", "D", "Z"],
            }
        )

    def daily(self, *, trade_date=None, fields=None, **kwargs) -> pd.DataFrame:
        self.daily_calls += 1
        return pd.DataFrame(
            {
                "ts_code": ["000004.SZ"],
                "trade_date": [trade_date],
                "open": [10.0],
                "high": [11.0],
                "low": [9.8],
                "close": [10.8],
                "pre_close": [10.0],
                "pct_chg": [8.0],
                "vol": [1000.0],
                "amount": [100000.0],
            }
        )

    def daily_basic(self, *, trade_date=None, fields=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000004.SZ"],
                "trade_date": [trade_date],
                "turnover_rate": [3.0],
                "volume_ratio": [1.2],
            }
        )

    def stock_basic(self, *, list_status=None, fields=None, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000004.SZ"],
                "name": ["样本四"],
                "industry": ["AI算力"],
            }
        )


class TushareLimitMarketProviderTestCase(unittest.TestCase):
    def test_get_limit_stocks_from_limit_list_d(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = TushareLimitMarketProvider(
                client=FakeTushareLimitClient(),
                cache_dir=Path(directory),
            )

            limit_stocks = provider.get_limit_stocks(trade_date="20260912")

        self.assertEqual([stock.stock_code for stock in limit_stocks], ["000001.SZ", "000003.SZ"])
        self.assertEqual(limit_stocks[0].first_limit_time, "093000")
        self.assertEqual(limit_stocks[0].consecutive_limit_count, 2)
        self.assertEqual(limit_stocks[1].status, "failed_limit")

    def test_strong_stocks_reuse_daily_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = TushareLimitMarketProvider(
                client=FakeTushareLimitClient(),
                cache_dir=Path(directory),
            )

            strong_stocks = provider.get_strong_stocks(trade_date="20260912", min_pct_chg=7.0)

        self.assertEqual([stock.stock_code for stock in strong_stocks], ["000004.SZ"])

    def test_limit_snapshot_is_cached_by_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeTushareLimitClient()
            provider = TushareLimitMarketProvider(client=client, cache_dir=Path(directory))

            provider.get_limit_stocks(trade_date="20260912")
            provider.get_limit_stocks(trade_date="20260912")

        self.assertEqual(client.limit_calls, 1)


if __name__ == "__main__":
    unittest.main()
