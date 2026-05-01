from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from data.daily_market import AShareDailyMarketService, DailyMarketRequest


class DailyMarketServiceTestCase(unittest.TestCase):
    def test_get_daily_refills_missing_adj_factor_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = Mock()
            client.daily.return_value = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "trade_date": ["20240102", "20240103", "20240104"],
                    "open": [10.0, 11.0, 12.0],
                    "high": [10.5, 11.5, 12.5],
                    "low": [9.5, 10.5, 11.5],
                    "close": [10.0, 11.0, 12.0],
                    "pre_close": [9.8, 10.0, 11.0],
                    "change": [0.2, 1.0, 1.0],
                    "pct_chg": [2.0, 10.0, 9.1],
                    "vol": [100.0, 100.0, 100.0],
                    "amount": [1000.0, 1100.0, 1200.0],
                }
            )
            client.adj_factor.side_effect = [
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000001.SZ"],
                        "trade_date": ["20240102", "20240104"],
                        "adj_factor": [1.0, 1.2],
                    }
                ),
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "trade_date": ["20240103"],
                        "adj_factor": [1.1],
                    }
                ),
            ]

            service = AShareDailyMarketService(client=client, cache_dir=tmp_dir)
            request = DailyMarketRequest(
                ts_code="000001.SZ",
                start_date="20240102",
                end_date="20240104",
                adjust="qfq",
                fields=("trade_date", "ts_code", "close"),
            )

            result = service.get_daily(request)

            self.assertEqual(result["trade_date"].tolist(), ["20240102", "20240103", "20240104"])
            self.assertEqual(client.adj_factor.call_count, 2)
            self.assertEqual(client.adj_factor.call_args_list[1].kwargs["start_date"], "20240103")
            self.assertEqual(client.adj_factor.call_args_list[1].kwargs["end_date"], "20240103")

    def test_compress_date_ranges_groups_contiguous_dates(self) -> None:
        service = AShareDailyMarketService(client=Mock(), cache_dir=Path(tempfile.gettempdir()) / "a_stock_daily_test")
        ranges = service._compress_date_ranges(["20240102", "20240103", "20240105", "20240108", "20240109"])
        self.assertEqual(
            ranges,
            [("20240102", "20240103"), ("20240105", "20240105"), ("20240108", "20240109")],
        )


if __name__ == "__main__":
    unittest.main()
