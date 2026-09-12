from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_ai.providers.market import LocalCsvMarketProvider


class LocalCsvMarketProviderTestCase(unittest.TestCase):
    def test_missing_files_return_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = LocalCsvMarketProvider(directory)

            self.assertEqual(provider.get_limit_stocks(trade_date="20260912"), [])
            self.assertEqual(provider.get_strong_stocks(trade_date="20260912"), [])

    def test_load_limit_and_strong_stocks_for_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame(
                {
                    "trade_date": ["20260912", "20260911"],
                    "stock_code": ["000001.SZ", "000002.SZ"],
                    "stock_name": ["样本一", "样本二"],
                    "pct_chg": [10.0, 10.0],
                    "amount": [1000.0, 500.0],
                    "consecutive_limit_count": [2, 1],
                    "limit_reason": ["AI服务器", "白酒"],
                    "concepts": ["CPO|液冷", ""],
                }
            ).to_csv(path / "limit_stocks.csv", index=False)
            pd.DataFrame(
                {
                    "trade_date": ["20260912", "20260912"],
                    "stock_code": ["000003.SZ", "000004.SZ"],
                    "stock_name": ["样本三", "样本四"],
                    "pct_chg": [8.0, 6.5],
                    "concepts": ["光模块", "机器人"],
                }
            ).to_csv(path / "strong_stocks.csv", index=False)
            provider = LocalCsvMarketProvider(path)

            limit_stocks = provider.get_limit_stocks(trade_date="20260912")
            strong_stocks = provider.get_strong_stocks(trade_date="20260912", min_pct_chg=7.0)

            self.assertEqual([stock.stock_code for stock in limit_stocks], ["000001.SZ"])
            self.assertEqual(limit_stocks[0].concepts, ["CPO", "液冷"])
            self.assertEqual([stock.stock_code for stock in strong_stocks], ["000003.SZ"])

    def test_missing_required_columns_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"trade_date": ["20260912"]}).to_csv(path / "limit_stocks.csv", index=False)
            provider = LocalCsvMarketProvider(path)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                provider.get_limit_stocks(trade_date="20260912")


if __name__ == "__main__":
    unittest.main()
