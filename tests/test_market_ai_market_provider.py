from __future__ import annotations

import unittest

import pandas as pd

from market_ai.models import LimitStock, StrongStock
from market_ai.providers.market import MarketProvider


class FakeMarketProvider(MarketProvider):
    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        return [
            LimitStock(
                trade_date=trade_date,
                stock_code="000001.SZ",
                stock_name="示例股份",
                pct_chg=10.0,
                status="limit_up",
                source="fake",
            )
        ]

    def get_strong_stocks(self, *, trade_date: str, min_pct_chg: float = 7.0) -> list[StrongStock]:
        return [
            StrongStock(
                trade_date=trade_date,
                stock_code="000002.SZ",
                stock_name="强势股份",
                pct_chg=min_pct_chg,
                source="fake",
            )
        ]

    def get_daily_quotes(self, *, trade_date: str, stock_codes=None) -> pd.DataFrame:
        codes = list(stock_codes or ["000001.SZ", "000002.SZ"])
        return pd.DataFrame(
            {
                "trade_date": [trade_date] * len(codes),
                "stock_code": codes,
                "stock_name": ["示例股份"] * len(codes),
                "close": [10.0] * len(codes),
                "pct_chg": [1.0] * len(codes),
            }
        )


class MarketProviderTestCase(unittest.TestCase):
    def test_market_provider_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            MarketProvider()

    def test_fake_provider_returns_structured_limit_and_strong_stocks(self) -> None:
        provider = FakeMarketProvider()

        limit_stocks = provider.get_limit_stocks(trade_date="20260912")
        strong_stocks = provider.get_strong_stocks(trade_date="20260912", min_pct_chg=7.0)

        self.assertEqual(limit_stocks[0].stock_code, "000001.SZ")
        self.assertEqual(limit_stocks[0].status, "limit_up")
        self.assertEqual(strong_stocks[0].pct_chg, 7.0)

    def test_fake_provider_quotes_can_be_filtered_by_stock_codes(self) -> None:
        provider = FakeMarketProvider()

        quotes = provider.get_daily_quotes(trade_date="20260912", stock_codes=["000001.SZ"])

        self.assertEqual(quotes["stock_code"].tolist(), ["000001.SZ"])
        self.assertEqual(quotes["trade_date"].tolist(), ["20260912"])


if __name__ == "__main__":
    unittest.main()
