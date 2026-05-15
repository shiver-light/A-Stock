from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from data.us_market import USDailyRequest, USMarketDailyService, build_us_market_regime


class USMarketTestCase(unittest.TestCase):
    def test_get_daily_uses_tushare_us_daily_and_cache(self) -> None:
        client = Mock()
        client.us_daily.return_value = pd.DataFrame(
            {
                "ts_code": ["SPY", "SPY"],
                "trade_date": ["20240102", "20240103"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "pre_close": [100.0, 100.5],
                "change": [0.5, 1.0],
                "vol": [1000.0, 1100.0],
                "amount": [100000.0, 120000.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            service = USMarketDailyService(client=client, cache_dir=tmp_dir)
            result = service.get_daily(
                USDailyRequest(
                    ts_code="SPY",
                    start_date="20240102",
                    end_date="20240103",
                    fields=("trade_date", "ts_code", "close"),
                )
            )

            self.assertEqual(result["trade_date"].tolist(), ["20240102", "20240103"])
            self.assertEqual(result["close"].tolist(), [100.5, 101.5])
            client.us_daily.assert_called_once()

    def test_build_us_market_regime_aligns_to_prior_us_close_only(self) -> None:
        prices = _prices(
            {
                "SPY": [100.0, 101.0, 102.0],
                "QQQ": [100.0, 103.0, 105.0],
                "IWM": [100.0, 99.0, 99.5],
                "VXX": [100.0, 95.0, 93.0],
            },
            dates=["20240102", "20240103", "20240104"],
        )

        result = build_us_market_regime(prices, ["20240103", "20240104", "20240105"], lookback=1)

        self.assertEqual(result["trade_date"].tolist(), ["20240103", "20240104", "20240105"])
        self.assertEqual(result["source_us_trade_date"].tolist(), ["20240102", "20240103", "20240104"])
        self.assertEqual(result["us_tech_strong"].tolist(), [0, 1, 1])

    def test_build_us_market_regime_flags_risk_and_vix_stress(self) -> None:
        prices = _prices(
            {
                "SPY": [100.0, 101.0, 99.0],
                "QQQ": [100.0, 102.0, 98.0],
                "IWM": [100.0, 103.0, 97.0],
                "VXX": [100.0, 95.0, 120.0],
            },
            dates=["20240102", "20240103", "20240104"],
        )

        result = build_us_market_regime(prices, ["20240104", "20240105"], lookback=1, vix_stress_threshold=0.1)

        self.assertEqual(result["source_us_trade_date"].tolist(), ["20240103", "20240104"])
        self.assertEqual(result["us_risk_on"].tolist(), [1, 0])
        self.assertEqual(result["vix_stress"].tolist(), [0, 1])

    def test_build_us_market_regime_rejects_missing_required_code(self) -> None:
        prices = _prices(
            {
                "SPY": [100.0, 101.0],
                "QQQ": [100.0, 102.0],
                "IWM": [100.0, 103.0],
            },
            dates=["20240102", "20240103"],
        )

        with self.assertRaisesRegex(ValueError, "missing required ts_code"):
            build_us_market_regime(prices, ["20240104"], lookback=1)

    def test_build_us_market_regime_adds_theme_strength_columns(self) -> None:
        prices = _prices(
            {
                "SPY": [100.0, 101.0, 102.0],
                "QQQ": [100.0, 102.0, 103.0],
                "IWM": [100.0, 100.0, 101.0],
                "VXX": [100.0, 95.0, 93.0],
                "SOXX": [100.0, 106.0, 110.0],
            },
            dates=["20240102", "20240103", "20240104"],
        )

        result = build_us_market_regime(
            prices,
            ["20240104", "20240105"],
            lookback=1,
            theme_codes={"semiconductor": "SOXX"},
        )

        self.assertIn("semiconductor_strong", result.columns)
        self.assertIn("semiconductor_spy_relative_20d", result.columns)
        self.assertEqual(result["semiconductor_strong"].tolist(), [1, 1])


def _prices(values_by_code: dict[str, list[float]], *, dates: list[str]) -> pd.DataFrame:
    rows = []
    for code, values in values_by_code.items():
        for date, value in zip(dates, values, strict=True):
            rows.append({"trade_date": date, "ts_code": code, "close": value})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
