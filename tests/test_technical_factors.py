from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from factors.library import get_factor_function
from factors.technical import (
    amplitude_20d_factor,
    close_to_high_20d_factor,
    momentum_60d_factor,
    return_60d_factor,
    reversal_5d_factor,
    turnover_volatility_20d_factor,
    volatility_60d_factor,
)


class TechnicalFactorsTestCase(unittest.TestCase):
    def _date_range(self, periods: int) -> list[str]:
        return pd.date_range("2024-01-01", periods=periods, freq="D").strftime("%Y%m%d").tolist()

    @patch("factors.technical._load_qfq_daily")
    def test_return_60d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(65)
        closes = [100.0 * (1.01**index) for index in range(65)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": closes,
            }
        )

        result = return_60d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "return_60d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], (1.01**60) - 1.0, places=10)

    @patch("factors.technical._load_qfq_daily")
    def test_momentum_60d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(65)
        closes = [100.0 * (1.01**index) for index in range(65)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": closes,
            }
        )

        result = momentum_60d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "momentum_60d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], (1.01**60) - 1.0, places=10)

    @patch("factors.technical._load_qfq_daily")
    def test_volatility_60d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(65)
        closes = [100.0 + float(index % 5) for index in range(65)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": closes,
            }
        )

        result = volatility_60d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "volatility_60d")
        self.assertGreaterEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_daily")
    def test_reversal_5d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(10)
        closes = [100.0 * (1.02**index) for index in range(10)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 10,
                "close": closes,
            }
        )

        result = reversal_5d_factor(
            ts_code="000001.SZ",
            start_date=dates[5],
            end_date=dates[-1],
        )

        self.assertAlmostEqual(result.iloc[-1]["factor_value"], -((1.02**5) - 1.0), places=10)

    @patch("factors.technical.get_a_share_daily_valuation")
    def test_turnover_volatility_20d_factor(self, mock_get_daily_valuation) -> None:
        dates = self._date_range(25)
        mock_get_daily_valuation.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "turnover_rate_f": [1.0] * 25,
            }
        )

        result = turnover_volatility_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "turnover_volatility_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_amplitude_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [12.0] * 25,
                "low": [8.0] * 25,
                "pre_close": [10.0] * 25,
            }
        )

        result = amplitude_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "amplitude_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.4)

    @patch("factors.technical._load_qfq_market_data")
    def test_close_to_high_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "close": [8.0] * 25,
                "high": [10.0] * 25,
            }
        )

        result = close_to_high_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "close_to_high_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.8)

    def test_factor_library_registers_new_technical_factors(self) -> None:
        self.assertIs(get_factor_function("return_60d"), return_60d_factor)
        self.assertIs(get_factor_function("momentum_60d"), momentum_60d_factor)
        self.assertIs(get_factor_function("reversal_5d"), reversal_5d_factor)
        self.assertIs(get_factor_function("volatility_60d"), volatility_60d_factor)
        self.assertIs(get_factor_function("turnover_volatility_20d"), turnover_volatility_20d_factor)
        self.assertIs(get_factor_function("amplitude_20d"), amplitude_20d_factor)
        self.assertIs(get_factor_function("close_to_high_20d"), close_to_high_20d_factor)


if __name__ == "__main__":
    unittest.main()
