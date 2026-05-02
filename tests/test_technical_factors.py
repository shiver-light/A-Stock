from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from factors.fundamental import bp_factor, ep_ttm_factor, roe_ttm_factor
from factors.library import get_factor_function
from factors.technical import (
    amount_mean_20d_factor,
    amplitude_20d_factor,
    close_to_high_20d_factor,
    down_day_support_20d_factor,
    high_turnover_low_range_20d_factor,
    illiq_negative_factor,
    max_drawdown_60d_negative_factor,
    momentum_60d_factor,
    money_flow_strength_20d_factor,
    position_safety_60d_factor,
    return_120d_factor,
    return_5d_negative_factor,
    return_60d_factor,
    price_suppression_20d_factor,
    reversal_5d_factor,
    turnover_volatility_20d_factor,
    volatility_20d_negative_factor,
    volatility_60d_factor,
    volatility_60d_negative_factor,
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
    def test_return_120d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(130)
        closes = [100.0 * (1.005**index) for index in range(130)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 130,
                "close": closes,
            }
        )

        result = return_120d_factor(
            ts_code="000001.SZ",
            start_date=dates[120],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "return_120d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], (1.005**120) - 1.0, places=10)

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

    @patch("factors.technical._load_qfq_daily")
    def test_return_5d_negative_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(10)
        closes = [100.0 * (1.01**index) for index in range(10)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 10,
                "close": closes,
            }
        )

        result = return_5d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[5],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "return_5d_negative")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], -((1.01**5) - 1.0), places=10)

    @patch("factors.technical._load_qfq_daily")
    def test_volatility_20d_negative_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(25)
        closes = [100.0 + float(index % 4) for index in range(25)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "close": closes,
            }
        )

        result = volatility_20d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "volatility_20d_negative")
        self.assertLessEqual(result.iloc[-1]["factor_value"], 0.0)

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

    @patch("factors.technical._load_qfq_daily")
    def test_volatility_60d_negative_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(65)
        closes = [100.0 + float(index % 5) for index in range(65)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": closes,
            }
        )

        result = volatility_60d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "volatility_60d_negative")
        self.assertLessEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_daily")
    def test_max_drawdown_60d_negative_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(65)
        closes = [100.0 + float(index) for index in range(60)] + [140.0, 135.0, 130.0, 125.0, 120.0]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": closes,
            }
        )

        result = max_drawdown_60d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "max_drawdown_60d_negative")
        self.assertLessEqual(result.iloc[-1]["factor_value"], 0.0)

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

    @patch("factors.technical._load_qfq_market_data")
    def test_amount_mean_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "amount": [100.0] * 25,
            }
        )

        result = amount_mean_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "amount_mean_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 100.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_illiq_negative_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        closes = [100.0 * (1.01**index) for index in range(25)]
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "close": closes,
                "amount": [1000000.0] * 25,
            }
        )

        result = illiq_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "illiq_negative")
        self.assertLessEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_money_flow_strength_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [8.0] * 25,
                "close": [9.5] * 25,
                "amount": [100.0] * 25,
            }
        )

        result = money_flow_strength_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "money_flow_strength_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 50.0)

    @patch("factors.technical._load_qfq_market_data")
    @patch("factors.technical._load_turnover_data")
    def test_high_turnover_low_range_20d_factor(self, mock_load_turnover_data, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [11.0] * 25,
                "low": [9.0] * 25,
                "pre_close": [10.0] * 25,
            }
        )
        mock_load_turnover_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "turnover_rate_f": [2.0] * 25,
            }
        )

        result = high_turnover_low_range_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "high_turnover_low_range_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 10.0)

    @patch("factors.technical._load_qfq_daily")
    @patch("factors.technical._load_turnover_data")
    def test_price_suppression_20d_factor(self, mock_load_turnover_data, mock_load_qfq_daily) -> None:
        dates = self._date_range(25)
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "close": [10.0] * 25,
            }
        )
        mock_load_turnover_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "turnover_rate_f": [2.0] * 25,
            }
        )

        result = price_suppression_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "price_suppression_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 2.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_down_day_support_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [6.0] * 25,
                "close": [9.0] * 25,
                "pre_close": [9.5] * 25,
            }
        )

        result = down_day_support_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "down_day_support_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.75)

    @patch("factors.technical._load_qfq_market_data")
    def test_position_safety_60d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(65)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 65,
                "close": [8.0] * 65,
                "high": [10.0] * 65,
            }
        )

        result = position_safety_60d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "position_safety_60d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.2)

    @patch("factors.fundamental.get_a_share_daily_valuation")
    def test_ep_ttm_factor(self, mock_get_daily_valuation) -> None:
        mock_get_daily_valuation.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "pe_ttm": [10.0, -5.0],
            }
        )

        result = ep_ttm_factor(ts_code="000001.SZ", start_date="20240101", end_date="20240131")

        self.assertEqual(result.iloc[0]["factor_value"], 0.1)
        self.assertTrue(pd.isna(result.iloc[1]["factor_value"]))

    @patch("factors.fundamental.get_a_share_daily_valuation")
    def test_bp_factor(self, mock_get_daily_valuation) -> None:
        mock_get_daily_valuation.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "pb": [2.0, 0.0],
            }
        )

        result = bp_factor(ts_code="000001.SZ", start_date="20240101", end_date="20240131")

        self.assertEqual(result.iloc[0]["factor_value"], 0.5)
        self.assertTrue(pd.isna(result.iloc[1]["factor_value"]))

    @patch("factors.fundamental.get_a_share_fundamental_daily")
    def test_roe_ttm_factor(self, mock_get_daily_fundamental) -> None:
        mock_get_daily_fundamental.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "ts_code": ["000001.SZ"],
                "roe": [12.3],
            }
        )

        result = roe_ttm_factor(ts_code="000001.SZ", start_date="20240101", end_date="20240131")

        self.assertEqual(result.iloc[0]["factor_name"], "roe_ttm")
        self.assertEqual(result.iloc[0]["factor_value"], 12.3)

    def test_factor_library_registers_new_technical_factors(self) -> None:
        self.assertIs(get_factor_function("return_60d"), return_60d_factor)
        self.assertIs(get_factor_function("return_120d"), return_120d_factor)
        self.assertIs(get_factor_function("return_5d_negative"), return_5d_negative_factor)
        self.assertIs(get_factor_function("momentum_60d"), momentum_60d_factor)
        self.assertIs(get_factor_function("reversal_5d"), reversal_5d_factor)
        self.assertIs(get_factor_function("volatility_20d_negative"), volatility_20d_negative_factor)
        self.assertIs(get_factor_function("volatility_60d"), volatility_60d_factor)
        self.assertIs(get_factor_function("volatility_60d_negative"), volatility_60d_negative_factor)
        self.assertIs(get_factor_function("max_drawdown_60d_negative"), max_drawdown_60d_negative_factor)
        self.assertIs(get_factor_function("turnover_volatility_20d"), turnover_volatility_20d_factor)
        self.assertIs(get_factor_function("amount_mean_20d"), amount_mean_20d_factor)
        self.assertIs(get_factor_function("illiq_negative"), illiq_negative_factor)
        self.assertIs(get_factor_function("money_flow_strength_20d"), money_flow_strength_20d_factor)
        self.assertIs(get_factor_function("high_turnover_low_range_20d"), high_turnover_low_range_20d_factor)
        self.assertIs(get_factor_function("price_suppression_20d"), price_suppression_20d_factor)
        self.assertIs(get_factor_function("down_day_support_20d"), down_day_support_20d_factor)
        self.assertIs(get_factor_function("position_safety_60d"), position_safety_60d_factor)
        self.assertIs(get_factor_function("amplitude_20d"), amplitude_20d_factor)
        self.assertIs(get_factor_function("close_to_high_20d"), close_to_high_20d_factor)
        self.assertIs(get_factor_function("ep_ttm"), ep_ttm_factor)
        self.assertIs(get_factor_function("bp"), bp_factor)
        self.assertIs(get_factor_function("roe_ttm"), roe_ttm_factor)


if __name__ == "__main__":
    unittest.main()
