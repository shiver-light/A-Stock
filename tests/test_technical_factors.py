from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from factors.fundamental import bp_factor, ep_ttm_factor, roe_ttm_factor
from factors.library import get_factor_function
from factors.technical import (
    amount_mean_20d_factor,
    amount_mild_expansion_5d_factor,
    amplitude_20d_factor,
    close_to_high_20d_factor,
    close_near_high_on_high_amount_20d_factor,
    distribution_risk_20d_negative_factor,
    down_day_absorption_20d_factor,
    down_day_support_20d_factor,
    high_turnover_low_range_20d_factor,
    illiq_negative_factor,
    kdj_bullish_divergence_20d_factor,
    kdj_golden_cross_factor,
    kdj_j_turn_up_factor,
    liquidity_improvement_20d_factor,
    max_drawdown_60d_negative_factor,
    ma5_ma10_breakout_factor,
    momentum_60d_factor,
    money_flow_strength_20d_factor,
    price_new_low_20d_factor,
    position_safety_60d_factor,
    return_120d_factor,
    return_5d_negative_factor,
    return_60d_factor,
    price_suppression_20d_factor,
    pullback_after_trend_60d_factor,
    reversal_5d_factor,
    small_body_high_turnover_20d_factor,
    low_range_high_amount_days_ratio_20d_factor,
    gap_risk_20d_negative_factor,
    turnover_volatility_20d_factor,
    turnover_stability_20d_factor,
    volatility_contraction_20d_factor,
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

    @patch("factors.technical.get_a_share_daily_valuation")
    def test_turnover_stability_20d_factor(self, mock_get_daily_valuation) -> None:
        dates = self._date_range(25)
        mock_get_daily_valuation.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "turnover_rate_f": [1.0] * 25,
            }
        )

        result = turnover_stability_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "turnover_stability_20d")
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

    @patch("factors.technical._load_qfq_daily")
    def test_price_new_low_20d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(25)
        closes = [100.0 - index for index in range(25)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "close": closes,
            }
        )

        result = price_new_low_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "price_new_low_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 1.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_amount_mild_expansion_5d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        amounts = [100.0] * 20 + [160.0] * 5
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "amount": amounts,
            }
        )

        result = amount_mild_expansion_5d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "amount_mild_expansion_5d")
        self.assertGreater(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_daily")
    def test_ma5_ma10_breakout_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(15)
        closes = [10.0] * 10 + [10.2, 10.4, 10.6, 10.8, 11.0]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 15,
                "close": closes,
            }
        )

        result = ma5_ma10_breakout_factor(
            ts_code="000001.SZ",
            start_date=dates[10],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "ma5_ma10_breakout")
        self.assertGreater(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_kdj_j_turn_up_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        closes = [10.0] * 10 + [9.0, 8.0, 7.0, 6.0, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )

        result = kdj_j_turn_up_factor(
            ts_code="000001.SZ",
            start_date=dates[15],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_j_turn_up")
        self.assertTrue(result["factor_value"].notna().any())

    @patch("factors.technical._load_qfq_market_data")
    def test_kdj_golden_cross_factor_preserves_schema(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        closes = [10.0] * 10 + [9.0, 8.0, 7.0, 6.0, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )

        result = kdj_golden_cross_factor(
            ts_code="000001.SZ",
            start_date=dates[15],
            end_date=dates[-1],
        )

        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "factor_name", "factor_value"])
        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_golden_cross")

    @patch("factors.technical._load_qfq_market_data")
    def test_kdj_bullish_divergence_20d_factor_preserves_schema(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(35)
        closes = [10.0] * 10 + [9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0] + [5.2] * 10 + [4.9] * 5
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 35,
                "high": [value + 0.8 for value in closes],
                "low": [value - 0.8 for value in closes],
                "close": closes,
            }
        )

        result = kdj_bullish_divergence_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "factor_name", "factor_value"])
        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_bullish_divergence_20d")

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

    @patch("factors.technical._load_qfq_market_data")
    @patch("factors.technical._load_turnover_data")
    def test_small_body_high_turnover_20d_factor(self, mock_load_turnover_data, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "open": [10.0] * 25,
                "close": [10.1] * 25,
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

        result = small_body_high_turnover_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "small_body_high_turnover_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 200.0, places=1)

    @patch("factors.technical._load_qfq_market_data")
    def test_close_near_high_on_high_amount_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [8.0] * 25,
                "close": [9.5] * 25,
                "amount": [99.0] * 25,
            }
        )

        result = close_near_high_on_high_amount_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "close_near_high_on_high_amount_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.75 * np.log(100.0), places=6)

    @patch("factors.technical._load_qfq_market_data")
    def test_down_day_absorption_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [6.0] * 25,
                "close": [9.0] * 25,
                "pre_close": [9.5] * 25,
                "amount": [99.0] * 25,
            }
        )

        result = down_day_absorption_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "down_day_absorption_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.75 * np.log(100.0), places=6)

    @patch("factors.technical._load_qfq_market_data")
    def test_down_day_absorption_20d_factor_preserves_schema_without_down_days(
        self,
        mock_load_qfq_market_data,
    ) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [6.0] * 25,
                "close": [9.0] * 25,
                "pre_close": [8.5] * 25,
                "amount": [99.0] * 25,
            }
        )

        result = down_day_absorption_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(
            result.columns.tolist(),
            ["trade_date", "ts_code", "factor_name", "factor_value"],
        )
        self.assertEqual(result["trade_date"].tolist(), dates[20:])
        self.assertTrue(result["factor_value"].isna().all())

    @patch("factors.technical._load_qfq_market_data")
    def test_down_day_absorption_20d_factor_uses_available_down_days_within_window(
        self,
        mock_load_qfq_market_data,
    ) -> None:
        dates = self._date_range(25)
        close = [10.0] * 25
        pre_close = [9.0] * 25
        for index in [5, 10, 15]:
            close[index] = 9.0
            pre_close[index] = 9.5

        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [10.0] * 25,
                "low": [6.0] * 25,
                "close": close,
                "pre_close": pre_close,
                "amount": [99.0] * 25,
            }
        )

        result = down_day_absorption_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.75 * np.log(100.0), places=6)

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

    @patch("factors.technical._load_qfq_market_data")
    def test_gap_risk_20d_negative_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "open": [10.0] * 25,
                "pre_close": [10.0] * 25,
            }
        )

        result = gap_risk_20d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "gap_risk_20d_negative")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_low_range_high_amount_days_ratio_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "high": [11.0] * 25,
                "low": [9.0] * 25,
                "pre_close": [10.0] * 25,
                "amount": [100.0] * 25,
            }
        )

        result = low_range_high_amount_days_ratio_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "low_range_high_amount_days_ratio_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 0.3)

    @patch("factors.technical._load_qfq_daily")
    def test_pullback_after_trend_60d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(70)
        close = [100.0 + index for index in range(65)] + [164.0, 163.0, 162.0, 161.0, 160.0]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 70,
                "close": close,
            }
        )

        result = pullback_after_trend_60d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        expected_return_60d = (close[-1] / close[-61]) - 1.0
        expected_return_5d = (close[-1] / close[-6]) - 1.0
        self.assertEqual(result.iloc[-1]["factor_name"], "pullback_after_trend_60d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], expected_return_60d - expected_return_5d)

    @patch("factors.technical._load_qfq_market_data")
    @patch("factors.technical._load_turnover_data")
    def test_distribution_risk_20d_negative_factor(
        self,
        mock_load_turnover_data,
        mock_load_qfq_market_data,
    ) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "open": [9.0] * 25,
                "high": [12.0] * 25,
                "low": [8.0] * 25,
                "close": [9.0] * 25,
                "pre_close": [10.0] * 25,
                "amount": [99.0] * 25,
            }
        )
        mock_load_turnover_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 25,
                "turnover_rate_f": [2.0] * 25,
            }
        )

        result = distribution_risk_20d_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        expected_risk = 2.0 + (0.75 * np.log(100.0)) + (2.0 / 0.11)
        self.assertEqual(result.iloc[-1]["factor_name"], "distribution_risk_20d_negative")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], -expected_risk, places=6)

    @patch("factors.technical._load_qfq_market_data")
    @patch("factors.technical._load_turnover_data")
    def test_liquidity_improvement_20d_factor(
        self,
        mock_load_turnover_data,
        mock_load_qfq_market_data,
    ) -> None:
        dates = self._date_range(70)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 70,
                "amount": [100.0] * 50 + [200.0] * 20,
            }
        )
        mock_load_turnover_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 70,
                "turnover_rate_f": [1.0] * 50 + [2.0] * 20,
            }
        )

        result = liquidity_improvement_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "liquidity_improvement_20d")
        self.assertAlmostEqual(result.iloc[-1]["factor_value"], 1.5)

    @patch("factors.technical._load_qfq_market_data")
    def test_volatility_contraction_20d_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(70)
        close = [10.0 + (2.0 if index % 2 else 0.0) for index in range(50)]
        close.extend([10.0 + (0.2 if index % 2 else 0.0) for index in range(20)])
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 70,
                "high": [11.0] * 50 + [10.5] * 20,
                "low": [9.0] * 50 + [9.5] * 20,
                "close": close,
                "pre_close": [10.0] * 70,
            }
        )

        result = volatility_contraction_20d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "volatility_contraction_20d")
        self.assertTrue(np.isfinite(result.iloc[-1]["factor_value"]))
        self.assertLess(result.iloc[-1]["factor_value"], 0.0)

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
        self.assertIs(get_factor_function("amount_mild_expansion_5d"), amount_mild_expansion_5d_factor)
        self.assertIs(get_factor_function("illiq_negative"), illiq_negative_factor)
        self.assertIs(get_factor_function("money_flow_strength_20d"), money_flow_strength_20d_factor)
        self.assertIs(get_factor_function("price_new_low_20d"), price_new_low_20d_factor)
        self.assertIs(get_factor_function("kdj_bullish_divergence_20d"), kdj_bullish_divergence_20d_factor)
        self.assertIs(get_factor_function("kdj_j_turn_up"), kdj_j_turn_up_factor)
        self.assertIs(get_factor_function("kdj_golden_cross"), kdj_golden_cross_factor)
        self.assertIs(get_factor_function("ma5_ma10_breakout"), ma5_ma10_breakout_factor)
        self.assertIs(get_factor_function("high_turnover_low_range_20d"), high_turnover_low_range_20d_factor)
        self.assertIs(get_factor_function("price_suppression_20d"), price_suppression_20d_factor)
        self.assertIs(get_factor_function("down_day_support_20d"), down_day_support_20d_factor)
        self.assertIs(get_factor_function("small_body_high_turnover_20d"), small_body_high_turnover_20d_factor)
        self.assertIs(
            get_factor_function("close_near_high_on_high_amount_20d"),
            close_near_high_on_high_amount_20d_factor,
        )
        self.assertIs(get_factor_function("down_day_absorption_20d"), down_day_absorption_20d_factor)
        self.assertIs(get_factor_function("position_safety_60d"), position_safety_60d_factor)
        self.assertIs(get_factor_function("turnover_stability_20d"), turnover_stability_20d_factor)
        self.assertIs(get_factor_function("gap_risk_20d_negative"), gap_risk_20d_negative_factor)
        self.assertIs(
            get_factor_function("low_range_high_amount_days_ratio_20d"),
            low_range_high_amount_days_ratio_20d_factor,
        )
        self.assertIs(get_factor_function("amplitude_20d"), amplitude_20d_factor)
        self.assertIs(get_factor_function("close_to_high_20d"), close_to_high_20d_factor)
        self.assertIs(get_factor_function("pullback_after_trend_60d"), pullback_after_trend_60d_factor)
        self.assertIs(get_factor_function("distribution_risk_20d_negative"), distribution_risk_20d_negative_factor)
        self.assertIs(get_factor_function("liquidity_improvement_20d"), liquidity_improvement_20d_factor)
        self.assertIs(get_factor_function("volatility_contraction_20d"), volatility_contraction_20d_factor)
        self.assertIs(get_factor_function("ep_ttm"), ep_ttm_factor)
        self.assertIs(get_factor_function("bp"), bp_factor)
        self.assertIs(get_factor_function("roe_ttm"), roe_ttm_factor)


if __name__ == "__main__":
    unittest.main()
