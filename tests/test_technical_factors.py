from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from factors.fundamental import (
    bp_factor,
    ep_ttm_factor,
    holder_num_change_ratio_negative_fresh_3d_announced_today_factor,
    holder_num_change_ratio_negative_fresh_3d_decay_10d_factor,
    holder_num_change_ratio_negative_fresh_3d_decay_3d_factor,
    holder_num_change_ratio_negative_fresh_3d_decay_5d_factor,
    roe_ttm_factor,
)
from factors.library import get_factor_function
from factors.technical import (
    amount_mean_20d_factor,
    amount_mild_expansion_5d_factor,
    amplitude_20d_factor,
    close_to_high_20d_factor,
    close_near_high_on_high_amount_20d_factor,
    daily_macd_golden_cross_2d_factor,
    daily_return_factor,
    distribution_risk_20d_negative_factor,
    down_day_absorption_20d_factor,
    down_day_support_20d_factor,
    false_breakout_risk_negative_factor,
    high_turnover_low_range_20d_factor,
    illiq_negative_factor,
    kdj_bullish_divergence_20d_factor,
    kdj_golden_cross_factor,
    kdj_golden_cross_3d_factor,
    kdj_j_turn_up_factor,
    kdj_j_turn_up_3d_factor,
    kdj_low_zone_cross_factor,
    liquidity_improvement_20d_factor,
    low_position_120d_factor,
    macd_hist_slope_5d_factor,
    macd_zero_axis_strength_factor,
    max_drawdown_60d_negative_factor,
    ma5_ma10_breakout_factor,
    ma5_ma10_breakout_3d_factor,
    ma20_slope_1d_factor,
    momentum_60d_factor,
    money_flow_strength_20d_factor,
    price_new_low_20d_factor,
    price_position_120d_factor,
    position_safety_60d_factor,
    post_cross_pullback_factor,
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
    turnover_rate_f_factor,
    volatility_contraction_20d_factor,
    volatility_20d_negative_factor,
    volatility_60d_factor,
    volatility_60d_negative_factor,
    weekly_kdj_golden_cross_2d_factor,
    weekly_macd_golden_cross_2d_factor,
    volume_ratio_5d_factor,
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

    @patch("factors.technical._load_qfq_daily")
    def test_ma5_ma10_breakout_3d_factor_uses_trailing_signal(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(15)
        closes = [10.0] * 10 + [10.2, 10.4, 10.6, 10.8, 10.7]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * 15,
                "close": closes,
            }
        )

        result = ma5_ma10_breakout_3d_factor(
            ts_code="000001.SZ",
            start_date=dates[10],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "ma5_ma10_breakout_3d")
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
    def test_kdj_j_turn_up_3d_factor_carries_recent_past_signal(self, mock_load_qfq_market_data) -> None:
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

        result = kdj_j_turn_up_3d_factor(
            ts_code="000001.SZ",
            start_date=dates[15],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_j_turn_up_3d")
        self.assertGreater(result["factor_value"].notna().sum(), 0)

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
    def test_kdj_golden_cross_3d_factor_preserves_schema(self, mock_load_qfq_market_data) -> None:
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

        result = kdj_golden_cross_3d_factor(
            ts_code="000001.SZ",
            start_date=dates[15],
            end_date=dates[-1],
        )

        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "factor_name", "factor_value"])
        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_golden_cross_3d")

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

    @patch("factors.technical._load_qfq_daily")
    def test_weekly_macd_golden_cross_2d_factor_carries_today_and_yesterday(
        self,
        mock_load_qfq_daily,
    ) -> None:
        dates = self._date_range(130)
        closes = [10.0] * 70 + [10.0 + (index * 0.25) for index in range(60)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        result = weekly_macd_golden_cross_2d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        signals = result.loc[result["factor_value"].notna()].reset_index(drop=True)
        self.assertGreaterEqual(len(signals), 2)
        self.assertEqual(signals.iloc[0]["factor_name"], "weekly_macd_golden_cross_2d")
        self.assertEqual(signals.iloc[0]["factor_value"], 1.0)
        self.assertEqual(signals.iloc[1]["factor_value"], 1.0)

    @patch("factors.technical._load_qfq_daily")
    def test_daily_macd_golden_cross_2d_factor_carries_today_and_yesterday(
        self,
        mock_load_qfq_daily,
    ) -> None:
        dates = self._date_range(80)
        closes = [10.0] * 40 + [10.0 + (index * 0.2) for index in range(40)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        result = daily_macd_golden_cross_2d_factor(
            ts_code="000001.SZ",
            start_date=dates[35],
            end_date=dates[-1],
        )

        signals = result.loc[result["factor_value"].notna()].reset_index(drop=True)
        self.assertGreaterEqual(len(signals), 2)
        self.assertEqual(signals.iloc[0]["factor_name"], "daily_macd_golden_cross_2d")
        self.assertEqual(signals.iloc[0]["factor_value"], 1.0)
        self.assertEqual(signals.iloc[1]["factor_value"], 1.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_weekly_kdj_golden_cross_2d_factor_carries_today_and_yesterday(
        self,
        mock_load_qfq_market_data,
    ) -> None:
        dates = self._date_range(130)
        closes = [10.0] * 70 + [9.0] * 20 + [9.0 + (index * 0.2) for index in range(40)]
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )

        result = weekly_kdj_golden_cross_2d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        signals = result.loc[result["factor_value"].notna()].reset_index(drop=True)
        self.assertGreaterEqual(len(signals), 2)
        self.assertEqual(signals.iloc[0]["factor_name"], "weekly_kdj_golden_cross_2d")
        self.assertEqual(signals.iloc[0]["factor_value"], 1.0)
        self.assertEqual(signals.iloc[1]["factor_value"], 1.0)

    @patch("factors.technical._load_qfq_daily")
    def test_macd_hist_slope_5d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(80)
        closes = [10.0] * 40 + [10.0 + (index * 0.2) for index in range(40)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        result = macd_hist_slope_5d_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "macd_hist_slope_5d")
        self.assertTrue(result["factor_value"].notna().any())

    @patch("factors.technical._load_qfq_daily")
    def test_macd_zero_axis_strength_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(80)
        closes = [10.0] * 40 + [10.0 + (index * 0.2) for index in range(40)]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        result = macd_zero_axis_strength_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "macd_zero_axis_strength")
        self.assertTrue(result["factor_value"].notna().any())

    @patch("factors.technical._load_qfq_market_data")
    def test_kdj_low_zone_cross_factor_preserves_schema(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(35)
        closes = [10.0] * 10 + [9.0, 8.0, 7.0, 6.0, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0] + [6.2] * 15
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )

        result = kdj_low_zone_cross_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "factor_name", "factor_value"])
        self.assertEqual(result.iloc[-1]["factor_name"], "kdj_low_zone_cross")

    @patch("factors.technical._load_qfq_market_data")
    def test_post_cross_pullback_factor_preserves_schema(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(90)
        closes = [10.0] * 40 + [10.0 + (index * 0.2) for index in range(10)] + [11.8, 11.5, 11.3, 11.2] + [11.4] * 36
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "high": [value + 0.5 for value in closes],
                "low": [value - 0.5 for value in closes],
                "close": closes,
            }
        )

        result = post_cross_pullback_factor(
            ts_code="000001.SZ",
            start_date=dates[60],
            end_date=dates[-1],
        )

        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "factor_name", "factor_value"])
        self.assertEqual(result.iloc[-1]["factor_name"], "post_cross_pullback")

    @patch("factors.technical._load_qfq_market_data")
    def test_false_breakout_risk_negative_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(25)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "open": [10.0] * len(dates),
                "high": [11.0] * len(dates),
                "low": [9.5] * len(dates),
                "close": [9.8] * len(dates),
                "pre_close": [10.0] * len(dates),
                "amount": [100.0] * len(dates),
            }
        )

        result = false_breakout_risk_negative_factor(
            ts_code="000001.SZ",
            start_date=dates[20],
            end_date=dates[-1],
        )

        self.assertEqual(result.iloc[-1]["factor_name"], "false_breakout_risk_negative")
        self.assertLessEqual(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_market_data")
    def test_volume_ratio_5d_factor_uses_prior_volume(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(8)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "vol": [100.0, 100.0, 100.0, 100.0, 100.0, 250.0, 300.0, 350.0],
            }
        )

        result = volume_ratio_5d_factor(ts_code="000001.SZ", start_date=dates[5], end_date=dates[5])

        self.assertEqual(result.iloc[0]["factor_name"], "volume_ratio_5d")
        self.assertAlmostEqual(result.iloc[0]["factor_value"], 2.5)

    @patch("factors.technical._load_turnover_data")
    def test_turnover_rate_f_factor_preserves_percent_units(self, mock_load_turnover_data) -> None:
        dates = self._date_range(2)
        mock_load_turnover_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "turnover_rate_f": [3.2, 9.8],
            }
        )

        result = turnover_rate_f_factor(ts_code="000001.SZ", start_date=dates[0], end_date=dates[-1])

        self.assertEqual(result["factor_value"].tolist(), [3.2, 9.8])

    @patch("factors.technical._load_qfq_daily")
    def test_ma20_slope_1d_factor(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(25)
        closes = [10.0] * 20 + [10.1, 10.2, 10.3, 10.4, 10.5]
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        result = ma20_slope_1d_factor(ts_code="000001.SZ", start_date=dates[21], end_date=dates[-1])

        self.assertEqual(result.iloc[-1]["factor_name"], "ma20_slope_1d")
        self.assertGreater(result.iloc[-1]["factor_value"], 0.0)

    @patch("factors.technical._load_qfq_daily")
    def test_price_position_120d_and_low_position_120d_factors(self, mock_load_qfq_daily) -> None:
        dates = self._date_range(130)
        closes = [10.0 + index for index in range(120)] + [34.0] * 10
        mock_load_qfq_daily.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": closes,
            }
        )

        position = price_position_120d_factor(ts_code="000001.SZ", start_date=dates[-1], end_date=dates[-1])
        low_position = low_position_120d_factor(ts_code="000001.SZ", start_date=dates[-1], end_date=dates[-1])

        self.assertEqual(position.iloc[0]["factor_name"], "price_position_120d")
        self.assertEqual(low_position.iloc[0]["factor_name"], "low_position_120d")
        self.assertLess(position.iloc[0]["factor_value"], 0.25)
        self.assertGreater(low_position.iloc[0]["factor_value"], 0.75)

    @patch("factors.technical._load_qfq_market_data")
    def test_daily_return_factor(self, mock_load_qfq_market_data) -> None:
        dates = self._date_range(2)
        mock_load_qfq_market_data.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["000001.SZ"] * len(dates),
                "close": [10.2, 10.5],
                "pre_close": [10.0, 10.0],
            }
        )

        result = daily_return_factor(ts_code="000001.SZ", start_date=dates[0], end_date=dates[-1])

        self.assertAlmostEqual(result.iloc[0]["factor_value"], 0.02)
        self.assertAlmostEqual(result.iloc[1]["factor_value"], 0.05)

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

    @patch("factors.fundamental.get_a_share_holder_number_daily")
    def test_holder_num_change_ratio_negative_fresh_3d_announced_today_factor(self, mock_get_holder_daily) -> None:
        mock_get_holder_daily.return_value = pd.DataFrame(
            {
                "trade_date": ["20250410", "20250411", "20250414"],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "ann_date": ["20250410", "20250410", "20250414"],
                "holder_num_change_ratio_negative": [0.20, 0.20, 0.10],
                "holder_report_lag_trading_days": [2, 2, 3],
            }
        )

        result = holder_num_change_ratio_negative_fresh_3d_announced_today_factor(
            ts_code="000001.SZ",
            start_date="20250410",
            end_date="20250414",
        )

        self.assertEqual(result.iloc[0]["factor_value"], 0.20)
        self.assertTrue(pd.isna(result.iloc[1]["factor_value"]))
        self.assertTrue(pd.isna(result.iloc[2]["factor_value"]))

    @patch("factors.fundamental.get_a_share_holder_number_daily")
    def test_holder_num_change_ratio_negative_fresh_3d_decay_10d_factor(self, mock_get_holder_daily) -> None:
        mock_get_holder_daily.return_value = pd.DataFrame(
            {
                "trade_date": ["20250410", "20250411", "20250424", "20250425"],
                "ts_code": ["000001.SZ"] * 4,
                "holder_num_change_ratio_negative": [0.22, 0.22, 0.22, 0.22],
                "holder_report_lag_trading_days": [3, 3, 3, 4],
                "holder_announcement_age_trading_days": [0, 1, 10, 0],
            }
        )

        result = holder_num_change_ratio_negative_fresh_3d_decay_10d_factor(
            ts_code="000001.SZ",
            start_date="20250410",
            end_date="20250425",
        )

        self.assertAlmostEqual(result.iloc[0]["factor_value"], 0.22)
        self.assertAlmostEqual(result.iloc[1]["factor_value"], 0.22 * 10.0 / 11.0)
        self.assertAlmostEqual(result.iloc[2]["factor_value"], 0.22 / 11.0)
        self.assertTrue(pd.isna(result.iloc[3]["factor_value"]))

    @patch("factors.fundamental.get_a_share_holder_number_daily")
    def test_holder_num_change_ratio_negative_fresh_3d_decay_3d_factor(self, mock_get_holder_daily) -> None:
        mock_get_holder_daily.return_value = pd.DataFrame(
            {
                "trade_date": ["20250410", "20250411", "20250415", "20250416"],
                "ts_code": ["000001.SZ"] * 4,
                "holder_num_change_ratio_negative": [0.20, 0.20, 0.20, 0.20],
                "holder_report_lag_trading_days": [3, 3, 3, 3],
                "holder_announcement_age_trading_days": [0, 1, 3, 4],
            }
        )

        result = holder_num_change_ratio_negative_fresh_3d_decay_3d_factor(
            ts_code="000001.SZ",
            start_date="20250410",
            end_date="20250416",
        )

        self.assertAlmostEqual(result.iloc[0]["factor_value"], 0.20)
        self.assertAlmostEqual(result.iloc[1]["factor_value"], 0.20 * 3.0 / 4.0)
        self.assertAlmostEqual(result.iloc[2]["factor_value"], 0.20 / 4.0)
        self.assertTrue(pd.isna(result.iloc[3]["factor_value"]))

    @patch("factors.fundamental.get_a_share_holder_number_daily")
    def test_holder_num_change_ratio_negative_fresh_3d_decay_5d_factor(self, mock_get_holder_daily) -> None:
        mock_get_holder_daily.return_value = pd.DataFrame(
            {
                "trade_date": ["20250410", "20250411", "20250417", "20250418"],
                "ts_code": ["000001.SZ"] * 4,
                "holder_num_change_ratio_negative": [0.18, 0.18, 0.18, 0.18],
                "holder_report_lag_trading_days": [3, 3, 3, 4],
                "holder_announcement_age_trading_days": [0, 1, 5, 0],
            }
        )

        result = holder_num_change_ratio_negative_fresh_3d_decay_5d_factor(
            ts_code="000001.SZ",
            start_date="20250410",
            end_date="20250418",
        )

        self.assertAlmostEqual(result.iloc[0]["factor_value"], 0.18)
        self.assertAlmostEqual(result.iloc[1]["factor_value"], 0.18 * 5.0 / 6.0)
        self.assertAlmostEqual(result.iloc[2]["factor_value"], 0.18 / 6.0)
        self.assertTrue(pd.isna(result.iloc[3]["factor_value"]))

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
        self.assertIs(get_factor_function("volume_ratio_5d"), volume_ratio_5d_factor)
        self.assertIs(get_factor_function("turnover_rate_f"), turnover_rate_f_factor)
        self.assertIs(get_factor_function("ma20_slope_1d"), ma20_slope_1d_factor)
        self.assertIs(get_factor_function("price_position_120d"), price_position_120d_factor)
        self.assertIs(get_factor_function("low_position_120d"), low_position_120d_factor)
        self.assertIs(get_factor_function("daily_return"), daily_return_factor)
        self.assertIs(get_factor_function("illiq_negative"), illiq_negative_factor)
        self.assertIs(get_factor_function("money_flow_strength_20d"), money_flow_strength_20d_factor)
        self.assertIs(get_factor_function("price_new_low_20d"), price_new_low_20d_factor)
        self.assertIs(get_factor_function("kdj_bullish_divergence_20d"), kdj_bullish_divergence_20d_factor)
        self.assertIs(get_factor_function("kdj_j_turn_up"), kdj_j_turn_up_factor)
        self.assertIs(get_factor_function("kdj_j_turn_up_3d"), kdj_j_turn_up_3d_factor)
        self.assertIs(get_factor_function("kdj_golden_cross"), kdj_golden_cross_factor)
        self.assertIs(get_factor_function("kdj_golden_cross_3d"), kdj_golden_cross_3d_factor)
        self.assertIs(get_factor_function("daily_macd_golden_cross_2d"), daily_macd_golden_cross_2d_factor)
        self.assertIs(get_factor_function("macd_hist_slope_5d"), macd_hist_slope_5d_factor)
        self.assertIs(get_factor_function("macd_zero_axis_strength"), macd_zero_axis_strength_factor)
        self.assertIs(get_factor_function("weekly_kdj_golden_cross_2d"), weekly_kdj_golden_cross_2d_factor)
        self.assertIs(get_factor_function("weekly_macd_golden_cross_2d"), weekly_macd_golden_cross_2d_factor)
        self.assertIs(get_factor_function("kdj_low_zone_cross"), kdj_low_zone_cross_factor)
        self.assertIs(get_factor_function("post_cross_pullback"), post_cross_pullback_factor)
        self.assertIs(get_factor_function("false_breakout_risk_negative"), false_breakout_risk_negative_factor)
        self.assertIs(get_factor_function("ma5_ma10_breakout"), ma5_ma10_breakout_factor)
        self.assertIs(get_factor_function("ma5_ma10_breakout_3d"), ma5_ma10_breakout_3d_factor)
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
        self.assertIs(
            get_factor_function("holder_num_change_ratio_negative_fresh_3d_decay_10d"),
            holder_num_change_ratio_negative_fresh_3d_decay_10d_factor,
        )
        self.assertIs(
            get_factor_function("holder_num_change_ratio_negative_fresh_3d_decay_3d"),
            holder_num_change_ratio_negative_fresh_3d_decay_3d_factor,
        )
        self.assertIs(
            get_factor_function("holder_num_change_ratio_negative_fresh_3d_decay_5d"),
            holder_num_change_ratio_negative_fresh_3d_decay_5d_factor,
        )


if __name__ == "__main__":
    unittest.main()
