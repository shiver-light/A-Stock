from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.research_accumulation import (
    _filter_dataset_by_universe_membership,
    _filter_growth_board_membership,
    _filter_growth_board_ts_codes,
    _sample_dates,
)
from analysis.accumulation_research import (
    DEFAULT_ACCUMULATION_FEATURES,
    apply_accumulation_score,
    build_accumulation_dataset,
    evaluate_condition_ladder,
    filter_research_samples,
    fit_accumulation_score_model,
    score_bucket_analysis,
    single_factor_analysis,
    split_by_time,
    summarize_feature_power,
)


class AccumulationResearchTestCase(unittest.TestCase):
    def _market_data(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=320, freq="D").strftime("%Y%m%d").tolist()
        frames = []
        for ts_code, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            close = []
            for index in range(320):
                if ts_code == "000001.SZ":
                    value = base + min(index, 160) * 0.01
                    if 220 <= index < 240:
                        value = 11.0
                    if 240 <= index < 261:
                        value = 11.0 + (index - 240) * 0.15
                else:
                    value = base - min(index, 200) * 0.02
                    if 220 <= index < 261:
                        value = 16.0 - (index - 220) * 0.03
                close.append(max(value, 1.0))
            frame = pd.DataFrame(
                {
                    "trade_date": dates,
                    "ts_code": ts_code,
                    "open": close,
                    "high": [value * 1.02 for value in close],
                    "low": [value * 0.98 for value in close],
                    "close": close,
                    "pre_close": [close[0], *close[:-1]],
                    "vol": [1000 + (index % 20) * 10 for index in range(320)],
                    "amount": [100000 + (index % 20) * 1000 for index in range(320)],
                }
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _turnover_data(self) -> pd.DataFrame:
        market = self._market_data()
        return market.loc[:, ["trade_date", "ts_code"]].assign(turnover_rate_f=5.0)

    def test_build_accumulation_dataset_creates_labels_and_features(self) -> None:
        dataset = build_accumulation_dataset(
            self._market_data(),
            self._turnover_data(),
            industry_map=pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "industry": ["银行", "银行"]}),
        )

        self.assertIn("future_20d_max_gain", dataset.columns)
        self.assertIn("price_position_120d", dataset.columns)
        self.assertIn("efficiency_ratio_20d", dataset.columns)
        self.assertIn("relative_industry_20d", dataset.columns)
        self.assertTrue(set(dataset["label"].dropna().unique()).issubset({0.0, 1.0}))

    def test_factor_analysis_condition_ladder_and_score_model(self) -> None:
        dataset = build_accumulation_dataset(
            self._market_data(),
            self._turnover_data(),
            industry_map=pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "industry": ["银行", "银行"]}),
        )
        samples = filter_research_samples(dataset)
        splits = split_by_time(samples, train_end="20201231", validation_end="20211231")
        stats = single_factor_analysis(samples, DEFAULT_ACCUMULATION_FEATURES[:5], n_quantiles=3)
        power = summarize_feature_power(stats)
        model = fit_accumulation_score_model(samples, power, max_features=3)
        scored = apply_accumulation_score(samples, model)
        buckets = score_bucket_analysis(scored)
        ladder = evaluate_condition_ladder(
            samples,
            [{"name": "低位", "feature": "price_position_120d", "op": "lte", "value": 0.3}],
        )

        self.assertIn("train", splits)
        self.assertFalse(stats.empty)
        self.assertIn("score", power.columns)
        self.assertIn("accumulation_score", scored.columns)
        self.assertIn("condition", ladder.columns)
        self.assertIsInstance(buckets, pd.DataFrame)

    def test_sample_dates_supports_monthly_and_weekly(self) -> None:
        monthly = _sample_dates("20240101", "20240315", "monthly")
        weekly = _sample_dates("20240101", "20240115", "weekly")

        self.assertEqual(monthly, ["20240131", "20240229", "20240315"])
        self.assertEqual(weekly[-1], "20240115")

    def test_filter_dataset_by_universe_membership_uses_historical_intervals(self) -> None:
        dataset = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240115", "20240201", "20240215"],
                "ts_code": ["000001.SZ"] * 4,
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        )
        membership = pd.DataFrame(
            {
                "as_of_date": ["20240110"],
                "ts_code": ["000001.SZ"],
                "source_universe": ["hs300"],
            }
        )

        result = _filter_dataset_by_universe_membership(dataset, membership)

        self.assertEqual(result["trade_date"].tolist(), ["20240115", "20240201", "20240215"])
        self.assertEqual(result["source_universe"].unique().tolist(), ["hs300"])

    @patch("scripts.research_accumulation.get_stock_basic_history")
    def test_filter_growth_board_ts_codes_excludes_chinext_and_star(self, mock_get_stock_basic_history) -> None:
        mock_get_stock_basic_history.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600001.SH"],
                "market": ["主板", "创业板", "科创板", "主板"],
            }
        )

        result = _filter_growth_board_ts_codes(
            ["000001.SZ", "300001.SZ", "688001.SH", "600001.SH"],
            refresh=False,
        )

        self.assertEqual(result, ["000001.SZ", "600001.SH"])

    @patch("scripts.research_accumulation.get_stock_basic_history")
    def test_filter_growth_board_membership_keeps_main_board_rows(self, mock_get_stock_basic_history) -> None:
        mock_get_stock_basic_history.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600001.SH"],
                "market": ["主板", "创业板", "科创板", "主板"],
            }
        )
        membership = pd.DataFrame(
            {
                "as_of_date": ["20240131"] * 4,
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600001.SH"],
                "source_universe": ["all_a"] * 4,
            }
        )

        result = _filter_growth_board_membership(membership, refresh=False)

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "600001.SH"])


if __name__ == "__main__":
    unittest.main()
