from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.main import (
    _apply_external_regime_filter,
    _apply_market_regime_filter,
    _apply_signal_filters,
    _build_market_regime_flags,
    build_factor_panel,
    run_minimal_pipeline,
    run_recommendation_pipeline,
)


class PipelineMainTestCase(unittest.TestCase):
    def _benchmark_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000300.SH", "000300.SH"],
                "open": [10.0, 10.1],
                "close": [10.1, 10.2],
                "pre_close": [10.0, 10.1],
            }
        )

    def _market_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "open": [10.0, 20.0, 10.5, 20.5],
                "close": [10.5, 20.5, 10.8, 20.8],
                "amount": [1000000.0, 2000000.0, 1100000.0, 2100000.0],
            }
        )

    def _selected(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "score": [1.0, 0.5],
                "rank": [1, 2],
                "selected": [True, False],
            }
        )

    def test_apply_signal_filters_uses_same_date_quantiles(self) -> None:
        factor_panel = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240102", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000001.SZ"],
                "accumulation_score": [0.9, 0.6, 0.2, 0.4],
                "distribution_score": [0.2, 0.8, 0.1, 0.3],
            }
        )

        result = _apply_signal_filters(
            factor_panel,
            [
                {"factor": "accumulation_score", "op": "quantile_gte", "value": 0.6},
                {"factor": "distribution_score", "op": "quantile_lte", "value": 0.7},
            ],
        )

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "000001.SZ"])

    def test_apply_signal_filters_rejects_unknown_operator(self) -> None:
        factor_panel = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "accumulation_score": [0.9]}
        )

        with self.assertRaisesRegex(ValueError, "Unsupported signal filter op"):
            _apply_signal_filters(
                factor_panel,
                [{"factor": "accumulation_score", "op": "bad_op", "value": 0.6}],
            )

    def test_apply_signal_filters_supports_absolute_thresholds(self) -> None:
        factor_panel = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "volume_ratio_5d": [2.4, 2.8, 3.1],
                "turnover_rate_f": [2.9, 5.0, 11.0],
                "daily_return": [0.03, 0.06, 0.04],
            }
        )

        result = _apply_signal_filters(
            factor_panel,
            [
                {"factor": "volume_ratio_5d", "op": "gte", "value": 2.5},
                {"factor": "turnover_rate_f", "op": "between", "min_value": 3.0, "max_value": 10.0},
                {"factor": "daily_return", "op": "lte", "value": 0.08},
            ],
        )

        self.assertEqual(result["ts_code"].tolist(), ["000002.SZ"])

    def test_build_market_regime_flags_uses_historical_benchmark_returns(self) -> None:
        benchmark = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 101.0, 102.0, 101.0],
            }
        )

        result = _build_market_regime_flags(benchmark, {"min_return_2d": 0.0})

        self.assertEqual(result["market_regime_allowed"].tolist(), [False, False, True, True])

    def test_apply_market_regime_filter_sets_blocked_dates_to_unselected(self) -> None:
        selection = pd.DataFrame(
            {
                "trade_date": ["20240103", "20240104"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "score": [1.0, 1.0],
                "rank": [1, 1],
                "selected": [True, True],
            }
        )
        benchmark = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 101.0, 102.0, 99.0],
            }
        )

        result, flags = _apply_market_regime_filter(selection, benchmark, {"min_return_2d": 0.0})

        self.assertEqual(flags["market_regime_allowed"].tolist(), [False, False, True, False])
        self.assertEqual(result["selected"].tolist(), [True, False])

    @patch("pipeline.main.load_external_regime_data")
    def test_apply_external_regime_filter_blocks_missing_or_false_dates(self, mock_load) -> None:
        mock_load.return_value = pd.DataFrame(
            {
                "trade_date": ["20240103", "20240104"],
                "us_risk_on": [1, 0],
            }
        )
        selection = pd.DataFrame(
            {
                "trade_date": ["20240103", "20240104", "20240105"],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "score": [1.0, 1.0, 1.0],
                "rank": [1, 1, 1],
                "selected": [True, True, True],
            }
        )

        result, flags = _apply_external_regime_filter(
            selection,
            {"path": "research/us_market_regime.csv", "allowed_col": "us_risk_on"},
        )

        self.assertEqual(flags["external_regime_allowed"].tolist(), [True, False])
        self.assertEqual(result["selected"].tolist(), [True, False, False])

    def test_build_market_regime_flags_supports_relative_strength_returns(self) -> None:
        benchmark = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 102.0, 104.0, 103.0],
            }
        )
        reference = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 101.0, 100.0, 104.0],
            }
        )

        result = _build_market_regime_flags(
            benchmark,
            {"relative_strength": {"reference_code": "000300.SH", "min_return_2d": 0.0}},
            reference_index_data={"000300.SH": reference},
        )

        self.assertEqual(result["market_regime_allowed"].tolist(), [False, False, True, False])

    def test_build_market_regime_flags_supports_relative_strength_ma(self) -> None:
        benchmark = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 103.0, 104.0, 105.0],
            }
        )
        reference = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102", "20240103", "20240104"],
                "close": [100.0, 101.0, 101.0, 110.0],
            }
        )

        result = _build_market_regime_flags(
            benchmark,
            {"relative_strength": {"reference_code": "000300.SH", "ratio_above_ma": 2}},
            reference_index_data={"000300.SH": reference},
        )

        self.assertEqual(result["market_regime_allowed"].tolist(), [False, True, True, False])

    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240103", "top_n": 2, "top_stocks": [{"ts_code": "000001.SZ"}]},
    )
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ"])
    def test_run_recommendation_pipeline_uses_daily_latest_selection(
        self,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_latest_selection,
    ) -> None:
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "turnover_mean_20d": [0.1, 0.2, 0.3, 0.4],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "score": [0.1, 0.2, 0.3, 0.4],
            }
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "score": [0.1, 0.2, 0.3, 0.4],
                "rank": [2, 1, 2, 1],
            }
        )
        mock_top_n_selection.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "score": [0.1, 0.2, 0.3, 0.4],
                "rank": [2, 1, 2, 1],
                "selected": [True, True, True, True],
            }
        )

        result = run_recommendation_pipeline(
            universe_name="hs300",
            start_date="20240101",
            end_date="20240131",
            top_n=2,
            factor_config={"turnover_mean_20d": 1.0},
        )

        self.assertIn("latest_selection", result)
        mock_resolve.assert_called_once_with(
            ts_codes=None,
            universe_name="hs300",
            as_of_date="20240131",
            signal_dates=None,
        )
        mock_latest_selection.assert_called_once()
        self.assertEqual(mock_latest_selection.call_args.kwargs["as_of_date"], "20240103")

    def test_build_factor_panel_uses_registered_factor_function(self) -> None:
        def fake_factor(*, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            self.assertEqual(ts_code, "000001.SZ")
            self.assertEqual(start_date, "20240101")
            self.assertEqual(end_date, "20240131")
            return pd.DataFrame(
                {
                    "trade_date": ["20240102"],
                    "ts_code": ["000001.SZ"],
                    "factor_value": [0.12],
                }
            )

        with patch("pipeline.main.get_factor_function", return_value=fake_factor) as mock_get_factor:
            result = build_factor_panel(
                ts_codes=["000001.SZ"],
                start_date="20240101",
                end_date="20240131",
                factor_config={"momentum_60d": 1.0},
            )

        mock_get_factor.assert_called_once_with("momentum_60d")
        self.assertEqual(result.columns.tolist(), ["trade_date", "ts_code", "momentum_60d"])
        self.assertEqual(result.iloc[0]["momentum_60d"], 0.12)

    def test_build_factor_panel_unknown_factor_raises_clear_error(self) -> None:
        with patch("pipeline.main.get_factor_function", side_effect=KeyError("Unknown factor: unknown_factor")):
            with self.assertRaisesRegex(ValueError, "Unsupported factor in factor_config: unknown_factor"):
                build_factor_panel(
                    ts_codes=["000001.SZ"],
                    start_date="20240101",
                    end_date="20240131",
                    factor_config={"unknown_factor": 1.0},
                )

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240102", "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.01})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.02})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01]}),
            pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "weight": [1.0]}),
        ),
    )
    @patch("pipeline.main._build_market_panel")
    @patch("pipeline.main._build_raw_factor_panel")
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ"])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_returns_data_diagnostics_when_enabled(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_raw_factor_panel,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "ts_code": ["000001.SZ"],
                "return_20d": [0.1],
                "volatility_20d": [0.2],
            }
        )
        mock_build_raw_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "return_20d": [0.1, None, 0.2],
                "volatility_20d": [0.3, 0.4, None],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0]}
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0], "rank": [1]}
        )
        mock_top_n_selection.return_value = self._selected()
        mock_build_market_panel.return_value = self._market_panel()

        result = run_minimal_pipeline(
            ts_codes=["000001.SZ", "000002.SZ"],
            start_date="20240101",
            end_date="20240131",
            factor_config={"return_20d": 1.0, "volatility_20d": -1.0},
            enable_data_diagnostics=True,
        )

        self.assertIn("diagnostics", result)
        self.assertIn("universe_count_by_date", result["diagnostics"])
        self.assertIn("factor_coverage", result["diagnostics"])
        self.assertIn("complete_case_count_by_date", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["universe_count_by_date"]["date_count"], 2)
        self.assertEqual(
            result["diagnostics"]["complete_case_count_by_date"]["per_date"][0]["complete_case_count"],
            1,
        )

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240102", "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.01})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.02})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01]}),
            pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "weight": [1.0]}),
        ),
    )
    @patch("pipeline.main._build_market_panel")
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ"])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_default_behavior_compatible(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "return_20d": [0.1, 0.2],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0]}
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0], "rank": [1]}
        )
        mock_top_n_selection.return_value = self._selected()
        mock_build_market_panel.return_value = self._market_panel()

        result = run_minimal_pipeline(
            ts_codes=["000001.SZ", "000002.SZ"],
            start_date="20240101",
            end_date="20240131",
            factor_config={"return_20d": 1.0},
        )

        self.assertIn("factor_data", result)
        self.assertIn("report_text", result)
        self.assertNotIn("factor_diagnostics", result)
        self.assertNotIn("factor_report_text", result)

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240102", "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.01})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.02})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01]}),
            pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "weight": [1.0]}),
        ),
    )
    @patch("pipeline.main._build_market_panel")
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ", "000003.SZ"])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_applies_signal_filters_before_scoring(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "position_safety_60d": [0.3, 0.8, 0.7],
                "money_flow_strength_20d": [0.9, 0.4, 0.8],
                "close_near_high_on_high_amount_20d": [0.2, 0.6, 0.9],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0]}
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0], "rank": [1]}
        )
        mock_top_n_selection.return_value = self._selected()
        mock_build_market_panel.return_value = self._market_panel()

        result = run_minimal_pipeline(
            ts_codes=["000001.SZ", "000002.SZ", "000003.SZ"],
            start_date="20240101",
            end_date="20240131",
            factor_config={"position_safety_60d": 1.0},
            signal_filters=[
                {"factor": "money_flow_strength_20d", "op": "quantile_gte", "value": 0.6},
                {"factor": "close_near_high_on_high_amount_20d", "op": "quantile_lte", "value": 0.7},
            ],
        )

        self.assertIn("filtered_factor_data", result)
        self.assertEqual(result["filtered_factor_data"]["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(
            mock_build_factor_panel.call_args.kwargs["factor_config"],
            {
                "position_safety_60d": 1.0,
                "money_flow_strength_20d": 1.0,
                "close_near_high_on_high_amount_20d": 1.0,
            },
        )
        scored_input = mock_combine_scores.call_args.args[0]
        self.assertEqual(scored_input["ts_code"].tolist(), ["000001.SZ"])

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240102", "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.01})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.02})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01]}),
            pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "weight": [1.0]}),
        ),
    )
    @patch("pipeline.main._build_market_panel")
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ"])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_returns_factor_diagnostics_when_enabled(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "return_20d": [1.0, 2.0, 1.0, 2.0],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "score": [0.6, 0.4],
            }
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "score": [0.6, 0.4],
                "rank": [1, 2],
            }
        )
        mock_top_n_selection.return_value = self._selected()
        mock_build_market_panel.return_value = self._market_panel()

        result = run_minimal_pipeline(
            ts_codes=["000001.SZ", "000002.SZ"],
            start_date="20240101",
            end_date="20240131",
            factor_config={"return_20d": 1.0},
            enable_factor_diagnostics=True,
            analysis_horizons=(1,),
        )

        self.assertIn("factor_diagnostics", result)
        self.assertIn("factor_report_text", result)
        self.assertIn("return_20d", result["factor_diagnostics"])
        self.assertEqual(len(result["factor_diagnostics"]["return_20d"]), 1)
        self.assertIn("return_20d", result["factor_report_text"])

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": "20240102", "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.01})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.02})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame({"trade_date": ["20240102"], "benchmark_return": [0.0]}),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame({"trade_date": ["20240102"], "strategy_return": [0.01]}),
            pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "weight": [1.0]}),
        ),
    )
    @patch("pipeline.main._build_market_panel")
    @patch("pipeline.main.top_n_selection")
    @patch("pipeline.main.rank_signal")
    @patch("pipeline.main.combine_factor_scores")
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ", "000002.SZ"])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_passes_backtest_config(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "return_20d": [0.1, 0.2],
            }
        )
        mock_combine_scores.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0]}
        )
        mock_rank_signal.return_value = pd.DataFrame(
            {"trade_date": ["20240102"], "ts_code": ["000001.SZ"], "score": [1.0], "rank": [1]}
        )
        mock_top_n_selection.return_value = self._selected()
        mock_build_market_panel.return_value = self._market_panel()

        result = run_minimal_pipeline(
            ts_codes=["000001.SZ", "000002.SZ"],
            start_date="20240101",
            end_date="20240131",
            factor_config={"return_20d": 1.0},
            backtest_config={"slippage_bps": 12.0, "min_amount": 1000000.0},
        )

        self.assertIn("performance", result)
        _, kwargs = mock_run_backtest.call_args
        self.assertEqual(kwargs["slippage_bps"], 12.0)
        self.assertEqual(kwargs["min_amount"], 1000000.0)

    @patch("pipeline.main.render_strategy_report_text", return_value="strategy report")
    @patch("pipeline.main.format_strategy_report", return_value={"report": "ok"})
    @patch(
        "pipeline.main.format_latest_selection",
        return_value={"as_of_date": None, "top_n": 1, "top_stocks": []},
    )
    @patch("pipeline.main.calc_relative_performance", return_value={"excess_cumulative_return": 0.0})
    @patch("pipeline.main.calc_performance", return_value={"cumulative_return": 0.0})
    @patch(
        "pipeline.main.attach_benchmark",
        return_value=pd.DataFrame(columns=["trade_date", "strategy_return", "benchmark_return"]),
    )
    @patch(
        "pipeline.main.calc_benchmark_returns",
        return_value=pd.DataFrame(columns=["trade_date", "benchmark_return"]),
    )
    @patch(
        "pipeline.main.run_backtest",
        return_value=(
            pd.DataFrame(columns=["trade_date", "strategy_return"]),
            pd.DataFrame(columns=["trade_date", "ts_code", "weight"]),
        ),
    )
    @patch("pipeline.main._build_market_panel", return_value=pd.DataFrame(columns=["trade_date", "ts_code", "open", "close"]))
    @patch("pipeline.main.top_n_selection", return_value=pd.DataFrame(columns=["trade_date", "ts_code", "score", "rank", "selected"]))
    @patch("pipeline.main.rank_signal", return_value=pd.DataFrame(columns=["trade_date", "ts_code", "score", "rank"]))
    @patch("pipeline.main.combine_factor_scores", return_value=pd.DataFrame(columns=["trade_date", "ts_code", "score"]))
    @patch("pipeline.main.build_factor_panel", return_value=pd.DataFrame(columns=["trade_date", "ts_code", "return_20d"]))
    @patch("pipeline.main._resolve_ts_codes", return_value=[])
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame(columns=["signal_date", "execution_date"]),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_empty_factor_panel_with_diagnostics_does_not_fail(
        self,
        mock_index_daily,
        mock_rebalance,
        mock_resolve,
        mock_build_factor_panel,
        mock_combine_scores,
        mock_rank_signal,
        mock_top_n_selection,
        mock_build_market_panel,
        mock_run_backtest,
        mock_benchmark_returns,
        mock_attach_benchmark,
        mock_performance,
        mock_relative_performance,
        mock_latest_selection,
        mock_format_report,
        mock_render_text,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()

        result = run_minimal_pipeline(
            ts_codes=[],
            start_date="20240101",
            end_date="20240131",
            factor_config={"return_20d": 1.0},
            enable_factor_diagnostics=True,
            analysis_horizons=(5,),
        )

        self.assertEqual(result["factor_diagnostics"]["return_20d"], [])
        self.assertEqual(result["factor_report_text"]["return_20d"], "")

    @patch("pipeline.main._resolve_ts_codes", return_value=["000001.SZ"])
    @patch("pipeline.main.build_factor_panel")
    @patch("pipeline.main._build_market_panel")
    @patch(
        "pipeline.main.get_rebalance_schedule",
        return_value=pd.DataFrame({"signal_date": ["20240102"], "execution_date": ["20240103"]}),
    )
    @patch("pipeline.main.get_a_share_index_daily")
    def test_run_minimal_pipeline_all_nan_absorption_factor_does_not_break_schema(
        self,
        mock_index_daily,
        mock_rebalance_schedule,
        mock_build_market_panel,
        mock_build_factor_panel,
        mock_resolve,
    ) -> None:
        mock_index_daily.return_value = self._benchmark_data()
        mock_build_market_panel.return_value = self._market_panel()
        mock_build_factor_panel.return_value = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "down_day_absorption_20d": [float("nan"), float("nan")],
                "position_safety_60d": [0.1, 0.2],
            }
        )

        result = run_minimal_pipeline(
            universe_name="zz1000",
            start_date="20240101",
            end_date="20240131",
            top_n=50,
            factor_config={
                "down_day_absorption_20d": 0.7,
                "position_safety_60d": 0.3,
            },
            enable_factor_diagnostics=True,
            analysis_horizons=(5,),
        )

        self.assertEqual(result["selected_signals"].columns.tolist(), ["trade_date", "ts_code", "score", "rank", "selected"])
        self.assertTrue(result["selected_signals"].empty)
        self.assertEqual(result["factor_diagnostics"]["down_day_absorption_20d"], [])
        self.assertEqual(result["latest_selection"]["top_stocks"], [])


if __name__ == "__main__":
    unittest.main()
