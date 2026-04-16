from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.main import run_minimal_pipeline


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


if __name__ == "__main__":
    unittest.main()
