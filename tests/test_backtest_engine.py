from __future__ import annotations

import unittest

import pandas as pd

from backtest.engine import calc_benchmark_returns, generate_weights, run_backtest


class BacktestEngineTestCase(unittest.TestCase):
    def test_calc_benchmark_returns_empty_input(self) -> None:
        benchmark_data = pd.DataFrame(columns=["trade_date", "open", "close", "pre_close"])

        result = calc_benchmark_returns(benchmark_data, execution_dates=["20240201"])

        self.assertEqual(list(result.columns), ["trade_date", "benchmark_return"])
        self.assertTrue(result.empty)

    def test_calc_benchmark_returns_uses_close_to_close_on_normal_days(self) -> None:
        benchmark_data = pd.DataFrame(
            {
                "trade_date": ["20240201"],
                "open": [101.0],
                "close": [103.0],
                "pre_close": [100.0],
            }
        )

        result = calc_benchmark_returns(benchmark_data, execution_dates=["20240202"])

        self.assertAlmostEqual(result.loc[0, "benchmark_return"], 103.0 / 100.0 - 1.0)

    def test_calc_benchmark_returns_uses_open_to_close_on_execution_days(self) -> None:
        benchmark_data = pd.DataFrame(
            {
                "trade_date": ["20240201"],
                "open": [101.0],
                "close": [103.0],
                "pre_close": [100.0],
            }
        )

        result = calc_benchmark_returns(benchmark_data, execution_dates=["20240201"])

        self.assertAlmostEqual(result.loc[0, "benchmark_return"], 103.0 / 101.0 - 1.0)

    def test_run_backtest_without_constraints_preserves_baseline_behavior(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201", "20240202"],
                "ts_code": ["A", "A", "A"],
                "open": [10.0, 10.0, 11.0],
                "close": [10.0, 11.0, 12.0],
            }
        )

        returns, holdings = run_backtest(signals, market_data, fee_bps=0.0)

        self.assertAlmostEqual(
            returns.loc[returns["trade_date"] == "20240201", "strategy_return"].iloc[0],
            0.1,
        )
        self.assertAlmostEqual(
            returns.loc[returns["trade_date"] == "20240202", "strategy_return"].iloc[0],
            12.0 / 11.0 - 1.0,
        )
        self.assertEqual(holdings.loc[holdings["trade_date"] == "20240201", "ts_code"].tolist(), ["A"])

    def test_run_backtest_exec_day_weights_drift_to_close(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240131"],
                "ts_code": ["A", "B"],
                "selected": [True, True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240131", "20240201", "20240201"],
                "ts_code": ["A", "B", "A", "B"],
                "open": [10.0, 10.0, 10.0, 10.0],
                "close": [10.0, 10.0, 20.0, 10.0],
            }
        )

        _, holdings = run_backtest(signals, market_data, fee_bps=0.0)

        day_holdings = holdings.loc[holdings["trade_date"] == "20240201"].sort_values("ts_code").reset_index(drop=True)
        self.assertAlmostEqual(day_holdings.loc[0, "weight"], 2.0 / 3.0)
        self.assertAlmostEqual(day_holdings.loc[1, "weight"], 1.0 / 3.0)

    def test_run_backtest_non_exec_day_weights_continue_to_drift(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240131"],
                "ts_code": ["A", "B"],
                "selected": [True, True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240131", "20240201", "20240201", "20240202", "20240202"],
                "ts_code": ["A", "B", "A", "B", "A", "B"],
                "open": [10.0, 10.0, 10.0, 10.0, 20.0, 10.0],
                "close": [10.0, 10.0, 20.0, 10.0, 20.0, 20.0],
            }
        )

        returns, holdings = run_backtest(signals, market_data, fee_bps=0.0)

        day_one = holdings.loc[holdings["trade_date"] == "20240201"].sort_values("ts_code").reset_index(drop=True)
        day_two = holdings.loc[holdings["trade_date"] == "20240202"].sort_values("ts_code").reset_index(drop=True)
        day_two_return = returns.loc[returns["trade_date"] == "20240202", "strategy_return"].iloc[0]

        self.assertAlmostEqual(day_one.loc[0, "weight"], 2.0 / 3.0)
        self.assertAlmostEqual(day_one.loc[1, "weight"], 1.0 / 3.0)
        self.assertAlmostEqual(day_two.loc[0, "weight"], 0.5)
        self.assertAlmostEqual(day_two.loc[1, "weight"], 0.5)
        self.assertAlmostEqual(day_two_return, 1.0 / 3.0)

    def test_run_backtest_slippage_adds_to_cost(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201"],
                "ts_code": ["A", "A"],
                "open": [10.0, 10.0],
                "close": [10.0, 11.0],
            }
        )

        returns, _ = run_backtest(signals, market_data, fee_bps=0.0, slippage_bps=10.0)

        row = returns.loc[returns["trade_date"] == "20240201"].iloc[0]
        self.assertAlmostEqual(row["turnover"], 1.0)
        self.assertAlmostEqual(row["cost"], 0.001)
        self.assertAlmostEqual(row["strategy_return"], 0.1 - 0.001)

    def test_run_backtest_blocked_limit_down_sell_keeps_existing_position(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240229"],
                "ts_code": ["A", "B"],
                "selected": [True, True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": [
                    "20240131",
                    "20240131",
                    "20240201",
                    "20240201",
                    "20240229",
                    "20240229",
                    "20240301",
                    "20240301",
                ],
                "ts_code": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "open": [10.0, 20.0, 10.0, 20.0, 11.0, 20.5, 9.0, 20.0],
                "close": [10.0, 20.0, 11.0, 20.0, 11.0, 20.5, 9.5, 21.0],
                "down_limit": [9.0, 18.0, 9.0, 18.0, 10.0, 19.0, 9.0, 19.0],
            }
        )

        returns, holdings = run_backtest(
            signals,
            market_data,
            fee_bps=0.0,
            block_limit_down_sell=True,
        )

        march_holdings = holdings.loc[holdings["trade_date"] == "20240301", "ts_code"].tolist()
        march_return = returns.loc[returns["trade_date"] == "20240301", "strategy_return"].iloc[0]

        self.assertEqual(march_holdings, ["A"])
        self.assertAlmostEqual(march_return, 9.5 / 9.0 - 1.0)

    def test_run_backtest_block_suspended_blocks_buy_when_field_present(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201"],
                "ts_code": ["A", "A"],
                "open": [10.0, 10.0],
                "close": [10.0, 11.0],
                "suspended": [False, True],
            }
        )

        returns, holdings = run_backtest(
            signals,
            market_data,
            fee_bps=0.0,
            block_suspended=True,
        )

        exec_day = returns.loc[returns["trade_date"] == "20240201"].iloc[0]
        self.assertAlmostEqual(exec_day["turnover"], 0.0)
        self.assertAlmostEqual(exec_day["strategy_return"], 0.0)
        self.assertTrue(holdings.empty)

    def test_run_backtest_block_limit_up_buy_blocks_new_entry(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201"],
                "ts_code": ["A", "A"],
                "open": [10.0, 10.0],
                "close": [10.0, 10.5],
                "up_limit": [11.0, 10.0],
            }
        )

        returns, holdings = run_backtest(
            signals,
            market_data,
            fee_bps=0.0,
            block_limit_up_buy=True,
        )

        exec_day = returns.loc[returns["trade_date"] == "20240201"].iloc[0]
        self.assertAlmostEqual(exec_day["turnover"], 0.0)
        self.assertAlmostEqual(exec_day["strategy_return"], 0.0)
        self.assertTrue(holdings.empty)

    def test_run_backtest_min_amount_gracefully_ignores_missing_field(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [True],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201"],
                "ts_code": ["A", "A"],
                "open": [10.0, 10.0],
                "close": [10.0, 11.0],
            }
        )

        returns, holdings = run_backtest(
            signals,
            market_data,
            fee_bps=0.0,
            min_amount=1_000_000.0,
        )

        exec_day = returns.loc[returns["trade_date"] == "20240201"].iloc[0]
        self.assertAlmostEqual(exec_day["turnover"], 1.0)
        self.assertAlmostEqual(exec_day["strategy_return"], 0.1)
        self.assertEqual(holdings.loc[holdings["trade_date"] == "20240201", "ts_code"].tolist(), ["A"])

    def test_generate_weights_returns_empty_schema_when_no_assets_selected(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": ["20240131"],
                "ts_code": ["A"],
                "selected": [False],
            }
        )
        market_data = pd.DataFrame(
            {
                "trade_date": ["20240131", "20240201"],
                "ts_code": ["A", "A"],
                "open": [10.0, 10.0],
                "close": [10.0, 10.0],
            }
        )

        weights = generate_weights(signals, market_data)

        self.assertEqual(weights.columns.tolist(), ["trade_date", "ts_code", "target_weight", "signal_date"])
        self.assertTrue(weights.empty)


if __name__ == "__main__":
    unittest.main()
