from __future__ import annotations

import unittest

import pandas as pd

from backtest.engine import run_backtest


class BacktestEngineTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
