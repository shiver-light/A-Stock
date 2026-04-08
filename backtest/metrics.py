"""Performance metrics for daily strategy return series."""

from __future__ import annotations

import math

import pandas as pd


def calc_performance(
    strategy_returns: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    return_col: str = "strategy_return",
) -> dict[str, float]:
    if strategy_returns.empty:
        return {
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    data = strategy_returns.loc[:, [date_col, return_col]].copy()
    data = data.sort_values(date_col).reset_index(drop=True)
    daily_returns = data[return_col].fillna(0.0)

    equity = (1.0 + daily_returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    num_days = len(daily_returns)
    annual_return = float(equity.iloc[-1] ** (252 / num_days) - 1.0) if num_days > 0 else 0.0

    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    volatility = float(daily_returns.std(ddof=0))
    sharpe = 0.0
    if volatility > 0:
        sharpe = float((daily_returns.mean() / volatility) * math.sqrt(252))

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }
