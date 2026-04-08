"""Minimal report formatting for strategy outputs."""

from __future__ import annotations

import pandas as pd


def format_latest_selection(
    selection: pd.DataFrame,
    *,
    top_n: int,
    as_of_date: str | None = None,
) -> dict[str, object]:
    if selection.empty:
        return {
            "as_of_date": as_of_date,
            "top_n": top_n,
            "top_stocks": [],
        }

    data = selection.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    latest_date = as_of_date or str(data["trade_date"].max())
    latest = data.loc[data["trade_date"] == latest_date].copy()
    latest = latest.sort_values(["selected", "rank", "ts_code"], ascending=[False, True, True])
    latest = latest.head(top_n)

    top_stocks = latest.loc[:, ["ts_code", "score", "rank", "selected"]].to_dict(orient="records")
    return {
        "as_of_date": latest_date,
        "top_n": top_n,
        "top_stocks": top_stocks,
    }


def format_strategy_report(
    latest_selection: dict[str, object],
    performance: dict[str, float],
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "universe": config.get("universe", []),
        "selection_logic": config.get("selection_logic", "factor score ranking"),
        "factor_config": config.get("factor_config", {}),
        "top_stocks": latest_selection.get("top_stocks", []),
        "backtest_summary": {
            "cumulative_return": performance.get("cumulative_return", 0.0),
            "annual_return": performance.get("annual_return", 0.0),
            "max_drawdown": performance.get("max_drawdown", 0.0),
            "sharpe": performance.get("sharpe", 0.0),
        },
        "benchmark": {
            "benchmark_code": config.get("benchmark_code"),
            "benchmark_cumulative_return": performance.get("benchmark_cumulative_return", 0.0),
            "excess_cumulative_return": performance.get("excess_cumulative_return", 0.0),
            "excess_annual_return": performance.get("excess_annual_return", 0.0),
        },
        "risk": [
            "基于历史数据，不保证未来收益。",
            "可能存在回撤和风格失效风险。",
            "当前MVP未建模停牌、涨跌停和滑点约束。",
            "当前benchmark使用指数日线收益作为对照，不代表可完全复制的指数投资结果。",
        ],
    }
