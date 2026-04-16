"""Research result summarization."""

from __future__ import annotations

import pandas as pd


def build_research_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in results:
        config = item["config"]
        performance = item["performance"]
        rows.append(
            {
                "name": item["name"],
                "universe_name": config.get("universe_name", "custom"),
                "top_n": config.get("top_n"),
                "benchmark_code": config.get("benchmark_code", "000300.SH"),
                "factor_config": config.get("factor_config"),
                "cumulative_return": performance.get("cumulative_return"),
                "annual_return": performance.get("annual_return"),
                "max_drawdown": performance.get("max_drawdown"),
                "sharpe": performance.get("sharpe"),
                "benchmark_cumulative_return": performance.get("benchmark_cumulative_return"),
                "excess_cumulative_return": performance.get("excess_cumulative_return"),
                "excess_annual_return": performance.get("excess_annual_return"),
            }
        )
    return pd.DataFrame(rows)


def sort_research_summary(summary: pd.DataFrame, by: str = "sharpe", ascending: bool = False) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    if by not in summary.columns:
        raise ValueError(f"Unknown sort column: {by}")
    return summary.sort_values(by=by, ascending=ascending).reset_index(drop=True)
