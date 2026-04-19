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
    performance: dict[str, object],
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
            "mean_daily_turnover": performance.get("mean_daily_turnover", 0.0),
            "mean_rebalance_turnover": performance.get("mean_rebalance_turnover", 0.0),
            "median_rebalance_turnover": performance.get("median_rebalance_turnover", 0.0),
            "max_rebalance_turnover": performance.get("max_rebalance_turnover", 0.0),
            "positive_month_ratio": performance.get("positive_month_ratio", 0.0),
        },
        "benchmark": {
            "benchmark_code": config.get("benchmark_code"),
            "benchmark_cumulative_return": performance.get("benchmark_cumulative_return", 0.0),
            "excess_cumulative_return": performance.get("excess_cumulative_return", 0.0),
            "excess_annual_return": performance.get("excess_annual_return", 0.0),
            "positive_excess_month_ratio": performance.get("positive_excess_month_ratio", 0.0),
        },
        "robustness": {
            "rolling_5m_excess_return": performance.get("rolling_5m_excess_return", []),
            "latest_rolling_5m_excess_return": performance.get("latest_rolling_5m_excess_return", 0.0),
            "mean_rolling_5m_excess_return": performance.get("mean_rolling_5m_excess_return", 0.0),
            "worst_rolling_5m_excess_return": performance.get("worst_rolling_5m_excess_return", 0.0),
            "rolling_5m_sharpe": performance.get("rolling_5m_sharpe", []),
            "latest_rolling_5m_sharpe": performance.get("latest_rolling_5m_sharpe", 0.0),
            "mean_rolling_5m_sharpe": performance.get("mean_rolling_5m_sharpe", 0.0),
            "worst_rolling_5m_sharpe": performance.get("worst_rolling_5m_sharpe", 0.0),
        },
        "risk": [
            "基于历史数据，不保证未来收益。",
            "可能存在回撤和风格失效风险。",
            "当前MVP未建模停牌、涨跌停和滑点约束。",
            "当前benchmark使用指数日线收益作为对照，不代表可完全复制的指数投资结果。",
        ],
    }


def render_strategy_report_text(report: dict[str, object]) -> str:
    universe = report.get("universe", [])
    selection_logic = report.get("selection_logic", "")
    factor_config = report.get("factor_config", {})
    top_stocks = report.get("top_stocks", [])
    backtest_summary = report.get("backtest_summary", {})
    benchmark = report.get("benchmark", {})
    robustness = report.get("robustness", {})
    risks = report.get("risk", [])

    factor_lines = [
        f"{name}: {weight}"
        for name, weight in factor_config.items()
    ]
    stock_lines = [
        f"{item['rank']}. {item['ts_code']} score={item['score']:.6f}"
        for item in top_stocks
    ]

    lines = [
        "universe:",
        ", ".join(universe) if universe else "N/A",
        "",
        "selection logic:",
        str(selection_logic),
        "",
        "factor config:",
    ]
    lines.extend(factor_lines or ["N/A"])
    lines.extend(
        [
            "",
            "top stocks:",
        ]
    )
    lines.extend(stock_lines or ["N/A"])
    lines.extend(
        [
            "",
            "backtest summary:",
            f"strategy cumulative return: {backtest_summary.get('cumulative_return', 0.0):.6f}",
            f"strategy annual return: {backtest_summary.get('annual_return', 0.0):.6f}",
            f"max drawdown: {backtest_summary.get('max_drawdown', 0.0):.6f}",
            f"sharpe: {backtest_summary.get('sharpe', 0.0):.6f}",
            f"mean daily turnover: {backtest_summary.get('mean_daily_turnover', 0.0):.6f}",
            f"mean rebalance turnover: {backtest_summary.get('mean_rebalance_turnover', 0.0):.6f}",
            f"median rebalance turnover: {backtest_summary.get('median_rebalance_turnover', 0.0):.6f}",
            f"max rebalance turnover: {backtest_summary.get('max_rebalance_turnover', 0.0):.6f}",
            f"positive month ratio: {backtest_summary.get('positive_month_ratio', 0.0):.6f}",
            "",
            "benchmark summary:",
            f"benchmark code: {benchmark.get('benchmark_code')}",
            f"benchmark cumulative return: {benchmark.get('benchmark_cumulative_return', 0.0):.6f}",
            f"excess cumulative return: {benchmark.get('excess_cumulative_return', 0.0):.6f}",
            f"excess annual return: {benchmark.get('excess_annual_return', 0.0):.6f}",
            f"positive excess month ratio: {benchmark.get('positive_excess_month_ratio', 0.0):.6f}",
            "",
            "robustness summary:",
            f"latest rolling 5m excess return: {robustness.get('latest_rolling_5m_excess_return', 0.0):.6f}",
            f"mean rolling 5m excess return: {robustness.get('mean_rolling_5m_excess_return', 0.0):.6f}",
            f"worst rolling 5m excess return: {robustness.get('worst_rolling_5m_excess_return', 0.0):.6f}",
            f"latest rolling 5m sharpe: {robustness.get('latest_rolling_5m_sharpe', 0.0):.6f}",
            f"mean rolling 5m sharpe: {robustness.get('mean_rolling_5m_sharpe', 0.0):.6f}",
            f"worst rolling 5m sharpe: {robustness.get('worst_rolling_5m_sharpe', 0.0):.6f}",
            "",
            "risk:",
        ]
    )
    lines.extend(risks or ["N/A"])
    return "\n".join(lines)
