"""Research result summarization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.storage import read_json


SUMMARY_METRIC_COLUMNS = (
    "cumulative_return",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "benchmark_cumulative_return",
    "excess_cumulative_return",
    "excess_annual_return",
    "mean_daily_turnover",
    "mean_rebalance_turnover",
    "median_rebalance_turnover",
    "max_rebalance_turnover",
    "positive_month_ratio",
    "positive_excess_month_ratio",
    "latest_rolling_5m_excess_return",
    "mean_rolling_5m_excess_return",
    "worst_rolling_5m_excess_return",
    "latest_rolling_5m_sharpe",
    "mean_rolling_5m_sharpe",
    "worst_rolling_5m_sharpe",
    "latest_forward_1d_max_gain",
    "mean_forward_1d_max_gain",
    "best_forward_1d_max_gain",
    "latest_forward_3d_max_gain",
    "mean_forward_3d_max_gain",
    "best_forward_3d_max_gain",
    "latest_forward_7d_max_gain",
    "mean_forward_7d_max_gain",
    "best_forward_7d_max_gain",
    "latest_forward_1w_max_gain",
    "mean_forward_1w_max_gain",
    "best_forward_1w_max_gain",
)


def _json_summary_field(value: object) -> str:
    if value in (None, {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_research_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in results:
        config = item["config"]
        performance = item["performance"]
        row = {
            "name": item["name"],
            "universe_name": config.get("universe_name", "custom"),
            "top_n": config.get("top_n"),
            "benchmark_code": config.get("benchmark_code", "000300.SH"),
            "factor_config": config.get("factor_config"),
            "signal_filters": config.get("signal_filters", []),
            "market_regime_filter": _json_summary_field(config.get("market_regime_filter")),
            "external_regime_filter": _json_summary_field(config.get("external_regime_filter")),
        }
        for metric_name in SUMMARY_METRIC_COLUMNS:
            row[metric_name] = performance.get(metric_name)
        rows.append(row)
    return pd.DataFrame(rows)


def sort_research_summary(summary: pd.DataFrame, by: str = "sharpe", ascending: bool = False) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    if by not in summary.columns:
        raise ValueError(f"Unknown sort column: {by}")
    return summary.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def rebuild_summary_from_disk(run_dir: str | Path) -> pd.DataFrame:
    run_path = Path(run_dir)
    experiments_dir = run_path / "experiments"
    if not experiments_dir.exists():
        return pd.DataFrame()

    results: list[dict[str, object]] = []
    for experiment_dir in sorted(experiments_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        status_path = experiment_dir / "status.json"
        config_path = experiment_dir / "config.json"
        metrics_path = experiment_dir / "metrics.json"
        if not status_path.exists() or not config_path.exists() or not metrics_path.exists():
            continue

        status = read_json(status_path)
        if status.get("status") != "completed":
            continue
        config = read_json(config_path)
        metrics = read_json(metrics_path)
        results.append(
            {
                "name": experiment_dir.name,
                "config": config,
                "performance": metrics,
            }
        )

    return build_research_summary(results)
