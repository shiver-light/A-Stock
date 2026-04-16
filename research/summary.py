"""Research result summarization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.storage import read_json


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
