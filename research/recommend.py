"""Daily recommendation generation from completed research runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import run_minimal_pipeline
from research.storage import read_json


def _config_signature(config: dict[str, object]) -> str:
    signature_payload = {
        "universe_name": config.get("universe_name"),
        "ts_codes": sorted(config.get("ts_codes", []) or []),
        "top_n": config.get("top_n"),
        "benchmark_code": config.get("benchmark_code", "000300.SH"),
        "factor_config": config.get("factor_config", {}),
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def load_completed_experiment_results(run_dir: str | Path) -> list[dict[str, object]]:
    """Load completed experiment configs and metrics from a research run directory."""

    experiments_dir = Path(run_dir) / "experiments"
    if not experiments_dir.exists():
        return []

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

        results.append(
            {
                "name": experiment_dir.name,
                "config": read_json(config_path),
                "performance": read_json(metrics_path),
            }
        )
    return results


def select_recommendation_models(
    run_dir: str | Path,
    *,
    top_k_models: int = 5,
    min_sharpe: float = 0.1,
    min_excess_cumulative_return: float = 0.0,
    max_drawdown: float = -0.3,
    min_positive_excess_month_ratio: float = 0.5,
) -> list[dict[str, object]]:
    """Select stable completed experiments for recommendation generation."""

    completed = load_completed_experiment_results(run_dir)
    filtered: list[dict[str, object]] = []
    for item in completed:
        performance = item["performance"]
        sharpe = float(performance.get("sharpe", 0.0))
        excess = float(performance.get("excess_cumulative_return", 0.0))
        drawdown = float(performance.get("max_drawdown", 0.0))
        positive_excess_month_ratio = float(performance.get("positive_excess_month_ratio", 0.0))

        if sharpe < min_sharpe:
            continue
        if excess < min_excess_cumulative_return:
            continue
        if drawdown < max_drawdown:
            continue
        if positive_excess_month_ratio < min_positive_excess_month_ratio:
            continue
        filtered.append(item)

    filtered.sort(
        key=lambda item: (
            float(item["performance"].get("sharpe", 0.0)),
            float(item["performance"].get("excess_cumulative_return", 0.0)),
        ),
        reverse=True,
    )

    deduped: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    for item in filtered:
        signature = _config_signature(item["config"])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(item)
        if len(deduped) >= top_k_models:
            break

    return deduped


def build_recommendation_consensus(
    model_outputs: list[dict[str, object]],
    *,
    top_k_stocks: int = 20,
) -> list[dict[str, object]]:
    """Aggregate per-model latest selections into a consensus recommendation list."""

    rows: list[dict[str, object]] = []
    for model_output in model_outputs:
        for stock in model_output.get("top_stocks", []):
            if not stock.get("selected", False):
                continue
            rows.append(
                {
                    "ts_code": stock["ts_code"],
                    "model_name": model_output["name"],
                    "rank": int(stock["rank"]),
                    "score": float(stock["score"]),
                    "sharpe": float(model_output["performance"].get("sharpe", 0.0)),
                    "excess_cumulative_return": float(
                        model_output["performance"].get("excess_cumulative_return", 0.0)
                    ),
                }
            )

    if not rows:
        return []

    data = pd.DataFrame(rows)
    consensus_rows: list[dict[str, object]] = []
    for ts_code, group in data.groupby("ts_code", sort=False):
        source_models = group["model_name"].drop_duplicates().sort_values().tolist()
        consensus_rows.append(
            {
                "ts_code": ts_code,
                "consensus_count": int(group["model_name"].nunique()),
                "avg_rank": float(group["rank"].mean()),
                "best_rank": int(group["rank"].min()),
                "avg_score": float(group["score"].mean()),
                "source_models": source_models,
                "source_model_count": int(len(source_models)),
            }
        )

    consensus = pd.DataFrame(consensus_rows)
    consensus = consensus.sort_values(
        ["consensus_count", "best_rank", "avg_rank", "avg_score", "ts_code"],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)
    return consensus.head(top_k_stocks).to_dict(orient="records")


def generate_daily_recommendations_from_run(
    run_dir: str | Path,
    *,
    as_of_date: str,
    start_date: str | None = None,
    top_k_models: int = 5,
    top_k_stocks: int = 20,
    min_sharpe: float = 0.1,
    min_excess_cumulative_return: float = 0.0,
    max_drawdown: float = -0.3,
    min_positive_excess_month_ratio: float = 0.5,
) -> dict[str, object]:
    """Generate current recommendations by rerunning stable research models to a target date."""

    selected_models = select_recommendation_models(
        run_dir,
        top_k_models=top_k_models,
        min_sharpe=min_sharpe,
        min_excess_cumulative_return=min_excess_cumulative_return,
        max_drawdown=max_drawdown,
        min_positive_excess_month_ratio=min_positive_excess_month_ratio,
    )

    model_outputs: list[dict[str, object]] = []
    for item in selected_models:
        config = item["config"]
        pipeline_result = run_minimal_pipeline(
            ts_codes=config.get("ts_codes"),
            universe_name=config.get("universe_name"),
            start_date=start_date or str(config["start_date"]),
            end_date=as_of_date,
            top_n=int(config.get("top_n", 20)),
            benchmark_code=config.get("benchmark_code", "000300.SH"),
            factor_config=config.get("factor_config"),
        )

        model_outputs.append(
            {
                "name": item["name"],
                "config": config,
                "performance": item["performance"],
                "latest_selection": pipeline_result["latest_selection"],
                "top_stocks": pipeline_result["latest_selection"].get("top_stocks", []),
            }
        )

    recommendations = build_recommendation_consensus(model_outputs, top_k_stocks=top_k_stocks)
    return {
        "run_dir": str(run_dir),
        "as_of_date": as_of_date,
        "selection_rules": {
            "top_k_models": top_k_models,
            "top_k_stocks": top_k_stocks,
            "min_sharpe": min_sharpe,
            "min_excess_cumulative_return": min_excess_cumulative_return,
            "max_drawdown": max_drawdown,
            "min_positive_excess_month_ratio": min_positive_excess_month_ratio,
        },
        "selected_models": [
            {
                "name": item["name"],
                "universe_name": item["config"].get("universe_name", "custom"),
                "top_n": item["config"].get("top_n"),
                "factor_config": item["config"].get("factor_config", {}),
                "sharpe": float(item["performance"].get("sharpe", 0.0)),
                "excess_cumulative_return": float(item["performance"].get("excess_cumulative_return", 0.0)),
                "max_drawdown": float(item["performance"].get("max_drawdown", 0.0)),
                "positive_excess_month_ratio": float(item["performance"].get("positive_excess_month_ratio", 0.0)),
            }
            for item in model_outputs
        ],
        "recommendations": recommendations,
    }


def render_recommendation_text(report: dict[str, object]) -> str:
    """Render a compact daily recommendation report."""

    selection_rules = report.get("selection_rules", {})
    selected_models = report.get("selected_models", [])
    recommendations = report.get("recommendations", [])

    lines = [
        "research run:",
        str(report.get("run_dir", "N/A")),
        "",
        "as of date:",
        str(report.get("as_of_date", "N/A")),
        "",
        "selection rules:",
        f"top_k_models={selection_rules.get('top_k_models')}",
        f"top_k_stocks={selection_rules.get('top_k_stocks')}",
        f"min_sharpe={selection_rules.get('min_sharpe')}",
        f"min_excess_cumulative_return={selection_rules.get('min_excess_cumulative_return')}",
        f"max_drawdown={selection_rules.get('max_drawdown')}",
        f"min_positive_excess_month_ratio={selection_rules.get('min_positive_excess_month_ratio')}",
        "",
        "selected models:",
    ]

    if selected_models:
        for item in selected_models:
            lines.append(
                f"{item['name']} universe={item['universe_name']} top_n={item['top_n']} "
                f"sharpe={item['sharpe']:.6f} excess={item['excess_cumulative_return']:.6f} "
                f"mdd={item['max_drawdown']:.6f}"
            )
    else:
        lines.append("N/A")

    lines.extend(["", "recommendations:"])
    if recommendations:
        for index, item in enumerate(recommendations, start=1):
            lines.append(
                f"{index}. {item['ts_code']} consensus={item['consensus_count']} "
                f"best_rank={item['best_rank']} avg_rank={item['avg_rank']:.2f} "
                f"avg_score={item['avg_score']:.6f}"
            )
            lines.append(f"   source_models={', '.join(item['source_models'])}")
    else:
        lines.append("N/A")

    return "\n".join(lines)
