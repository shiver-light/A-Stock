"""Experiment runner for batch research."""

from __future__ import annotations

from copy import deepcopy

from pipeline import run_minimal_pipeline


def run_experiments(config: dict[str, object]) -> list[dict[str, object]]:
    global_config = deepcopy(config.get("global", {}))
    experiments = config.get("experiments", [])
    results: list[dict[str, object]] = []

    for experiment in experiments:
        experiment_config = deepcopy(global_config)
        experiment_config.update(experiment)

        name = experiment_config.get("name")
        if not name:
            raise ValueError("Each experiment must have a name.")

        pipeline_result = run_minimal_pipeline(
            ts_codes=experiment_config.get("ts_codes"),
            universe_name=experiment_config.get("universe_name"),
            start_date=experiment_config["start_date"],
            end_date=experiment_config["end_date"],
            top_n=int(experiment_config.get("top_n", 20)),
            benchmark_code=experiment_config.get("benchmark_code", "000300.SH"),
            factor_config=experiment_config.get("factor_config"),
        )

        results.append(
            {
                "name": name,
                "config": experiment_config,
                "performance": pipeline_result["performance"],
                "latest_selection": pipeline_result["latest_selection"],
                "report": pipeline_result["report"],
                "report_text": pipeline_result["report_text"],
            }
        )

    return results
