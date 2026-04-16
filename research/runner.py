"""Experiment runner for batch research."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pipeline import run_minimal_pipeline
from research.resume import normalize_stale_statuses
from research.state import load_status, mark_completed, mark_failed, mark_running
from research.storage import (
    ensure_run_dir,
    get_experiment_dir,
    read_json,
    utc_now_iso,
    write_json,
    write_summary,
    write_text,
    write_yaml,
)
from research.summary import rebuild_summary_from_disk


def run_experiments(
    config: dict[str, object],
    *,
    output_dir: str | Path = "research_runs",
    run_name: str | None = None,
    resume: bool = True,
) -> list[dict[str, object]]:
    global_config = deepcopy(config.get("global", {}))
    experiments = config.get("experiments", [])
    run_name = run_name or f"run_{utc_now_iso().replace(':', '').replace('-', '')}"
    run_dir = ensure_run_dir(output_dir, run_name)
    write_yaml(run_dir / "run_config.yaml", config)
    write_json(
        run_dir / "run_status.json",
        {
            "run_name": run_name,
            "status": "running",
            "updated_at": utc_now_iso(),
            "experiment_count": len(experiments),
        },
    )

    if resume:
        normalize_stale_statuses(run_dir)

    results: list[dict[str, object]] = []

    for experiment in experiments:
        experiment_config = deepcopy(global_config)
        experiment_config.update(experiment)

        name = experiment_config.get("name")
        if not name:
            raise ValueError("Each experiment must have a name.")

        experiment_dir = get_experiment_dir(run_dir, name)
        config_path = experiment_dir / "config.json"
        status_path = experiment_dir / "status.json"
        metrics_path = experiment_dir / "metrics.json"
        latest_selection_path = experiment_dir / "latest_selection.json"
        report_path = experiment_dir / "report.json"
        report_text_path = experiment_dir / "report.txt"
        error_path = experiment_dir / "error.txt"

        write_json(config_path, experiment_config)
        status = load_status(status_path, name)
        if status.get("status") == "completed" and metrics_path.exists() and report_path.exists():
            results.append(
                {
                    "name": name,
                    "config": read_json(config_path),
                    "performance": read_json(metrics_path),
                    "latest_selection": read_json(latest_selection_path) if latest_selection_path.exists() else {},
                    "report": read_json(report_path),
                    "report_text": report_text_path.read_text(encoding="utf-8") if report_text_path.exists() else "",
                }
            )
            continue

        status = mark_running(status_path, status)
        try:
            pipeline_result = run_minimal_pipeline(
                ts_codes=experiment_config.get("ts_codes"),
                universe_name=experiment_config.get("universe_name"),
                start_date=experiment_config["start_date"],
                end_date=experiment_config["end_date"],
                top_n=int(experiment_config.get("top_n", 20)),
                benchmark_code=experiment_config.get("benchmark_code", "000300.SH"),
                factor_config=experiment_config.get("factor_config"),
            )

            write_json(metrics_path, pipeline_result["performance"])
            write_json(latest_selection_path, pipeline_result["latest_selection"])
            write_json(report_path, pipeline_result["report"])
            write_text(report_text_path, pipeline_result["report_text"])
            if error_path.exists():
                error_path.unlink()
            mark_completed(status_path, status)

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
        except Exception as exc:
            write_text(error_path, str(exc))
            mark_failed(status_path, status, str(exc))
        finally:
            partial_summary = rebuild_summary_from_disk(run_dir)
            write_summary(partial_summary, run_dir)

    partial_summary = rebuild_summary_from_disk(run_dir)
    write_summary(partial_summary, run_dir)
    write_json(
        run_dir / "run_status.json",
        {
            "run_name": run_name,
            "status": "completed",
            "updated_at": utc_now_iso(),
            "experiment_count": len(experiments),
            "completed_count": int(len(partial_summary)),
        },
    )
    return results
