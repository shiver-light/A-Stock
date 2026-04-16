"""Resume helpers for research runs."""

from __future__ import annotations

from pathlib import Path

from research.state import load_status, mark_stale


def normalize_stale_statuses(run_dir: str | Path) -> None:
    experiments_dir = Path(run_dir) / "experiments"
    if not experiments_dir.exists():
        return

    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        status_path = experiment_dir / "status.json"
        status = load_status(status_path, experiment_dir.name)
        if status.get("status") == "running":
            mark_stale(status_path, status, "Interrupted in previous run.")
