"""Filesystem persistence helpers for research runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_run_dir(base_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    return run_dir


def get_experiment_dir(run_dir: str | Path, experiment_name: str) -> Path:
    experiment_dir = Path(run_dir) / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(content)


def write_yaml_copy(src_path: str | Path, dst_path: str | Path) -> None:
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def write_summary(summary: pd.DataFrame, run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.parquet"
    summary_csv_path = run_path / "summary.csv"
    if summary.empty:
        summary = pd.DataFrame()
    summary.to_parquet(summary_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
