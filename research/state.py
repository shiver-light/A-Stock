"""Experiment state transitions."""

from __future__ import annotations

from pathlib import Path

from research.storage import read_json, utc_now_iso, write_json


def default_status_payload(name: str) -> dict[str, object]:
    now = utc_now_iso()
    return {
        "name": name,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "updated_at": now,
        "attempt": 0,
        "error_message": None,
    }


def load_status(status_path: str | Path, experiment_name: str) -> dict[str, object]:
    path = Path(status_path)
    if not path.exists():
        payload = default_status_payload(experiment_name)
        write_json(path, payload)
        return payload
    return read_json(path)


def mark_running(status_path: str | Path, payload: dict[str, object]) -> dict[str, object]:
    now = utc_now_iso()
    payload = dict(payload)
    payload["status"] = "running"
    payload["started_at"] = payload.get("started_at") or now
    payload["updated_at"] = now
    payload["attempt"] = int(payload.get("attempt", 0)) + 1
    payload["error_message"] = None
    write_json(status_path, payload)
    return payload


def mark_completed(status_path: str | Path, payload: dict[str, object]) -> dict[str, object]:
    now = utc_now_iso()
    payload = dict(payload)
    payload["status"] = "completed"
    payload["finished_at"] = now
    payload["updated_at"] = now
    payload["error_message"] = None
    write_json(status_path, payload)
    return payload


def mark_failed(status_path: str | Path, payload: dict[str, object], error_message: str) -> dict[str, object]:
    now = utc_now_iso()
    payload = dict(payload)
    payload["status"] = "failed"
    payload["finished_at"] = now
    payload["updated_at"] = now
    payload["error_message"] = error_message
    write_json(status_path, payload)
    return payload


def mark_stale(status_path: str | Path, payload: dict[str, object], error_message: str) -> dict[str, object]:
    now = utc_now_iso()
    payload = dict(payload)
    payload["status"] = "stale"
    payload["updated_at"] = now
    payload["error_message"] = error_message
    write_json(status_path, payload)
    return payload
