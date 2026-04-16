"""Research config loading."""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


def load_research_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Research config file not found: {config_path}")

    if yaml is None:
        raise RuntimeError("PyYAML is required to load research experiment configs.")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if "experiments" not in payload or not isinstance(payload["experiments"], list):
        raise ValueError("Research config must contain an 'experiments' list.")
    return payload
