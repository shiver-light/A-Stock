"""Build time-split validation research configs for selected theme models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.theme_research import (
    DEFAULT_THEME_VALIDATION_MODELS,
    DEFAULT_THEME_VALIDATION_WINDOWS,
    build_and_write_theme_validation_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build time-split validation config for theme models.")
    parser.add_argument(
        "--base-config",
        default="research/experiments_hs300_zz500_taxonomy_theme.yaml",
        help="Base theme research YAML path.",
    )
    parser.add_argument(
        "--output-config",
        default="research/experiments_hs300_zz500_taxonomy_theme_validation.yaml",
        help="Output validation research YAML path.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        help="Selected model name. Repeatable. Defaults to the recommended shortlist.",
    )
    parser.add_argument(
        "--window",
        action="append",
        dest="windows",
        help="Validation window in NAME:START_DATE:END_DATE format. Repeatable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    windows = _parse_windows(args.windows) if args.windows else DEFAULT_THEME_VALIDATION_WINDOWS
    result = build_and_write_theme_validation_config(
        base_config_path=args.base_config,
        output_config_path=args.output_config,
        selected_model_names=tuple(args.model_names or DEFAULT_THEME_VALIDATION_MODELS),
        windows=windows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_windows(items: list[str]) -> tuple[dict[str, str], ...]:
    windows: list[dict[str, str]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"--window must use NAME:START_DATE:END_DATE format: {item}")
        name, start_date, end_date = [part.strip() for part in parts]
        windows.append({"name": name, "start_date": start_date, "end_date": end_date})
    return tuple(windows)


if __name__ == "__main__":
    raise SystemExit(main())
