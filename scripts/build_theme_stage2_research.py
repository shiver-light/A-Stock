"""Build stage2 theme research configs from selected validated models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.theme_research import (
    DEFAULT_THEME_STAGE2_BASE_MODELS,
    build_and_write_theme_stage2_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage2 factor-ablation config for theme models.")
    parser.add_argument(
        "--base-config",
        default="research/experiments_hs300_zz500_taxonomy_theme.yaml",
        help="Base theme research YAML path.",
    )
    parser.add_argument(
        "--output-config",
        default="research/experiments_hs300_zz500_taxonomy_theme_stage2.yaml",
        help="Output stage2 research YAML path.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        help="Selected base model name. Repeatable. Defaults to the recommended shortlist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_and_write_theme_stage2_config(
        base_config_path=args.base_config,
        output_config_path=args.output_config,
        selected_model_names=tuple(args.model_names or DEFAULT_THEME_STAGE2_BASE_MODELS),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
