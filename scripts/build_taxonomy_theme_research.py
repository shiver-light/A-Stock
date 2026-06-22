"""Build research configs from a theme taxonomy tag table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.theme_research import build_and_write_taxonomy_theme_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build taxonomy-driven theme tags and research YAML.")
    parser.add_argument("--taxonomy", default="research/theme_taxonomy.yaml", help="Theme taxonomy YAML path.")
    parser.add_argument("--start-date", default="20250101", help="Research start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="Research end date in YYYYMMDD format.")
    parser.add_argument(
        "--tag-as-of-date",
        default=None,
        help="Theme tag snapshot date. Defaults to --start-date to avoid label look-ahead.",
    )
    parser.add_argument(
        "--pool-as-of-date",
        default=None,
        help="Index membership snapshot date. Defaults to --start-date to avoid end-date membership look-ahead.",
    )
    parser.add_argument(
        "--universe-name",
        action="append",
        choices=["hs300", "zz500"],
        dest="universe_names",
        help="Universe to include. Can be repeated. Defaults to hs300 and zz500.",
    )
    parser.add_argument("--theme", action="append", dest="themes", help="Theme to include. Repeatable.")
    parser.add_argument("--sub-theme", action="append", dest="sub_themes", help="Sub-theme to include. Repeatable.")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum theme tag confidence.")
    parser.add_argument(
        "--output-config",
        default="research/experiments_hs300_zz500_taxonomy_theme.yaml",
        help="Output research YAML path.",
    )
    parser.add_argument(
        "--output-pool",
        default="research/theme_pool_hs300_zz500_taxonomy.csv",
        help="Output theme stock-pool CSV path.",
    )
    parser.add_argument(
        "--output-tags",
        default="research/theme_tags_hs300_zz500_taxonomy.csv",
        help="Output theme tag CSV path.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh Tushare stock_basic/index_weight caches.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_and_write_taxonomy_theme_research(
        taxonomy_path=args.taxonomy,
        start_date=args.start_date,
        end_date=args.end_date,
        output_config_path=args.output_config,
        output_pool_path=args.output_pool,
        output_tags_path=args.output_tags,
        universe_names=tuple(args.universe_names or ["hs300", "zz500"]),
        themes=tuple(args.themes) if args.themes else None,
        sub_themes=tuple(args.sub_themes) if args.sub_themes else None,
        min_confidence=args.min_confidence,
        tag_as_of_date=args.tag_as_of_date,
        pool_as_of_date=args.pool_as_of_date,
        refresh=args.refresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
