"""Build technology-theme HS300/ZZ500 research configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.theme_research import (
    DEFAULT_TECH_INDUSTRY_KEYWORDS,
    DEFAULT_TECH_NAME_KEYWORDS,
    build_and_write_technology_theme_research,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build technology-theme research YAML and stock-pool CSV.")
    parser.add_argument("--start-date", default="20250101", help="Research start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="Research end date in YYYYMMDD format.")
    parser.add_argument(
        "--pool-as-of-date",
        default=None,
        help="Date used to snapshot index membership. Defaults to --start-date to avoid end-date membership look-ahead.",
    )
    parser.add_argument(
        "--universe-name",
        action="append",
        choices=["hs300", "zz500"],
        dest="universe_names",
        help="Universe to include. Can be repeated. Defaults to hs300 and zz500.",
    )
    parser.add_argument(
        "--industry-keyword",
        action="append",
        dest="industry_keywords",
        help="Extra or replacement stock_basic.industry keyword. Repeatable.",
    )
    parser.add_argument(
        "--name-keyword",
        action="append",
        dest="name_keywords",
        help="Extra or replacement stock name keyword. Repeatable.",
    )
    parser.add_argument(
        "--output-config",
        default="research/experiments_hs300_zz500_technology_theme.yaml",
        help="Output research YAML path.",
    )
    parser.add_argument(
        "--output-pool",
        default="research/theme_pool_hs300_zz500_technology.csv",
        help="Output theme stock-pool CSV path.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh Tushare stock_basic/index_weight caches.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    industry_keywords = tuple(args.industry_keywords) if args.industry_keywords else DEFAULT_TECH_INDUSTRY_KEYWORDS
    name_keywords = tuple(args.name_keywords) if args.name_keywords else DEFAULT_TECH_NAME_KEYWORDS
    result = build_and_write_technology_theme_research(
        start_date=args.start_date,
        end_date=args.end_date,
        pool_as_of_date=args.pool_as_of_date,
        output_config_path=args.output_config,
        output_pool_path=args.output_pool,
        universe_names=tuple(args.universe_names or ["hs300", "zz500"]),
        industry_keywords=industry_keywords,
        name_keywords=name_keywords,
        refresh=args.refresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
