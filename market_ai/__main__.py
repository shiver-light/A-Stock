"""Command line entry point for daily market radar workflows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_ai.config import load_market_radar_config
from market_ai.providers.market import LocalCsvMarketProvider
from market_ai.providers.news import LocalCsvNewsProvider
from market_ai.reports import render_daily_radar_report_markdown
from market_ai.themes import load_theme_taxonomy
from market_ai.workflow import build_daily_radar_report


CN_TZ = timezone(timedelta(hours=8))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m market_ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate one daily market radar report.")
    run_parser.add_argument("--trade-date", required=True, help="Trade date, for example 20260912.")
    run_parser.add_argument("--config", default=None, help="Radar YAML config path.")
    run_parser.add_argument("--taxonomy", default=None, help="Theme taxonomy YAML path.")
    run_parser.add_argument("--market-provider", default="local_csv", choices=["local_csv"])
    run_parser.add_argument("--market-data-dir", required=True, help="Directory containing local market CSV files.")
    run_parser.add_argument("--news-csv", default=None, help="Optional local news CSV path.")
    run_parser.add_argument("--output-dir", default=None, help="Output directory for .json and .md reports.")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _run(args: argparse.Namespace) -> int:
    config = load_market_radar_config(args.config)
    taxonomy = load_theme_taxonomy(args.taxonomy)
    market_provider = LocalCsvMarketProvider(args.market_data_dir)
    news_provider = LocalCsvNewsProvider(args.news_csv) if args.news_csv else None
    output_dir = Path(args.output_dir or config.output.report_dir)

    news_end = _trade_date_close_time(args.trade_date)
    news_start = news_end - timedelta(hours=config.news.lookback_hours)
    report = build_daily_radar_report(
        trade_date=args.trade_date,
        market_provider=market_provider,
        taxonomy=taxonomy,
        config=config,
        news_provider=news_provider,
        news_start_time=news_start,
        news_end_time=news_end,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.trade_date}.json"
    md_path = output_dir / f"{args.trade_date}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(render_daily_radar_report_markdown(report), encoding="utf-8")

    print(f"wrote market radar report: {md_path}")
    print(f"wrote structured report: {json_path}")
    return 0


def _trade_date_close_time(trade_date: str) -> datetime:
    return datetime.strptime(str(trade_date), "%Y%m%d").replace(hour=16, minute=30, tzinfo=CN_TZ)


if __name__ == "__main__":
    raise SystemExit(main())
