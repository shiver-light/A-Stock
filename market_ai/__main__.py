"""Command line entry point for daily market radar workflows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_ai.config import load_market_radar_config
from market_ai.lifecycle import load_theme_score_history
from market_ai.providers.market import LocalCsvMarketProvider, TushareDailyMarketProvider, TushareLimitMarketProvider
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
    run_parser.add_argument(
        "--market-provider",
        default="local_csv",
        choices=["local_csv", "tushare_daily", "tushare_limit"],
    )
    run_parser.add_argument("--market-data-dir", default=None, help="Directory containing local market CSV files.")
    run_parser.add_argument("--universe-name", default=None, help="Optional universe filter for tushare_daily.")
    run_parser.add_argument("--refresh", action="store_true", help="Refresh provider cache when supported.")
    run_parser.add_argument("--news-csv", default=None, help="Optional local news CSV path.")
    run_parser.add_argument("--stock-theme-labels", default=None, help="Optional stock theme label CSV path.")
    run_parser.add_argument("--min-stock-theme-confidence", type=float, default=0.0)
    run_parser.add_argument("--theme-score-history", default=None, help="Optional historical theme score CSV path.")
    run_parser.add_argument("--output-dir", default=None, help="Output directory for .json and .md reports.")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _run(args: argparse.Namespace) -> int:
    config = load_market_radar_config(args.config)
    taxonomy = load_theme_taxonomy(args.taxonomy)
    if args.market_provider == "local_csv":
        if not args.market_data_dir:
            raise ValueError("--market-data-dir is required when --market-provider local_csv.")
        market_provider = LocalCsvMarketProvider(args.market_data_dir)
    elif args.market_provider == "tushare_daily":
        market_provider = TushareDailyMarketProvider(universe_name=args.universe_name, refresh=args.refresh)
    else:
        market_provider = TushareLimitMarketProvider(universe_name=args.universe_name, refresh=args.refresh)
    news_provider = LocalCsvNewsProvider(args.news_csv) if args.news_csv else None
    output_dir = Path(args.output_dir or config.output.report_dir)

    news_end = _trade_date_close_time(args.trade_date)
    news_start = news_end - timedelta(hours=config.news.lookback_hours)
    history_path = Path(args.theme_score_history) if args.theme_score_history else output_dir / "theme_scores.csv"
    theme_score_history = load_theme_score_history(history_path)
    report = build_daily_radar_report(
        trade_date=args.trade_date,
        market_provider=market_provider,
        taxonomy=taxonomy,
        config=config,
        news_provider=news_provider,
        news_start_time=news_start,
        news_end_time=news_end,
        stock_theme_labels_path=args.stock_theme_labels,
        min_stock_theme_confidence=args.min_stock_theme_confidence,
        theme_score_history=theme_score_history,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.trade_date}.json"
    md_path = output_dir / f"{args.trade_date}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(render_daily_radar_report_markdown(report), encoding="utf-8")
    _append_theme_score_history(history_path, report.core_themes)

    print(f"wrote market radar report: {md_path}")
    print(f"wrote structured report: {json_path}")
    return 0


def _trade_date_close_time(trade_date: str) -> datetime:
    return datetime.strptime(str(trade_date), "%Y%m%d").replace(hour=16, minute=30, tzinfo=CN_TZ)


def _append_theme_score_history(path: Path, themes: list) -> None:
    if not themes:
        return
    rows = [
        {
            "trade_date": theme.trade_date,
            "theme": theme.theme,
            "score": theme.score,
            "rank": theme.rank,
            "lifecycle_stage": theme.lifecycle_stage,
        }
        for theme in themes
    ]
    current = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, current], ignore_index=True)
    else:
        combined = current
    combined["trade_date"] = combined["trade_date"].astype(str).str.replace(r"\\.0$", "", regex=True)
    combined = (
        combined.sort_values(["trade_date", "theme"])
        .drop_duplicates(subset=["trade_date", "theme"], keep="last")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
