"""Command line entry point for daily market radar workflows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_ai.config import load_market_radar_config
from data.tushare_client import TushareClient
from market_ai.lifecycle import load_theme_score_history
from market_ai.providers.market import LocalCsvMarketProvider, TushareDailyMarketProvider, TushareLimitMarketProvider
from market_ai.providers.news import LocalCsvNewsProvider, TushareMajorNewsProvider
from market_ai.providers.news.importer import import_news_csvs
from market_ai.reports import render_daily_radar_report_markdown
from market_ai.themes.enrichment import enrich_stock_theme_labels
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

    range_parser = subparsers.add_parser("run-range", help="Generate daily radar reports for an A-share date range.")
    range_parser.add_argument("--start-date", required=True, help="Start trade date, for example 20260901.")
    range_parser.add_argument("--end-date", required=True, help="End trade date, for example 20260911.")
    range_parser.add_argument("--config", default=None, help="Radar YAML config path.")
    range_parser.add_argument("--taxonomy", default=None, help="Theme taxonomy YAML path.")
    range_parser.add_argument(
        "--market-provider",
        default="tushare_limit",
        choices=["tushare_daily", "tushare_limit"],
    )
    range_parser.add_argument("--universe-name", default=None, help="Optional universe filter.")
    range_parser.add_argument("--refresh", action="store_true", help="Refresh provider cache when supported.")
    range_parser.add_argument("--news-csv", default=None, help="Optional local news CSV path.")
    range_parser.add_argument("--stock-theme-labels", default=None, help="Optional stock theme label CSV path.")
    range_parser.add_argument("--min-stock-theme-confidence", type=float, default=0.0)
    range_parser.add_argument("--theme-score-history", default=None, help="Optional historical theme score CSV path.")
    range_parser.add_argument("--output-dir", default=None, help="Output directory for .json and .md reports.")

    enrich_parser = subparsers.add_parser("enrich-themes", help="Build reusable stock theme label CSVs.")
    enrich_parser.add_argument("--start-date", required=True, help="Start trade date, for example 20260907.")
    enrich_parser.add_argument("--end-date", required=True, help="End trade date, for example 20260911.")
    enrich_parser.add_argument("--taxonomy", default=None, help="Theme taxonomy YAML path.")
    enrich_parser.add_argument(
        "--base-labels",
        action="append",
        default=[],
        help="Existing stock theme label CSV. Can be supplied multiple times.",
    )
    enrich_parser.add_argument(
        "--reason-csv",
        action="append",
        default=[],
        help="CSV containing limit/news/announcement reasons to classify. Can be supplied multiple times.",
    )
    enrich_parser.add_argument("--input-dir", default=None, help="Reserved for compatibility; existing reports are not mutated.")
    enrich_parser.add_argument("--output-file", required=True, help="Output stock theme label CSV path.")

    import_news_parser = subparsers.add_parser("import-news", help="Normalize external news CSVs for radar reports.")
    import_news_parser.add_argument("--start-date", required=True, help="Start trade date, for example 20260907.")
    import_news_parser.add_argument("--end-date", required=True, help="End trade date, for example 20260911.")
    import_news_parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="External news or announcement CSV. Can be supplied multiple times.",
    )
    import_news_parser.add_argument("--output-file", required=True, help="Output normalized news CSV path.")
    import_news_parser.add_argument("--config", default=None, help="Radar YAML config path.")
    import_news_parser.add_argument("--default-source", default="manual", help="Source used when input source is empty.")
    import_news_parser.add_argument("--rerun-radar", action="store_true", help="Rerun radar reports after importing news.")
    import_news_parser.add_argument("--taxonomy", default=None, help="Theme taxonomy YAML path used by rerun-radar.")
    import_news_parser.add_argument(
        "--market-provider",
        default="tushare_limit",
        choices=["tushare_daily", "tushare_limit"],
        help="Market provider used by rerun-radar.",
    )
    import_news_parser.add_argument("--universe-name", default=None, help="Optional universe filter used by rerun-radar.")
    import_news_parser.add_argument("--refresh", action="store_true", help="Refresh provider cache when rerun-radar is used.")
    import_news_parser.add_argument("--stock-theme-labels", default=None, help="Stock theme label CSV used by rerun-radar.")
    import_news_parser.add_argument("--min-stock-theme-confidence", type=float, default=0.0)
    import_news_parser.add_argument("--theme-score-history", default=None, help="Theme score history used by rerun-radar.")
    import_news_parser.add_argument("--output-dir", default=None, help="Output directory used by rerun-radar.")

    collect_news_parser = subparsers.add_parser("collect-news", help="Collect news into a radar-compatible CSV.")
    collect_news_parser.add_argument("--start-date", required=True, help="Start trade date, for example 20260907.")
    collect_news_parser.add_argument("--end-date", required=True, help="End trade date, for example 20260911.")
    collect_news_parser.add_argument("--provider", default="tushare_major_news", choices=["tushare_major_news"])
    collect_news_parser.add_argument("--output-file", required=True, help="Output normalized news CSV path.")
    collect_news_parser.add_argument("--config", default=None, help="Radar YAML config path.")
    collect_news_parser.add_argument("--refresh", action="store_true", help="Refresh provider cache.")
    collect_news_parser.add_argument("--cache-dir", default=None, help="Optional provider cache directory.")
    collect_news_parser.add_argument("--slice-hours", type=int, default=6, help="Tushare request window size in hours.")
    collect_news_parser.add_argument("--rerun-radar", action="store_true", help="Rerun radar reports after collecting news.")
    collect_news_parser.add_argument("--taxonomy", default=None, help="Theme taxonomy YAML path used by rerun-radar.")
    collect_news_parser.add_argument(
        "--market-provider",
        default="tushare_limit",
        choices=["tushare_daily", "tushare_limit"],
        help="Market provider used by rerun-radar.",
    )
    collect_news_parser.add_argument("--universe-name", default=None, help="Optional universe filter used by rerun-radar.")
    collect_news_parser.add_argument("--stock-theme-labels", default=None, help="Stock theme label CSV used by rerun-radar.")
    collect_news_parser.add_argument("--min-stock-theme-confidence", type=float, default=0.0)
    collect_news_parser.add_argument("--theme-score-history", default=None, help="Theme score history used by rerun-radar.")
    collect_news_parser.add_argument("--output-dir", default=None, help="Output directory used by rerun-radar.")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "run-range":
        return _run_range(args)
    if args.command == "enrich-themes":
        return _enrich_themes(args)
    if args.command == "import-news":
        return _import_news(args)
    if args.command == "collect-news":
        return _collect_news(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _run(args: argparse.Namespace) -> int:
    _run_one_report(args=args, trade_date=args.trade_date)
    return 0


def _run_range(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    trade_dates = _get_a_share_trade_dates(args.start_date, args.end_date)
    if not trade_dates:
        print("no open A-share trade dates found in range")
        return 0
    for trade_date in trade_dates:
        _run_one_report(args=args, trade_date=trade_date)
    print(f"completed market radar range: {trade_dates[0]} to {trade_dates[-1]}, days={len(trade_dates)}")
    return 0


def _enrich_themes(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    taxonomy = load_theme_taxonomy(args.taxonomy)
    base_label_paths = list(args.base_labels)
    if not base_label_paths and not args.reason_csv and args.input_dir:
        base_label_paths = _discover_stock_theme_label_files(Path(args.input_dir))
    result = enrich_stock_theme_labels(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output_file,
        taxonomy=taxonomy,
        base_label_paths=base_label_paths,
        reason_csv_paths=args.reason_csv,
    )
    print(f"wrote stock theme labels: {args.output_file}")
    print(f"label_rows={len(result)}")
    if not result.empty:
        print(f"trade_dates={result['trade_date'].nunique()} unique_stocks={result['stock_code'].nunique()}")
    return 0


def _import_news(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    config = load_market_radar_config(args.config)
    result = import_news_csvs(
        args.input_csv,
        output_file=args.output_file,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_hours=config.news.lookback_hours,
        default_source=args.default_source,
    )
    print(f"wrote normalized news: {args.output_file}")
    print(f"news_rows={len(result)}")
    if not result.empty:
        print(f"sources={result['source'].nunique()} first={result['published_at'].min()} last={result['published_at'].max()}")
    if args.rerun_radar:
        rerun_args = argparse.Namespace(**vars(args))
        rerun_args.news_csv = args.output_file
        return _run_range(rerun_args)
    return 0


def _collect_news(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    config = load_market_radar_config(args.config)
    start_time = _trade_date_close_time(args.start_date) - timedelta(hours=config.news.lookback_hours)
    end_time = _trade_date_close_time(args.end_date)
    if args.provider != "tushare_major_news":
        raise ValueError(f"Unsupported news provider: {args.provider}")
    cache_dir = args.cache_dir or config.output.cache_dir + "/news/tushare_major_news"
    provider = TushareMajorNewsProvider(cache_dir=cache_dir, refresh=args.refresh, slice_hours=args.slice_hours)
    data = provider.fetch_news_frame(start_time=start_time, end_time=end_time)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"wrote collected news: {output_path}")
    print(f"news_rows={len(data)}")
    if not data.empty:
        print(f"sources={data['source'].nunique()} first={data['published_at'].min()} last={data['published_at'].max()}")
    if args.rerun_radar:
        rerun_args = argparse.Namespace(**vars(args))
        rerun_args.news_csv = str(output_path)
        return _run_range(rerun_args)
    return 0


def _discover_stock_theme_label_files(input_dir: Path) -> list[Path]:
    """Find existing generated stock theme label CSVs under an output directory."""
    if not input_dir.exists():
        return []
    return sorted(input_dir.glob("theme_labels*/limit_up_stock_theme_labels*.csv"))


def _run_one_report(*, args: argparse.Namespace, trade_date: str) -> None:
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

    news_end = _trade_date_close_time(trade_date)
    news_start = news_end - timedelta(hours=config.news.lookback_hours)
    history_path = Path(args.theme_score_history) if args.theme_score_history else output_dir / "theme_scores.csv"
    theme_score_history = load_theme_score_history(history_path)
    report = build_daily_radar_report(
        trade_date=trade_date,
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
    json_path = output_dir / f"{trade_date}.json"
    md_path = output_dir / f"{trade_date}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(render_daily_radar_report_markdown(report), encoding="utf-8")
    _append_theme_score_history(history_path, report.core_themes)
    _append_theme_daily_snapshot(output_dir / "theme_daily_snapshot.csv", report)

    print(f"wrote market radar report: {md_path}")
    print(f"wrote structured report: {json_path}")


def _trade_date_close_time(trade_date: str) -> datetime:
    return datetime.strptime(str(trade_date), "%Y%m%d").replace(hour=16, minute=30, tzinfo=CN_TZ)


def _get_a_share_trade_dates(start_date: str, end_date: str) -> list[str]:
    client = TushareClient()
    calendar = client.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date,is_open",
    )
    if calendar is None or calendar.empty:
        return []
    return sorted(calendar["cal_date"].astype(str).tolist())


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


def _append_theme_daily_snapshot(path: Path, report) -> None:
    """Append one report's theme snapshot with idempotent trade_date/theme upsert."""
    if not report.core_themes:
        return
    catalysts = {item.theme: item for item in report.theme_catalysts}
    rows = []
    for theme in report.core_themes:
        catalyst = catalysts.get(theme.theme)
        components = dict(theme.components or {})
        rows.append(
            {
                "trade_date": report.trade_date,
                "theme": theme.theme,
                "rank": theme.rank,
                "score": theme.score,
                "lifecycle_stage": theme.lifecycle_stage,
                "validation_state": theme.validation_state or "",
                "limit_up_strength": components.get("LimitUpStrength", 0.0),
                "breadth": components.get("Breadth", 0.0),
                "leader_strength": components.get("LeaderStrength", 0.0),
                "news_catalyst": components.get("NewsCatalyst", 0.0),
                "volume_expansion": components.get("VolumeExpansion", 0.0),
                "persistence": components.get("Persistence", 0.0),
                "novelty": components.get("Novelty", 0.0),
                "confirmed_event_count": catalyst.confirmed_event_count if catalyst else 0,
                "unconfirmed_event_count": catalyst.unconfirmed_event_count if catalyst else 0,
                "related_event_count": catalyst.related_event_count if catalyst else 0,
                "primary_event": catalyst.primary_event if catalyst else "",
                "catalyst_conclusion": catalyst.conclusion if catalyst else "",
                "reason": " | ".join(theme.reasons or []),
            }
        )
    current = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, current], ignore_index=True)
    else:
        combined = current
    combined["trade_date"] = combined["trade_date"].astype(str).str.replace(r"\\.0$", "", regex=True)
    combined = (
        combined.sort_values(["trade_date", "rank", "theme"])
        .drop_duplicates(subset=["trade_date", "theme"], keep="last")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
