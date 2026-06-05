"""CLI entrypoint for batch research."""

from __future__ import annotations

import argparse
import json

from research import (
    archive_consensus_recommendation_csv,
    archive_daily_consensus_recommendations,
    generate_active_pullback_recommendations,
    generate_daily_consensus_recommendations,
    generate_daily_recommendations_from_run,
    load_research_config,
    rebuild_summary_from_disk,
    render_active_pullback_text,
    render_consensus_recommendation_text,
    render_recommendation_text,
    run_experiments,
    sort_research_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch quant research experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a batch of experiments.")
    run_parser.add_argument("--config", default="research/experiments.yaml", help="Path to research config YAML.")
    run_parser.add_argument("--output-dir", default="research_runs", help="Directory to store experiment outputs.")
    run_parser.add_argument("--run-name", default=None, help="Optional run name. Auto-generated if omitted.")

    resume_parser = subparsers.add_parser("resume", help="Resume a previous research run.")
    resume_parser.add_argument("--config", default="research/experiments.yaml", help="Path to research config YAML.")
    resume_parser.add_argument("--output-dir", default="research_runs", help="Directory to store experiment outputs.")
    resume_parser.add_argument("--run-name", required=True, help="Existing run name to resume.")

    summary_parser = subparsers.add_parser("summary", help="Build partial summary from completed experiment outputs.")
    summary_parser.add_argument("--run-dir", required=True, help="Path to an existing research run directory.")
    summary_parser.add_argument(
        "--sort-by",
        default="sharpe",
        choices=["sharpe", "excess_cumulative_return", "max_drawdown"],
        help="Column to sort the summary by.",
    )
    summary_parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output a plain table or JSON payload.",
    )

    recommend_parser = subparsers.add_parser("recommend", help="Generate daily recommendations from a completed research run.")
    recommend_parser.add_argument("--run-dir", required=True, help="Path to an existing research run directory.")
    recommend_parser.add_argument("--as-of-date", required=True, help="Recommendation end date in YYYYMMDD format.")
    recommend_parser.add_argument("--start-date", default=None, help="Optional override for pipeline start date.")
    recommend_parser.add_argument("--top-k-models", type=int, default=5, help="Maximum number of stable models to reuse.")
    recommend_parser.add_argument("--top-k-stocks", type=int, default=20, help="Maximum number of consensus recommendations.")
    recommend_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output text report or JSON payload.",
    )

    active_pullback_parser = subparsers.add_parser(
        "active-pullback",
        help="Find active recent winners that are in short-term pullback.",
    )
    active_pullback_parser.add_argument("--as-of-date", required=True, help="Signal end date in YYYYMMDD format.")
    active_pullback_parser.add_argument("--start-date", default=None, help="Optional data start date in YYYYMMDD format.")
    active_pullback_parser.add_argument("--universe-name", default="zz1000", help="Universe name, e.g. zz1000.")
    active_pullback_parser.add_argument("--benchmark-code", default="000852.SH", help="Benchmark index code.")
    active_pullback_parser.add_argument("--top-n", type=int, default=30, help="Number of selected candidates.")
    active_pullback_parser.add_argument(
        "--return-quantile",
        type=float,
        default=0.70,
        help="Minimum same-date quantile for return_20d active-pool filter.",
    )
    active_pullback_parser.add_argument(
        "--turnover-quantile",
        type=float,
        default=0.70,
        help="Minimum same-date quantile for turnover_mean_20d active-pool filter.",
    )
    active_pullback_parser.add_argument(
        "--amount-quantile",
        type=float,
        default=None,
        help="Optional minimum same-date quantile for amount_mean_20d.",
    )
    active_pullback_parser.add_argument(
        "--money-flow-quantile",
        type=float,
        default=None,
        help="Optional minimum same-date quantile for money_flow_strength_20d.",
    )
    active_pullback_parser.add_argument(
        "--exclude-chinext",
        action="store_true",
        help="Exclude ChiNext stocks, identified by 300/301 ts_code prefixes, before factor calculation.",
    )
    active_pullback_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output text report or JSON payload.",
    )

    consensus_parser = subparsers.add_parser(
        "recommend-consensus",
        help="Generate A/B/C daily recommendation buckets from explicit model roles.",
    )
    consensus_parser.add_argument("--run-dir", required=True, help="Path to an existing research run directory.")
    consensus_parser.add_argument("--as-of-date", required=True, help="Recommendation end date in YYYYMMDD format.")
    consensus_parser.add_argument("--start-date", default=None, help="Optional override for pipeline start date.")
    consensus_parser.add_argument(
        "--core-models",
        nargs="*",
        default=["hstm2_04_hs300_turnover50_ret60_40_rev5_10_top10"],
        help="Model names used as the main trade core.",
    )
    consensus_parser.add_argument(
        "--confirm-models",
        nargs="*",
        default=["hstm2_02_hs300_turnover60_ret60_30_rev5_10_top15"],
        help="Model names used for confirmation.",
    )
    consensus_parser.add_argument(
        "--watch-models",
        nargs="*",
        default=[],
        help="Model names used only for observation.",
    )
    consensus_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output text report or JSON payload.",
    )
    consensus_parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a recommendation CSV to the daily archive directory.",
    )
    consensus_parser.add_argument(
        "--archive-dir",
        default=None,
        help="CSV archive root directory. Defaults to ~/Documents/A-Stock/daily_recommendations.",
    )
    consensus_parser.add_argument(
        "--csv-date",
        default=None,
        help="CSV date folder. Defaults to --as-of-date; set this to the next trading day if needed.",
    )
    consensus_parser.add_argument(
        "--csv-universe-name",
        default=None,
        help="Optional CSV universe filename prefix. Defaults to the selected models' universe name.",
    )

    archive_parser = subparsers.add_parser(
        "archive-daily-consensus",
        help="Archive HS300/ZZ500/ZZ1000 consensus recommendations for the next trading day.",
    )
    archive_parser.add_argument(
        "--signal-date",
        default=None,
        help="Signal date in YYYYMMDD format. Defaults to today in Asia/Shanghai.",
    )
    archive_parser.add_argument(
        "--archive-dir",
        default=None,
        help="Archive root directory. Defaults to ~/Documents/A-Stock/daily_recommendations.",
    )
    archive_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output compact text status or JSON payload.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"run", "resume"}:
        config = load_research_config(args.config)
        results = run_experiments(
            config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            resume=args.command == "resume",
        )
        print(f"completed_results={len(results)}")
    elif args.command == "recommend":
        recommendation = generate_daily_recommendations_from_run(
            args.run_dir,
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            top_k_models=args.top_k_models,
            top_k_stocks=args.top_k_stocks,
        )
        if args.output == "json":
            print(json.dumps(recommendation, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_recommendation_text(recommendation))
    elif args.command == "active-pullback":
        recommendation = generate_active_pullback_recommendations(
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            universe_name=args.universe_name,
            top_n=args.top_n,
            benchmark_code=args.benchmark_code,
            return_quantile=args.return_quantile,
            turnover_quantile=args.turnover_quantile,
            amount_quantile=args.amount_quantile,
            money_flow_quantile=args.money_flow_quantile,
            exclude_chinext=args.exclude_chinext,
        )
        if args.output == "json":
            print(json.dumps(recommendation, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_active_pullback_text(recommendation))
    elif args.command == "recommend-consensus":
        recommendation = generate_daily_consensus_recommendations(
            args.run_dir,
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            core_model_names=args.core_models,
            confirm_model_names=args.confirm_models,
            watch_model_names=args.watch_models,
        )
        if args.write_csv:
            recommendation["csv_archive"] = archive_consensus_recommendation_csv(
                recommendation,
                universe_name=args.csv_universe_name,
                archive_dir=args.archive_dir,
                archive_date=args.csv_date,
            )
        if args.output == "json":
            print(json.dumps(recommendation, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_consensus_recommendation_text(recommendation))
            if args.write_csv:
                csv_archive = recommendation["csv_archive"]
                print("")
                print(f"csv_path: {csv_archive['csv_path']}")
    elif args.command == "archive-daily-consensus":
        result = archive_daily_consensus_recommendations(
            signal_date=args.signal_date,
            archive_dir=args.archive_dir,
        )
        if args.output == "json":
            payload = {key: value for key, value in result.items() if key != "reports"}
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        elif result.get("status") == "skipped":
            print(
                f"status=skipped signal_date={result.get('signal_date')} "
                f"reason={result.get('reason')}"
            )
        else:
            print(
                f"status=completed signal_date={result.get('signal_date')} "
                f"target_trade_date={result.get('target_trade_date')} "
                f"archive_path={result.get('archive_path')}"
            )
    else:
        summary = rebuild_summary_from_disk(args.run_dir)
        ascending = args.sort_by == "max_drawdown"
        ranked = sort_research_summary(summary, by=args.sort_by, ascending=ascending)
        if args.output == "json":
            print(json.dumps(ranked.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str))
        else:
            print(ranked.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
