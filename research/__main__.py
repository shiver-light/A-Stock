"""CLI entrypoint for batch research."""

from __future__ import annotations

import argparse
import json

from research import (
    generate_daily_consensus_recommendations,
    generate_daily_recommendations_from_run,
    load_research_config,
    rebuild_summary_from_disk,
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
        default=["c01_hs300_turnover_top10"],
        help="Model names used as the main trade core.",
    )
    consensus_parser.add_argument(
        "--confirm-models",
        nargs="*",
        default=["c03_hs300_turnover_ret60_70_30_top20"],
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
    elif args.command == "recommend-consensus":
        recommendation = generate_daily_consensus_recommendations(
            args.run_dir,
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            core_model_names=args.core_models,
            confirm_model_names=args.confirm_models,
            watch_model_names=args.watch_models,
        )
        if args.output == "json":
            print(json.dumps(recommendation, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_consensus_recommendation_text(recommendation))
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
