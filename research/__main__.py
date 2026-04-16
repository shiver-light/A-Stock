"""CLI entrypoint for batch research."""

from __future__ import annotations

import argparse
import json

from research import build_research_summary, load_research_config, run_experiments, sort_research_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch quant research experiments.")
    parser.add_argument(
        "--config",
        default="research/experiments.yaml",
        help="Path to research experiment config YAML.",
    )
    parser.add_argument(
        "--sort-by",
        default="sharpe",
        choices=["sharpe", "excess_cumulative_return", "max_drawdown"],
        help="Column to sort the summary by.",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output a plain table or JSON payload.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_research_config(args.config)
    results = run_experiments(config)
    summary = build_research_summary(results)
    ascending = args.sort_by == "max_drawdown"
    ranked = sort_research_summary(summary, by=args.sort_by, ascending=ascending)

    if args.output == "json":
        payload = {
            "results": results,
            "summary": ranked.to_dict(orient="records"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(ranked.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
