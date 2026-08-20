"""CLI entrypoint for the minimal quant pipeline."""

from __future__ import annotations

import argparse
import json

from pipeline.main import run_minimal_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal A-share quant pipeline.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--universe-name",
        choices=[
            "all_a",
            "all_a_ex_chinext_st",
            "hs300",
            "zz500",
            "zz1000",
            "zz2000",
            "sse50",
            "main_board",
            "chinext",
        ],
        help="Universe name to resolve dynamically.",
    )
    source_group.add_argument(
        "--ts-codes",
        nargs="+",
        help="Explicit ts_code list such as 000001.SZ 600000.SH.",
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--top-n", type=int, default=20, help="Top N stocks to select per signal date.")
    parser.add_argument("--benchmark-code", default="000300.SH", help="Benchmark index code.")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output text report or compact JSON summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = run_minimal_pipeline(
        ts_codes=args.ts_codes,
        universe_name=args.universe_name,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        benchmark_code=args.benchmark_code,
    )

    if args.output == "json":
        payload = {
            "latest_selection": result["latest_selection"],
            "performance": result["performance"],
            "report": result["report"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(result["report_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
