"""CLI for recent up/down efficiency diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.efficiency_report import build_efficiency_report, render_efficiency_report_text
from data import (
    AShareDailyMarketService,
    DailyMarketRequest,
    get_a_share_daily_valuation,
    get_a_share_index_daily,
    get_stock_basic_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze recent up/down efficiency for one A-share stock.")
    parser.add_argument("ts_code", help="Stock code, e.g. 000001.SZ")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--start-date", default=None, help="Optional data start date in YYYYMMDD format.")
    parser.add_argument("--window", type=int, default=10, help="Number of recent trading days to analyze.")
    parser.add_argument("--recent-days", type=int, default=3, help="Recent days compared with earlier window.")
    parser.add_argument(
        "--benchmark-code",
        default="000985.CSI",
        help="Benchmark index for market context. Use 'none' to disable.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh Tushare cache.")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output text report or JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start_date = args.start_date or _default_start_date(args.end_date)
    market = AShareDailyMarketService().get_daily(
        DailyMarketRequest(
            ts_code=args.ts_code,
            start_date=start_date,
            end_date=args.end_date,
            fields=("trade_date", "ts_code", "high", "low", "close", "pre_close", "vol"),
            refresh=args.refresh,
        )
    )
    valuation = get_a_share_daily_valuation(
        ts_code=args.ts_code,
        start_date=start_date,
        end_date=args.end_date,
        refresh=args.refresh,
    )
    data = market.merge(
        valuation.loc[:, ["trade_date", "ts_code", "turnover_rate_f"]],
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    stock_label = _stock_label(args.ts_code)
    benchmark_data = None
    benchmark_label = None
    if args.benchmark_code.lower() != "none":
        benchmark_data = get_a_share_index_daily(
            ts_code=args.benchmark_code,
            start_date=start_date,
            end_date=args.end_date,
            refresh=args.refresh,
        )
        benchmark_label = args.benchmark_code
    report = build_efficiency_report(
        data,
        stock_label=stock_label,
        benchmark_data=benchmark_data,
        benchmark_label=benchmark_label,
        window=args.window,
        recent_days=args.recent_days,
    )
    if args.output == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_efficiency_report_text(report))
    return 0


def _default_start_date(end_date: str) -> str:
    return (pd.Timestamp(end_date) - pd.Timedelta(days=90)).strftime("%Y%m%d")


def _stock_label(ts_code: str) -> str:
    try:
        stock_basic = get_stock_basic_history(refresh=False)
    except Exception:
        return ts_code
    if stock_basic.empty or "ts_code" not in stock_basic.columns:
        return ts_code
    matched = stock_basic.loc[stock_basic["ts_code"].astype(str) == ts_code]
    if matched.empty or "name" not in matched.columns:
        return ts_code
    name = str(matched.iloc[0]["name"])
    return f"{name}/{ts_code}" if name else ts_code


if __name__ == "__main__":
    raise SystemExit(main())
