"""Build US-market regime CSV for A-share research.

The output is aligned to A-share trade dates using only US closes from strictly
earlier US trade dates. This avoids using same-calendar-date US closes that are
not known when the A-share signal is generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.us_market import USMarketDailyService, USDailyRequest, build_us_market_regime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research/us_market_regime.csv from Tushare us_daily data.")
    parser.add_argument("--start-date", required=True, help="A-share output start date, YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="A-share output end date, YYYYMMDD.")
    parser.add_argument("--output", default="research/us_market_regime.csv", help="Output CSV path.")
    parser.add_argument("--cache-dir", default="data/cache/tushare", help="Tushare cache directory.")
    parser.add_argument("--lookback", type=int, default=20, help="Rolling lookback in US trading days.")
    parser.add_argument("--spy-code", default="SPY", help="SPY-like US risk proxy code in Tushare.")
    parser.add_argument("--qqq-code", default="QQQ", help="QQQ-like US tech proxy code in Tushare.")
    parser.add_argument("--iwm-code", default="IWM", help="IWM-like US small-cap proxy code in Tushare.")
    parser.add_argument("--vix-proxy-code", default="VXX", help="VIX stress proxy code in Tushare.")
    parser.add_argument("--vix-stress-threshold", type=float, default=0.1)
    parser.add_argument("--tech-relative-threshold", type=float, default=0.0)
    parser.add_argument("--smallcap-relative-threshold", type=float, default=0.0)
    parser.add_argument(
        "--theme-code",
        action="append",
        default=[],
        metavar="NAME=CODE",
        help="Optional theme ETF code, for example semiconductor=SOXX. Can be repeated.",
    )
    parser.add_argument("--theme-relative-threshold", type=float, default=0.0)
    parser.add_argument("--refresh", action="store_true", help="Refresh cached Tushare data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = USMarketDailyService(cache_dir=args.cache_dir)

    fetch_start = _calendar_days_before(args.start_date, max(args.lookback * 4, 90))
    fetch_end = args.end_date
    theme_codes = _parse_theme_codes(args.theme_code)
    codes = _unique_codes([args.spy_code, args.qqq_code, args.iwm_code, args.vix_proxy_code, *theme_codes.values()])

    frames: list[pd.DataFrame] = []
    for code in codes:
        request = USDailyRequest(
            ts_code=code,
            start_date=fetch_start,
            end_date=fetch_end,
            refresh=args.refresh,
        )
        frames.append(service.get_daily(request))

    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    a_share_dates = service.get_a_share_open_dates(start_date=args.start_date, end_date=args.end_date)
    regime = build_us_market_regime(
        prices,
        a_share_dates,
        spy_code=args.spy_code,
        qqq_code=args.qqq_code,
        iwm_code=args.iwm_code,
        vix_proxy_code=args.vix_proxy_code,
        lookback=args.lookback,
        tech_relative_threshold=args.tech_relative_threshold,
        smallcap_relative_threshold=args.smallcap_relative_threshold,
        vix_stress_threshold=args.vix_stress_threshold,
        theme_codes=theme_codes,
        theme_relative_threshold=args.theme_relative_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regime.to_csv(output_path, index=False)
    print(f"wrote {len(regime)} rows to {output_path}")
    if not regime.empty:
        print(f"date range: {regime['trade_date'].min()} -> {regime['trade_date'].max()}")
    return 0


def _calendar_days_before(date: str, days: int) -> str:
    timestamp = pd.to_datetime(date, format="%Y%m%d") - pd.Timedelta(days=days)
    return timestamp.strftime("%Y%m%d")


def _parse_theme_codes(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--theme-code must use NAME=CODE format: {item}")
        name, code = item.split("=", 1)
        name = name.strip()
        code = code.strip()
        if not name or not code:
            raise ValueError(f"--theme-code must use non-empty NAME=CODE format: {item}")
        result[name] = code
    return result


def _unique_codes(codes: list[str]) -> list[str]:
    result: list[str] = []
    for code in codes:
        if code not in result:
            result.append(code)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
