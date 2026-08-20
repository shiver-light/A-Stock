"""CLI for low-position accumulation feature research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.accumulation_research import (
    DEFAULT_ACCUMULATION_FEATURES,
    apply_accumulation_score,
    build_accumulation_dataset,
    evaluate_condition_ladder,
    filter_research_samples,
    fit_accumulation_score_model,
    score_bucket_analysis,
    single_factor_analysis,
    split_by_time,
    summarize_feature_power,
)
from data import (
    AShareDailyMarketService,
    DailyMarketRequest,
    get_a_share_daily_valuation,
    get_a_share_index_daily,
    get_stock_basic_history,
)
from universe import get_universe_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research low-position accumulation volume-price features.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--ts-codes", nargs="+", help="Stock codes to analyze, e.g. 000001.SZ 600000.SH.")
    source_group.add_argument(
        "--universes",
        nargs="+",
        choices=["hs300", "zz500", "zz1000", "zz2000", "zz2000_ex_bj"],
        help="Universe names to resolve with historical constituents.",
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument("--output-dir", default="output/accumulation_research", help="Directory for CSV/JSON outputs.")
    parser.add_argument(
        "--sample-frequency",
        choices=["monthly", "weekly", "daily"],
        default="monthly",
        help="Dates used to resolve historical constituents when --universes is used.",
    )
    parser.add_argument(
        "--max-stocks-per-universe",
        type=int,
        default=None,
        help="Optional cap per universe for smoke tests or staged runs.",
    )
    parser.add_argument("--train-end", default="20211231", help="Training set end date.")
    parser.add_argument("--validation-end", default="20231231", help="Validation set end date.")
    parser.add_argument("--benchmark-hs300", default="000300.SH", help="HS300 benchmark code.")
    parser.add_argument("--benchmark-zz1000", default="000852.SH", help="CSI1000 benchmark code.")
    parser.add_argument("--refresh", action="store_true", help="Refresh Tushare cache.")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Print text or JSON summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_membership = pd.DataFrame()
    if args.universes:
        ts_codes, universe_membership = _resolve_universe_ts_codes(
            args.universes,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_frequency=args.sample_frequency,
            max_stocks_per_universe=args.max_stocks_per_universe,
            refresh=args.refresh,
        )
    else:
        ts_codes = sorted(set(args.ts_codes or []))
    market_data, turnover_data = _load_stock_data(
        ts_codes,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh,
    )
    stock_basic = get_stock_basic_history(refresh=False)
    benchmarks = {
        "hs300": get_a_share_index_daily(
            ts_code=args.benchmark_hs300,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh=args.refresh,
        ),
        "zz1000": get_a_share_index_daily(
            ts_code=args.benchmark_zz1000,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh=args.refresh,
        ),
    }
    dataset = build_accumulation_dataset(
        market_data,
        turnover_data,
        industry_map=stock_basic,
        benchmark_returns=benchmarks,
    )
    if not universe_membership.empty:
        dataset = _filter_dataset_by_universe_membership(dataset, universe_membership)
    samples = filter_research_samples(dataset)
    splits = split_by_time(samples, train_end=args.train_end, validation_end=args.validation_end)
    train_stats = single_factor_analysis(splits["train"], DEFAULT_ACCUMULATION_FEATURES)
    feature_power = summarize_feature_power(train_stats)
    model = fit_accumulation_score_model(splits["train"], feature_power)
    scored_frames = {name: apply_accumulation_score(frame, model) for name, frame in splits.items()}
    bucket_frames = {name: score_bucket_analysis(frame) for name, frame in scored_frames.items()}
    ladder = evaluate_condition_ladder(
        samples,
        [
            {"name": "低位横盘", "feature": "price_position_120d", "op": "lte", "value": 0.30},
            {"name": "上涨效率改善", "feature": "up_efficiency_trend_3_20", "op": "gte", "value": 0.0},
            {"name": "下跌效率降低", "feature": "down_efficiency_trend_3_20", "op": "lte", "value": 0.0},
            {"name": "二探缩量", "feature": "second_bottom_volume_shrink_20d", "op": "gte", "value": 1.0},
            {"name": "放量不跌", "feature": "volume_no_down_20d", "op": "quantile_gte", "value": 0.6},
            {"name": "行业抗跌", "feature": "relative_industry_20d", "op": "gte", "value": 0.0},
        ],
    )

    dataset.to_parquet(output_dir / "dataset.parquet", index=False)
    samples.to_parquet(output_dir / "samples.parquet", index=False)
    train_stats.to_csv(output_dir / "single_factor_quantiles.csv", index=False)
    feature_power.to_csv(output_dir / "feature_power.csv", index=False)
    ladder.to_csv(output_dir / "condition_ladder.csv", index=False)
    for split_name, frame in bucket_frames.items():
        frame.to_csv(output_dir / f"score_buckets_{split_name}.csv", index=False)
    if not universe_membership.empty:
        universe_membership.to_csv(output_dir / "universe_membership.csv", index=False)
    (output_dir / "score_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "universes": args.universes or [],
        "ts_code_count": int(len(ts_codes)),
        "sample_frequency": args.sample_frequency if args.universes else None,
        "sample_count": int(len(samples)),
        "positive_ratio": _safe_ratio(samples),
        "splits": {name: {"sample_count": int(len(frame)), "positive_ratio": _safe_ratio(frame)} for name, frame in splits.items()},
        "top_features": feature_power.head(10).to_dict(orient="records"),
        "score_model": model,
        "output_dir": str(output_dir),
    }
    if args.output == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(_render_text_summary(summary))
    return 0


def _load_stock_data(
    ts_codes: list[str],
    *,
    start_date: str,
    end_date: str,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_frames = []
    turnover_frames = []
    service = AShareDailyMarketService()
    for ts_code in ts_codes:
        market_frames.append(
            service.get_daily(
                DailyMarketRequest(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields=("trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount"),
                    refresh=refresh,
                )
            )
        )
        turnover_frames.append(
            get_a_share_daily_valuation(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                refresh=refresh,
            )
        )
    return pd.concat(market_frames, ignore_index=True), pd.concat(turnover_frames, ignore_index=True)


def _resolve_universe_ts_codes(
    universes: list[str],
    *,
    start_date: str,
    end_date: str,
    sample_frequency: str,
    max_stocks_per_universe: int | None,
    refresh: bool,
) -> tuple[list[str], pd.DataFrame]:
    sample_dates = _sample_dates(start_date, end_date, sample_frequency)
    frames = []
    for universe_name in universes:
        history = get_universe_history(universe_name, sample_dates, refresh=refresh)
        if history.empty:
            continue
        history = history.loc[history["in_universe"].astype(bool)].copy()
        history["source_universe"] = universe_name
        if max_stocks_per_universe is not None:
            capped_frames = []
            for as_of_date, group in history.groupby("as_of_date", sort=True):
                capped_frames.append(group.sort_values("ts_code").head(max_stocks_per_universe))
            history = pd.concat(capped_frames, ignore_index=True) if capped_frames else history.iloc[0:0].copy()
        frames.append(history)
    if not frames:
        raise ValueError("No universe members resolved.")
    membership = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["source_universe", "as_of_date", "ts_code"])
        .drop_duplicates(subset=["source_universe", "as_of_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )
    ts_codes = sorted(membership["ts_code"].dropna().astype(str).unique().tolist())
    return ts_codes, membership


def _filter_dataset_by_universe_membership(dataset: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or membership.empty:
        return dataset.copy()
    members = membership.loc[:, ["as_of_date", "ts_code", "source_universe"]].copy()
    members["as_of_date"] = members["as_of_date"].astype(str)
    members["ts_code"] = members["ts_code"].astype(str)
    data = dataset.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["ts_code"] = data["ts_code"].astype(str)
    frames = []
    for source_universe, group in members.groupby("source_universe", sort=True):
        intervals = group.sort_values(["ts_code", "as_of_date"]).copy()
        intervals["next_as_of_date"] = intervals.groupby("ts_code")["as_of_date"].shift(-1)
        for row in intervals.itertuples(index=False):
            start = str(row.as_of_date)
            end = str(row.next_as_of_date) if pd.notna(row.next_as_of_date) else "99999999"
            matched = data.loc[
                (data["ts_code"] == row.ts_code)
                & (data["trade_date"] >= start)
                & (data["trade_date"] < end)
            ].copy()
            if matched.empty:
                continue
            matched["source_universe"] = source_universe
            frames.append(matched)
    if not frames:
        return data.iloc[0:0].copy()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["source_universe", "trade_date", "ts_code"])
        .drop_duplicates(subset=["source_universe", "trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _sample_dates(start_date: str, end_date: str, frequency: str) -> list[str]:
    dates = pd.date_range(start_date, end_date, freq="D")
    if dates.empty:
        raise ValueError("No sample dates in requested range.")
    if frequency == "daily":
        sampled = dates
    elif frequency == "weekly":
        sampled = dates.to_series().groupby(dates.to_period("W-FRI")).max()
    elif frequency == "monthly":
        sampled = dates.to_series().groupby(dates.to_period("M")).max()
    else:
        raise ValueError(f"Unsupported sample frequency: {frequency}")
    return [pd.Timestamp(date).strftime("%Y%m%d") for date in sampled]


def _safe_ratio(data: pd.DataFrame) -> float | None:
    if data.empty or "label" not in data.columns:
        return None
    return float(data["label"].mean())


def _render_text_summary(summary: dict[str, object]) -> str:
    lines = [
        "【低位横盘吸筹研究MVP】",
        "",
        f"股票数：{summary['ts_code_count']}",
        f"股票池：{', '.join(summary['universes']) if summary['universes'] else 'custom ts_codes'}",
        f"成分采样频率：{summary['sample_frequency'] or 'N/A'}",
        f"样本数：{summary['sample_count']}",
        f"正样本比例：{_format_pct(summary['positive_ratio'])}",
        "",
        "时间切分：",
    ]
    for name, stats in summary["splits"].items():
        lines.append(f"{name}: 样本数={stats['sample_count']} 正样本比例={_format_pct(stats['positive_ratio'])}")
    lines.extend(["", "训练集最有效特征Top10："])
    for item in summary["top_features"]:
        lines.append(
            f"{item['feature']}: score={item['score']:.4f}, "
            f"positive_spread={item['positive_ratio_spread']:.4f}, "
            f"return20_spread={item['return_20d_spread']:.4f}"
        )
    lines.extend(["", f"输出目录：{summary['output_dir']}"])
    return "\n".join(lines)


def _format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
