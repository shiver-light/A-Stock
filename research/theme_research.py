"""Helpers for building theme-focused research configs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data import INDEX_CODE_MAP, get_stock_basic_history
from research.storage import write_yaml
from universe import get_universe


DEFAULT_TECH_INDUSTRY_KEYWORDS = (
    "半导体",
    "通信设备",
    "元器件",
    "软件服务",
    "IT设备",
    "电气设备",
    "电器仪表",
    "新型电力",
    "专用机械",
    "机械基件",
    "机床制造",
)

DEFAULT_TECH_NAME_KEYWORDS = (
    "PCB",
    "CPO",
    "光模块",
    "液冷",
    "算力",
    "芯片",
    "机器人",
    "服务器",
)

DEFAULT_TECH_BACKTEST_CONFIG = {
    "slippage_bps": 15.0,
    "block_suspended": True,
    "block_limit_up_buy": True,
    "block_limit_down_sell": True,
    "min_amount": 500000.0,
}


def build_theme_stock_pool(
    stock_basic: pd.DataFrame,
    *,
    as_of_date: str,
    industry_keywords: Iterable[str] = DEFAULT_TECH_INDUSTRY_KEYWORDS,
    name_keywords: Iterable[str] = DEFAULT_TECH_NAME_KEYWORDS,
) -> pd.DataFrame:
    """Build a broad technology-theme pool from stock_basic industry/name fields."""

    if stock_basic.empty:
        return pd.DataFrame(columns=["ts_code", "name", "industry", "market", "exchange", "match_reason"])

    data = stock_basic.copy()
    for column in ["ts_code", "name", "industry", "market", "exchange", "list_date", "delist_date"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)

    listed = data["list_date"].ne("") & (data["list_date"] <= as_of_date)
    not_delisted = data["delist_date"].eq("") | (data["delist_date"] > as_of_date)
    active = data.loc[listed & not_delisted].copy()

    industry_patterns = tuple(keyword for keyword in industry_keywords if keyword)
    name_patterns = tuple(keyword for keyword in name_keywords if keyword)
    active["match_reason"] = active.apply(
        lambda row: _build_match_reason(
            industry=str(row["industry"]),
            name=str(row["name"]),
            industry_keywords=industry_patterns,
            name_keywords=name_patterns,
        ),
        axis=1,
    )
    result = active.loc[active["match_reason"].ne(""), ["ts_code", "name", "industry", "market", "exchange", "match_reason"]]
    return result.sort_values("ts_code").drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)


def intersect_theme_with_universe(theme_pool: pd.DataFrame, universe: pd.DataFrame, universe_name: str) -> pd.DataFrame:
    """Intersect a theme stock pool with one index universe snapshot."""

    if theme_pool.empty or universe.empty:
        return pd.DataFrame(columns=["universe_name", "ts_code", "name", "industry", "market", "exchange", "match_reason"])

    result = universe.loc[:, ["ts_code"]].merge(theme_pool, on="ts_code", how="inner")
    result.insert(0, "universe_name", universe_name)
    return result.sort_values(["universe_name", "ts_code"]).reset_index(drop=True)


def build_technology_theme_research_config(
    *,
    pools_by_universe: dict[str, list[str]],
    start_date: str,
    end_date: str,
    benchmark_by_universe: dict[str, str] | None = None,
    top_ns: tuple[int, ...] = (10, 20, 30),
) -> dict[str, object]:
    """Build a compact research config for technology-theme stock pools."""

    benchmark_by_universe = benchmark_by_universe or {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
    }
    experiments: list[dict[str, object]] = []
    for universe_name, ts_codes in pools_by_universe.items():
        if not ts_codes:
            continue
        for top_n in top_ns:
            experiments.extend(_build_universe_experiments(universe_name, sorted(set(ts_codes)), top_n))

    return {
        "global": {
            "start_date": start_date,
            "end_date": end_date,
            "enable_factor_diagnostics": True,
            "analysis_horizons": [5, 10, 20],
            "backtest_config": DEFAULT_TECH_BACKTEST_CONFIG,
            "research_notes": [
                "Technology theme pool is a reproducible approximation built from stock_basic industry/name keywords.",
                "Index membership is snapped at the configured pool_as_of_date to avoid using end-date constituents for the full backtest.",
                "Signals follow the existing pipeline assumption: T close signal, next rebalance execution.",
            ],
        },
        "experiments": [
            {
                **experiment,
                "benchmark_code": benchmark_by_universe.get(str(experiment["universe_name"]), "000300.SH"),
            }
            for experiment in experiments
        ],
    }


def build_and_write_technology_theme_research(
    *,
    start_date: str,
    end_date: str,
    output_config_path: str | Path,
    output_pool_path: str | Path,
    universe_names: tuple[str, ...] = ("hs300", "zz500"),
    pool_as_of_date: str | None = None,
    industry_keywords: Iterable[str] = DEFAULT_TECH_INDUSTRY_KEYWORDS,
    name_keywords: Iterable[str] = DEFAULT_TECH_NAME_KEYWORDS,
    refresh: bool = False,
) -> dict[str, object]:
    """Build theme pools and write a research YAML plus a stock-pool CSV."""

    pool_as_of_date = pool_as_of_date or start_date
    stock_basic = get_stock_basic_history(refresh=refresh)
    theme_pool = build_theme_stock_pool(
        stock_basic,
        as_of_date=pool_as_of_date,
        industry_keywords=industry_keywords,
        name_keywords=name_keywords,
    )

    pool_frames: list[pd.DataFrame] = []
    pools_by_universe: dict[str, list[str]] = {}
    for universe_name in universe_names:
        universe = get_universe(universe_name, as_of_date=pool_as_of_date, refresh=refresh)
        pool = intersect_theme_with_universe(theme_pool, universe, universe_name)
        pool_frames.append(pool)
        pools_by_universe[universe_name] = pool["ts_code"].astype(str).tolist()

    pool_output = (
        pd.concat(pool_frames, ignore_index=True)
        if pool_frames
        else pd.DataFrame(columns=["universe_name", "ts_code", "name", "industry", "market", "exchange", "match_reason"])
    )
    pool_output.insert(0, "pool_as_of_date", pool_as_of_date)
    pool_path = Path(output_pool_path)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_output.to_csv(pool_path, index=False)

    config = build_technology_theme_research_config(
        pools_by_universe=pools_by_universe,
        start_date=start_date,
        end_date=end_date,
    )
    config["global"]["pool_as_of_date"] = pool_as_of_date
    config["global"]["theme_pool_csv"] = str(pool_path)
    write_yaml(output_config_path, config)
    return {
        "config_path": str(output_config_path),
        "pool_path": str(pool_path),
        "pool_counts": {name: len(codes) for name, codes in pools_by_universe.items()},
        "experiment_count": len(config["experiments"]),
    }


def _build_universe_experiments(universe_name: str, ts_codes: list[str], top_n: int) -> list[dict[str, object]]:
    prefix = "hstech" if universe_name == "hs300" else f"{universe_name}tech"
    return [
        {
            "name": f"{prefix}_active_pullback_top{top_n}",
            "universe_name": universe_name,
            "ts_codes": list(ts_codes),
            "top_n": top_n,
            "signal_filters": [
                {"factor": "return_20d", "op": "quantile_gte", "value": 0.65},
                {"factor": "money_flow_strength_20d", "op": "quantile_gte", "value": 0.55},
            ],
            "factor_config": {
                "pullback_after_trend_60d": 0.40,
                "return_5d_negative": 0.25,
                "down_day_absorption_20d": 0.20,
                "close_to_high_20d": 0.15,
            },
        },
        {
            "name": f"{prefix}_momentum_quality_top{top_n}",
            "universe_name": universe_name,
            "ts_codes": list(ts_codes),
            "top_n": top_n,
            "signal_filters": [
                {"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50},
            ],
            "factor_config": {
                "return_60d": 0.35,
                "close_to_high_20d": 0.25,
                "money_flow_strength_20d": 0.20,
                "volatility_60d_negative": 0.20,
            },
        },
        {
            "name": f"{prefix}_accumulation_risk_control_top{top_n}",
            "universe_name": universe_name,
            "ts_codes": list(ts_codes),
            "top_n": top_n,
            "signal_filters": [
                {"factor": "return_20d", "op": "quantile_gte", "value": 0.55},
                {"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50},
            ],
            "factor_config": {
                "high_turnover_low_range_20d": 0.25,
                "small_body_high_turnover_20d": 0.20,
                "down_day_absorption_20d": 0.20,
                "distribution_risk_20d_negative": 0.20,
                "gap_risk_20d_negative": 0.15,
            },
        },
    ]


def _build_match_reason(
    *,
    industry: str,
    name: str,
    industry_keywords: tuple[str, ...],
    name_keywords: tuple[str, ...],
) -> str:
    matches: list[str] = []
    for keyword in industry_keywords:
        if keyword in industry:
            matches.append(f"industry:{keyword}")
    for keyword in name_keywords:
        if keyword in name:
            matches.append(f"name:{keyword}")
    return ";".join(matches)
