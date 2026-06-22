"""Helpers for building theme-focused research configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

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

THEME_TAG_COLUMNS = [
    "as_of_date",
    "ts_code",
    "name",
    "industry",
    "market",
    "exchange",
    "theme",
    "sub_theme",
    "source",
    "confidence",
    "evidence",
    "valid_from",
    "valid_to",
]

DEFAULT_THEME_VALIDATION_MODELS = (
    "hstech_momentum_quality_top10",
    "zz500tech_momentum_quality_top10",
    "zz500tech_momentum_quality_top20",
)

DEFAULT_THEME_VALIDATION_WINDOWS = (
    {"name": "2025h1", "start_date": "20250101", "end_date": "20250630"},
    {"name": "2025h2", "start_date": "20250701", "end_date": "20251231"},
    {"name": "2026ytd", "start_date": "20260101", "end_date": "20260608"},
)

DEFAULT_THEME_STAGE2_BASE_MODELS = (
    "hstech_momentum_quality_top10",
    "zz500tech_momentum_quality_top10",
    "zz500tech_momentum_quality_top20",
)

DEFAULT_THEME_STAGE2_VARIANTS = (
    {
        "suffix": "trend_liquidity",
        "signal_filters": [{"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50}],
        "factor_config": {"return_60d": 0.60, "amount_mean_20d": 0.40},
    },
    {
        "suffix": "trend_liquidity_closehigh",
        "signal_filters": [{"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50}],
        "factor_config": {"return_60d": 0.55, "amount_mean_20d": 0.30, "close_to_high_20d": 0.15},
    },
    {
        "suffix": "trend_liquidity_flow",
        "signal_filters": [{"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50}],
        "factor_config": {"return_60d": 0.45, "amount_mean_20d": 0.25, "money_flow_strength_20d": 0.30},
    },
    {
        "suffix": "trend_liquidity_lowvol_check",
        "signal_filters": [{"factor": "amount_mean_20d", "op": "quantile_gte", "value": 0.50}],
        "factor_config": {"return_60d": 0.45, "amount_mean_20d": 0.35, "volatility_60d_negative": 0.20},
    },
)


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


def load_theme_taxonomy(path: str | Path) -> dict[str, object]:
    """Load a theme taxonomy YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    themes = payload.get("themes")
    if not isinstance(themes, list):
        raise ValueError("Theme taxonomy must contain a 'themes' list.")
    return payload


def build_theme_tags(
    stock_basic: pd.DataFrame,
    taxonomy: dict[str, object],
    *,
    as_of_date: str,
    source: str = "stock_basic_keyword",
) -> pd.DataFrame:
    """Tag active stocks with theme/sub-theme labels from a taxonomy."""

    if stock_basic.empty:
        return pd.DataFrame(columns=THEME_TAG_COLUMNS)

    data = _normalize_stock_basic_for_theme(stock_basic, as_of_date)
    rows: list[dict[str, object]] = []
    for _, stock in data.iterrows():
        for theme in taxonomy.get("themes", []):
            theme_name = str(theme.get("name", "")).strip()
            for sub_theme in theme.get("sub_themes", []) or []:
                sub_theme_name = str(sub_theme.get("name", "")).strip()
                evidence = _match_taxonomy_rule(
                    name=str(stock["name"]),
                    industry=str(stock["industry"]),
                    industry_keywords=tuple(sub_theme.get("industry_keywords", []) or []),
                    name_keywords=tuple(sub_theme.get("name_keywords", []) or []),
                )
                if not theme_name or not sub_theme_name or not evidence:
                    continue
                rows.append(
                    {
                        "as_of_date": as_of_date,
                        "ts_code": stock["ts_code"],
                        "name": stock["name"],
                        "industry": stock["industry"],
                        "market": stock["market"],
                        "exchange": stock["exchange"],
                        "theme": theme_name,
                        "sub_theme": sub_theme_name,
                        "source": source,
                        "confidence": float(sub_theme.get("confidence", 0.5)),
                        "evidence": ";".join(evidence),
                        "valid_from": as_of_date,
                        "valid_to": "",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=THEME_TAG_COLUMNS)
    return (
        pd.DataFrame(rows)
        .loc[:, THEME_TAG_COLUMNS]
        .sort_values(["theme", "sub_theme", "ts_code"])
        .drop_duplicates(subset=["as_of_date", "ts_code", "theme", "sub_theme", "source"], keep="last")
        .reset_index(drop=True)
    )


def filter_theme_tags(
    tags: pd.DataFrame,
    *,
    themes: Iterable[str] | None = None,
    sub_themes: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Filter theme tags by theme, sub-theme, and confidence."""

    if tags.empty:
        return pd.DataFrame(columns=THEME_TAG_COLUMNS)
    result = tags.copy()
    if themes:
        result = result.loc[result["theme"].isin(set(themes))]
    if sub_themes:
        result = result.loc[result["sub_theme"].isin(set(sub_themes))]
    result = result.loc[pd.to_numeric(result["confidence"], errors="coerce").fillna(0.0) >= float(min_confidence)]
    return result.sort_values(["theme", "sub_theme", "ts_code"]).reset_index(drop=True)


def build_theme_pool_from_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Collapse tag rows into one stock-level theme pool with evidence summaries."""

    if tags.empty:
        return pd.DataFrame(columns=["ts_code", "name", "industry", "market", "exchange", "match_reason"])

    grouped = (
        tags.groupby("ts_code", sort=True)
        .agg(
            name=("name", "last"),
            industry=("industry", "last"),
            market=("market", "last"),
            exchange=("exchange", "last"),
            match_reason=("sub_theme", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    return grouped.loc[:, ["ts_code", "name", "industry", "market", "exchange", "match_reason"]]


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
        "zz1000": "000852.SH",
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


def build_and_write_taxonomy_theme_research(
    *,
    taxonomy_path: str | Path,
    start_date: str,
    end_date: str,
    output_config_path: str | Path,
    output_pool_path: str | Path,
    output_tags_path: str | Path,
    universe_names: tuple[str, ...] = ("hs300", "zz500"),
    themes: Iterable[str] | None = None,
    sub_themes: Iterable[str] | None = None,
    min_confidence: float = 0.6,
    tag_as_of_date: str | None = None,
    pool_as_of_date: str | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    """Build theme tags, stock pools, and research YAML from a taxonomy."""

    tag_as_of_date = tag_as_of_date or start_date
    pool_as_of_date = pool_as_of_date or start_date
    taxonomy = load_theme_taxonomy(taxonomy_path)
    stock_basic = get_stock_basic_history(refresh=refresh)
    tags = build_theme_tags(stock_basic, taxonomy, as_of_date=tag_as_of_date)
    filtered_tags = filter_theme_tags(tags, themes=themes, sub_themes=sub_themes, min_confidence=min_confidence)

    tags_path = Path(output_tags_path)
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_tags.to_csv(tags_path, index=False)

    theme_pool = build_theme_pool_from_tags(filtered_tags)
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
    pool_output.insert(0, "tag_as_of_date", tag_as_of_date)
    pool_path = Path(output_pool_path)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_output.to_csv(pool_path, index=False)

    config = build_technology_theme_research_config(
        pools_by_universe=pools_by_universe,
        start_date=start_date,
        end_date=end_date,
    )
    config["global"]["pool_as_of_date"] = pool_as_of_date
    config["global"]["tag_as_of_date"] = tag_as_of_date
    config["global"]["theme_taxonomy"] = str(taxonomy_path)
    config["global"]["theme_tags_csv"] = str(tags_path)
    config["global"]["theme_pool_csv"] = str(pool_path)
    config["global"]["selected_themes"] = sorted(set(themes or []))
    config["global"]["selected_sub_themes"] = sorted(set(sub_themes or []))
    config["global"]["min_theme_confidence"] = float(min_confidence)
    config["global"]["research_notes"].insert(
        0,
        "Theme labels are taxonomy keyword tags from local stock_basic fields; they are weaker than verified concept constituents.",
    )
    write_yaml(output_config_path, config)
    return {
        "config_path": str(output_config_path),
        "pool_path": str(pool_path),
        "tags_path": str(tags_path),
        "tag_count": int(len(filtered_tags)),
        "pool_counts": {name: len(codes) for name, codes in pools_by_universe.items()},
        "experiment_count": len(config["experiments"]),
    }


def build_theme_validation_config(
    base_config: dict[str, object],
    *,
    selected_model_names: Iterable[str] = DEFAULT_THEME_VALIDATION_MODELS,
    windows: Iterable[dict[str, str]] = DEFAULT_THEME_VALIDATION_WINDOWS,
) -> dict[str, object]:
    """Build time-split validation experiments from selected theme models."""

    experiments = base_config.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("base_config must contain an experiments list.")

    model_names = list(selected_model_names)
    experiments_by_name = {
        str(experiment.get("name")): experiment
        for experiment in experiments
        if isinstance(experiment, dict) and experiment.get("name")
    }
    missing = [name for name in model_names if name not in experiments_by_name]
    if missing:
        raise ValueError(f"Selected model(s) not found in base config: {', '.join(missing)}")

    validation_windows = [_normalize_validation_window(window) for window in windows]
    base_global = copy.deepcopy(base_config.get("global", {}))
    if not isinstance(base_global, dict):
        base_global = {}
    result_global = {
        key: value
        for key, value in base_global.items()
        if key not in {"start_date", "end_date"}
    }
    result_global["validation_windows"] = validation_windows
    result_global["selected_validation_models"] = model_names
    result_global["research_notes"] = list(result_global.get("research_notes", [])) + [
        "This config validates selected theme models across fixed time windows; it does not search new parameters.",
    ]

    validation_experiments: list[dict[str, object]] = []
    for model_name in model_names:
        base_experiment = experiments_by_name[model_name]
        for window in validation_windows:
            experiment = copy.deepcopy(base_experiment)
            experiment["name"] = f"{model_name}_{window['name']}"
            experiment["start_date"] = window["start_date"]
            experiment["end_date"] = window["end_date"]
            experiment["validation_base_model"] = model_name
            experiment["validation_window"] = window["name"]
            validation_experiments.append(experiment)

    return {
        "global": result_global,
        "experiments": validation_experiments,
    }


def build_and_write_theme_validation_config(
    *,
    base_config_path: str | Path,
    output_config_path: str | Path,
    selected_model_names: Iterable[str] = DEFAULT_THEME_VALIDATION_MODELS,
    windows: Iterable[dict[str, str]] = DEFAULT_THEME_VALIDATION_WINDOWS,
) -> dict[str, object]:
    """Read a theme research config and write a time-split validation config."""

    with Path(base_config_path).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    config = build_theme_validation_config(
        base_config,
        selected_model_names=selected_model_names,
        windows=windows,
    )
    write_yaml(output_config_path, config)
    return {
        "config_path": str(output_config_path),
        "experiment_count": len(config["experiments"]),
        "selected_models": list(selected_model_names),
        "windows": list(config["global"]["validation_windows"]),
    }


def build_theme_stage2_config(
    base_config: dict[str, object],
    *,
    selected_model_names: Iterable[str] = DEFAULT_THEME_STAGE2_BASE_MODELS,
    variants: Iterable[dict[str, object]] = DEFAULT_THEME_STAGE2_VARIANTS,
) -> dict[str, object]:
    """Build a small theme research matrix for factor ablation validation."""

    experiments = base_config.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("base_config must contain an experiments list.")

    model_names = list(selected_model_names)
    experiments_by_name = {
        str(experiment.get("name")): experiment
        for experiment in experiments
        if isinstance(experiment, dict) and experiment.get("name")
    }
    missing = [name for name in model_names if name not in experiments_by_name]
    if missing:
        raise ValueError(f"Selected model(s) not found in base config: {', '.join(missing)}")

    normalized_variants = [_normalize_stage2_variant(variant) for variant in variants]
    base_global = copy.deepcopy(base_config.get("global", {}))
    if not isinstance(base_global, dict):
        base_global = {}
    result_global = copy.deepcopy(base_global)
    result_global["stage2_base_models"] = model_names
    result_global["stage2_variant_names"] = [variant["suffix"] for variant in normalized_variants]
    result_global["research_notes"] = list(result_global.get("research_notes", [])) + [
        "Stage2 tests interpretable factor ablations from validation results; it is not a broad parameter search.",
        "Keep the original universe, benchmark, execution constraints, and rebalance assumptions unchanged.",
    ]

    stage2_experiments: list[dict[str, object]] = []
    for model_name in model_names:
        base_experiment = experiments_by_name[model_name]
        for variant in normalized_variants:
            experiment = copy.deepcopy(base_experiment)
            experiment["name"] = f"{model_name}_{variant['suffix']}"
            experiment["signal_filters"] = copy.deepcopy(variant["signal_filters"])
            experiment["factor_config"] = copy.deepcopy(variant["factor_config"])
            experiment["stage2_base_model"] = model_name
            experiment["stage2_variant"] = variant["suffix"]
            stage2_experiments.append(experiment)

    return {
        "global": result_global,
        "experiments": stage2_experiments,
    }


def build_and_write_theme_stage2_config(
    *,
    base_config_path: str | Path,
    output_config_path: str | Path,
    selected_model_names: Iterable[str] = DEFAULT_THEME_STAGE2_BASE_MODELS,
    variants: Iterable[dict[str, object]] = DEFAULT_THEME_STAGE2_VARIANTS,
) -> dict[str, object]:
    """Read a theme research config and write a small stage2 ablation config."""

    with Path(base_config_path).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    config = build_theme_stage2_config(
        base_config,
        selected_model_names=selected_model_names,
        variants=variants,
    )
    write_yaml(output_config_path, config)
    return {
        "config_path": str(output_config_path),
        "experiment_count": len(config["experiments"]),
        "selected_models": list(selected_model_names),
        "variants": list(config["global"]["stage2_variant_names"]),
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


def _normalize_validation_window(window: dict[str, str]) -> dict[str, str]:
    name = str(window.get("name", "")).strip()
    start_date = str(window.get("start_date", "")).strip()
    end_date = str(window.get("end_date", "")).strip()
    if not name or not start_date or not end_date:
        raise ValueError("Each validation window must contain name, start_date, and end_date.")
    if start_date > end_date:
        raise ValueError(f"Validation window start_date must be <= end_date: {name}")
    return {"name": name, "start_date": start_date, "end_date": end_date}


def _normalize_stage2_variant(variant: dict[str, object]) -> dict[str, object]:
    suffix = str(variant.get("suffix", "")).strip()
    signal_filters = variant.get("signal_filters", [])
    factor_config = variant.get("factor_config", {})
    if not suffix:
        raise ValueError("Each stage2 variant must contain suffix.")
    if not isinstance(signal_filters, list):
        raise ValueError(f"stage2 variant signal_filters must be a list: {suffix}")
    if not isinstance(factor_config, dict) or not factor_config:
        raise ValueError(f"stage2 variant factor_config must be a non-empty dict: {suffix}")
    return {
        "suffix": suffix,
        "signal_filters": copy.deepcopy(signal_filters),
        "factor_config": copy.deepcopy(factor_config),
    }


def _normalize_stock_basic_for_theme(stock_basic: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    data = stock_basic.copy()
    for column in ["ts_code", "name", "industry", "market", "exchange", "list_date", "delist_date"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)

    listed = data["list_date"].ne("") & (data["list_date"] <= as_of_date)
    not_delisted = data["delist_date"].eq("") | (data["delist_date"] > as_of_date)
    return data.loc[listed & not_delisted].sort_values("ts_code").drop_duplicates(subset=["ts_code"], keep="last")


def _match_taxonomy_rule(
    *,
    name: str,
    industry: str,
    industry_keywords: tuple[str, ...],
    name_keywords: tuple[str, ...],
) -> list[str]:
    evidence: list[str] = []
    for keyword in industry_keywords:
        if keyword and keyword in industry:
            evidence.append(f"industry:{keyword}")
    for keyword in name_keywords:
        if keyword and keyword in name:
            evidence.append(f"name:{keyword}")
    return evidence
