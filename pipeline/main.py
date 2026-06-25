"""Minimal end-to-end quant pipeline."""

from __future__ import annotations

from functools import reduce

import pandas as pd

from analysis import (
    build_factor_diagnostics_report,
    build_external_regime_flags,
    calc_factor_coverage,
    calc_forward_returns,
    calc_ic,
    calc_quantile_returns,
    calc_rank_ic,
    render_factor_report_text,
    load_external_regime_data,
    summarize_complete_case_count_by_date,
    summarize_factor_coverage,
    summarize_universe_count_by_date,
)
from backtest import (
    attach_benchmark,
    calc_benchmark_returns,
    calc_performance,
    calc_relative_performance,
    get_rebalance_schedule,
    run_backtest,
)
from data import get_a_share_daily_prices, get_a_share_index_daily
from factors.library import get_factor_function
from reports import format_latest_selection, format_strategy_report, render_strategy_report_text
from signals import combine_factor_scores, rank_signal, top_n_selection
from universe import get_universe, get_universe_history


def _factor_to_wide(
    factor_func,
    *,
    factor_name: str,
    ts_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    data = factor_func(ts_code=ts_code, start_date=start_date, end_date=end_date)
    data = data.loc[:, ["trade_date", "ts_code", "factor_value"]].copy()
    return data.rename(columns={"factor_value": factor_name})


def build_factor_panel(
    *,
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    factor_config: dict[str, float] | None = None,
) -> pd.DataFrame:
    factor_config = factor_config or {
        "return_20d": 1.0,
        "volatility_20d": -1.0,
        "turnover_mean_20d": 1.0,
    }

    factor_frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        frames = []
        for factor_name in factor_config:
            try:
                factor_func = get_factor_function(factor_name)
            except KeyError as exc:
                raise ValueError(f"Unsupported factor in factor_config: {factor_name}") from exc
            frames.append(
                _factor_to_wide(
                    factor_func,
                    factor_name=factor_name,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        merged = reduce(
            lambda left, right: left.merge(right, on=["trade_date", "ts_code"], how="inner"),
            frames,
        )
        factor_frames.append(merged)

    if not factor_frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", *factor_config.keys()])
    return (
        pd.concat(factor_frames, ignore_index=True)
        .sort_values(["trade_date", "ts_code"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _build_raw_factor_panel(
    *,
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    factor_config: dict[str, float] | None = None,
) -> pd.DataFrame:
    factor_config = factor_config or {
        "return_20d": 1.0,
        "volatility_20d": -1.0,
        "turnover_mean_20d": 1.0,
    }

    factor_frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        frames = []
        for factor_name in factor_config:
            try:
                factor_func = get_factor_function(factor_name)
            except KeyError as exc:
                raise ValueError(f"Unsupported factor in factor_config: {factor_name}") from exc
            frames.append(
                _factor_to_wide(
                    factor_func,
                    factor_name=factor_name,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        merged = reduce(
            lambda left, right: left.merge(right, on=["trade_date", "ts_code"], how="outer"),
            frames,
        )
        factor_frames.append(merged)

    if not factor_frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", *factor_config.keys()])
    return (
        pd.concat(factor_frames, ignore_index=True)
        .sort_values(["trade_date", "ts_code"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _get_signal_filter_factor_names(signal_filters: list[dict[str, object]] | None) -> list[str]:
    if not signal_filters:
        return []
    factor_names: list[str] = []
    for rule in signal_filters:
        factor_name = rule.get("factor")
        if not isinstance(factor_name, str) or not factor_name:
            raise ValueError("Each signal filter must contain a non-empty 'factor'.")
        if factor_name not in factor_names:
            factor_names.append(factor_name)
    return factor_names


def _merge_factor_configs(
    factor_config: dict[str, float],
    signal_filters: list[dict[str, object]] | None,
) -> dict[str, float]:
    merged = dict(factor_config)
    for factor_name in _get_signal_filter_factor_names(signal_filters):
        merged.setdefault(factor_name, 1.0)
    return merged


def _apply_signal_filters(
    factor_panel: pd.DataFrame,
    signal_filters: list[dict[str, object]] | None,
    *,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
) -> pd.DataFrame:
    """Apply per-date cross-sectional signal filters before score ranking.

    Supported operators:
    - quantile_gte: keep rows with factor >= same-date quantile threshold
    - quantile_lte: keep rows with factor <= same-date quantile threshold
    """

    if not signal_filters:
        return factor_panel.copy()
    if factor_panel.empty:
        return factor_panel.copy()

    result = factor_panel.copy()
    required_columns = {date_col, asset_col}
    missing_base = required_columns.difference(result.columns)
    if missing_base:
        raise ValueError(f"Missing required columns for signal filters: {sorted(missing_base)}")

    keep = pd.Series(True, index=result.index)
    for rule in signal_filters:
        factor_name = rule.get("factor")
        op = rule.get("op")
        value = rule.get("value")
        if not isinstance(factor_name, str) or not factor_name:
            raise ValueError("Each signal filter must contain a non-empty 'factor'.")
        if factor_name not in result.columns:
            raise ValueError(f"Signal filter factor is missing from factor panel: {factor_name}")
        if op not in {"quantile_gte", "quantile_lte"}:
            raise ValueError(f"Unsupported signal filter op: {op}")
        try:
            threshold_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Signal filter value must be numeric: {value}") from exc
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("Signal filter quantile value must be between 0 and 1.")

        thresholds = result.groupby(date_col)[factor_name].transform(lambda series: series.quantile(threshold_value))
        if op == "quantile_gte":
            rule_keep = result[factor_name].notna() & thresholds.notna() & (result[factor_name] >= thresholds)
        else:
            rule_keep = result[factor_name].notna() & thresholds.notna() & (result[factor_name] <= thresholds)
        keep = keep & rule_keep

    return result.loc[keep].sort_values([date_col, asset_col]).reset_index(drop=True)


def _build_market_regime_flags(
    benchmark_data: pd.DataFrame,
    market_regime_filter: dict[str, object] | None,
    *,
    reference_index_data: dict[str, pd.DataFrame] | None = None,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Build per-date benchmark regime flags using only historical close data."""

    if not market_regime_filter or market_regime_filter.get("enabled", True) is False:
        return pd.DataFrame(columns=[date_col, "market_regime_allowed"])
    if benchmark_data.empty:
        return pd.DataFrame(columns=[date_col, "market_regime_allowed"])
    if date_col not in benchmark_data.columns or "close" not in benchmark_data.columns:
        raise ValueError("market_regime_filter requires benchmark_data with trade_date and close columns.")

    data = benchmark_data.loc[:, [date_col, "close"]].copy()
    data[date_col] = data[date_col].astype(str)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    allowed = pd.Series(True, index=data.index)

    return_keys = [key for key in market_regime_filter if key.startswith("min_return_") and key.endswith("d")]
    for key in return_keys:
        if key not in market_regime_filter:
            continue
        window = int(key.removeprefix("min_return_").removesuffix("d"))
        if window <= 0:
            raise ValueError(f"market_regime_filter {key} window must be positive.")
        threshold = float(market_regime_filter[key])
        returns = data["close"].pct_change(window)
        allowed = allowed & returns.notna() & (returns >= threshold)

    if "close_above_ma" in market_regime_filter:
        window = int(market_regime_filter["close_above_ma"])
        if window <= 0:
            raise ValueError("market_regime_filter close_above_ma must be positive.")
        moving_average = data["close"].rolling(window).mean()
        allowed = allowed & moving_average.notna() & data["close"].notna() & (data["close"] >= moving_average)

    relative_rules = market_regime_filter.get("relative_strength")
    if relative_rules:
        rules = relative_rules if isinstance(relative_rules, list) else [relative_rules]
        reference_index_data = reference_index_data or {}
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("market_regime_filter relative_strength entries must be dictionaries.")
            reference_code = rule.get("reference_code")
            if not isinstance(reference_code, str) or not reference_code:
                raise ValueError("market_regime_filter relative_strength requires reference_code.")
            if reference_code not in reference_index_data:
                raise ValueError(f"Missing reference index data for relative_strength: {reference_code}")
            reference_data = reference_index_data[reference_code]
            if date_col not in reference_data.columns or "close" not in reference_data.columns:
                raise ValueError("relative_strength reference data must contain trade_date and close columns.")
            reference = reference_data.loc[:, [date_col, "close"]].copy()
            reference[date_col] = reference[date_col].astype(str)
            reference["reference_close"] = pd.to_numeric(reference["close"], errors="coerce")
            reference = (
                reference.drop(columns=["close"])
                .sort_values(date_col)
                .drop_duplicates(subset=[date_col], keep="last")
                .reset_index(drop=True)
            )
            relative = data.loc[:, [date_col, "close"]].merge(reference, on=date_col, how="left")
            relative_strength = relative["close"] / relative["reference_close"]

            relative_return_keys = [key for key in rule if key.startswith("min_return_") and key.endswith("d")]
            for key in relative_return_keys:
                window = int(key.removeprefix("min_return_").removesuffix("d"))
                if window <= 0:
                    raise ValueError(f"relative_strength {key} window must be positive.")
                threshold = float(rule[key])
                relative_return = relative_strength.pct_change(window)
                allowed = allowed & relative_return.notna() & (relative_return >= threshold)

            if "ratio_above_ma" in rule:
                window = int(rule["ratio_above_ma"])
                if window <= 0:
                    raise ValueError("relative_strength ratio_above_ma must be positive.")
                relative_ma = relative_strength.rolling(window).mean()
                allowed = allowed & relative_strength.notna() & relative_ma.notna() & (relative_strength >= relative_ma)

    return data.loc[:, [date_col]].assign(market_regime_allowed=allowed.astype(bool))


def _get_market_regime_reference_codes(market_regime_filter: dict[str, object] | None) -> list[str]:
    if not market_regime_filter or market_regime_filter.get("enabled", True) is False:
        return []
    relative_rules = market_regime_filter.get("relative_strength")
    if not relative_rules:
        return []
    rules = relative_rules if isinstance(relative_rules, list) else [relative_rules]
    codes: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        reference_code = rule.get("reference_code")
        if isinstance(reference_code, str) and reference_code and reference_code not in codes:
            codes.append(reference_code)
    return codes


def _load_market_regime_reference_data(
    market_regime_filter: dict[str, object] | None,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    reference_data: dict[str, pd.DataFrame] = {}
    for reference_code in _get_market_regime_reference_codes(market_regime_filter):
        reference_data[reference_code] = get_a_share_index_daily(
            ts_code=reference_code,
            start_date=start_date,
            end_date=end_date,
        )
    return reference_data


def _apply_market_regime_filter(
    selection: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    market_regime_filter: dict[str, object] | None,
    *,
    reference_index_data: dict[str, pd.DataFrame] | None = None,
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not market_regime_filter or market_regime_filter.get("enabled", True) is False:
        return selection.copy(), pd.DataFrame(columns=[date_col, "market_regime_allowed"])
    if selection.empty:
        return selection.copy(), _build_market_regime_flags(
            benchmark_data,
            market_regime_filter,
            reference_index_data=reference_index_data,
            date_col=date_col,
        )

    flags = _build_market_regime_flags(
        benchmark_data,
        market_regime_filter,
        reference_index_data=reference_index_data,
        date_col=date_col,
    )
    result = selection.copy()
    result[date_col] = result[date_col].astype(str)
    result = result.merge(flags, on=date_col, how="left")
    result["market_regime_allowed"] = result["market_regime_allowed"].fillna(False).astype(bool)
    result["selected"] = result["selected"] & result["market_regime_allowed"]
    result = result.drop(columns=["market_regime_allowed"])
    return result.sort_values([date_col, "rank", "ts_code"]).reset_index(drop=True), flags


def _load_external_regime_flags(
    external_regime_filter: dict[str, object] | None,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    if not external_regime_filter or external_regime_filter.get("enabled", True) is False:
        return pd.DataFrame(columns=[date_col, "external_regime_allowed"])
    path = external_regime_filter.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("external_regime_filter requires a non-empty path.")
    data = load_external_regime_data(path)
    return build_external_regime_flags(data, external_regime_filter, date_col=date_col)


def _apply_external_regime_filter(
    selection: pd.DataFrame,
    external_regime_filter: dict[str, object] | None,
    *,
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not external_regime_filter or external_regime_filter.get("enabled", True) is False:
        return selection.copy(), pd.DataFrame(columns=[date_col, "external_regime_allowed"])

    flags = _load_external_regime_flags(external_regime_filter, date_col=date_col)
    if selection.empty:
        return selection.copy(), flags

    result = selection.copy()
    result[date_col] = result[date_col].astype(str)
    result = result.merge(flags, on=date_col, how="left")
    result["external_regime_allowed"] = result["external_regime_allowed"].fillna(False).astype(bool)
    result["selected"] = result["selected"] & result["external_regime_allowed"]
    result = result.drop(columns=["external_regime_allowed"])
    return result.sort_values([date_col, "rank", "ts_code"]).reset_index(drop=True), flags


def _build_market_panel(
    *,
    ts_codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    market_frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        data = get_a_share_daily_prices(ts_code=ts_code, start_date=start_date, end_date=end_date)
        available_columns = [column for column in ["trade_date", "ts_code", "open", "close", "amount"] if column in data.columns]
        market_frames.append(data.loc[:, available_columns].copy())

    if not market_frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "open", "close", "amount"])
    return (
        pd.concat(market_frames, ignore_index=True)
        .sort_values(["trade_date", "ts_code"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def run_minimal_pipeline(
    *,
    ts_codes: list[str] | None = None,
    start_date: str,
    end_date: str,
    top_n: int = 20,
    benchmark_code: str = "000300.SH",
    universe_name: str | None = None,
    factor_config: dict[str, float] | None = None,
    signal_filters: list[dict[str, object]] | None = None,
    market_regime_filter: dict[str, object] | None = None,
    external_regime_filter: dict[str, object] | None = None,
    rebalance_frequency: str = "monthly",
    backtest_config: dict[str, object] | None = None,
    enable_factor_diagnostics: bool = False,
    enable_data_diagnostics: bool = False,
    analysis_horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, object]:
    benchmark_data = get_a_share_index_daily(
        ts_code=benchmark_code,
        start_date=start_date,
        end_date=end_date,
    )
    backtest_config = dict(backtest_config or {})
    if "rebalance_frequency" in backtest_config:
        rebalance_frequency = str(backtest_config.pop("rebalance_frequency"))
    trade_dates = benchmark_data["trade_date"].astype(str).drop_duplicates().sort_values().tolist()
    rebalance_schedule = get_rebalance_schedule(trade_dates, rebalance_frequency=rebalance_frequency)

    factor_config = factor_config or {
        "return_20d": 1.0,
        "volatility_20d": -1.0,
        "turnover_mean_20d": 1.0,
    }
    panel_factor_config = _merge_factor_configs(factor_config, signal_filters)
    resolved_ts_codes = _resolve_ts_codes(
        ts_codes=ts_codes,
        universe_name=universe_name,
        as_of_date=end_date,
        signal_dates=rebalance_schedule["signal_date"].tolist() if not rebalance_schedule.empty else None,
    )
    factor_panel = build_factor_panel(
        ts_codes=resolved_ts_codes,
        start_date=start_date,
        end_date=end_date,
        factor_config=panel_factor_config,
    )
    filtered_factor_panel = _apply_signal_filters(factor_panel, signal_filters)
    scored = combine_factor_scores(
        filtered_factor_panel,
        factor_cols=list(factor_config.keys()),
        directions={factor_name: 1 if weight >= 0 else -1 for factor_name, weight in factor_config.items()},
        weights={factor_name: abs(weight) for factor_name, weight in factor_config.items()},
    )
    ranked = rank_signal(scored, score_col="score")
    if universe_name and not ts_codes:
        ranked = _filter_ranked_by_universe_history(ranked, universe_name, rebalance_schedule["signal_date"].tolist())
    selected = top_n_selection(ranked, top_n=top_n)
    market_regime_reference_data = _load_market_regime_reference_data(
        market_regime_filter,
        start_date=start_date,
        end_date=end_date,
    )
    selected, market_regime = _apply_market_regime_filter(
        selected,
        benchmark_data,
        market_regime_filter,
        reference_index_data=market_regime_reference_data,
    )
    selected, external_regime = _apply_external_regime_filter(selected, external_regime_filter)

    market_panel = _build_market_panel(ts_codes=resolved_ts_codes, start_date=start_date, end_date=end_date)
    has_market_regime = bool(market_regime_filter and market_regime_filter.get("enabled", True) is not False)
    has_external_regime = bool(external_regime_filter and external_regime_filter.get("enabled", True) is not False)
    if has_market_regime or has_external_regime:
        backtest_config = {**backtest_config, "cash_on_empty_signal": True}
    strategy_returns, holdings = run_backtest(
        signals=selected.loc[:, ["trade_date", "ts_code", "selected"]],
        market_data=market_panel,
        rebalance_frequency=rebalance_frequency,
        **backtest_config,
    )
    benchmark_returns = calc_benchmark_returns(
        benchmark_data,
        execution_dates=rebalance_schedule["execution_date"].tolist() if not rebalance_schedule.empty else None,
    )
    returns_with_benchmark = attach_benchmark(strategy_returns, benchmark_returns)

    performance = calc_performance(strategy_returns)
    performance.update(calc_relative_performance(returns_with_benchmark))

    latest_date = None
    if not selected.empty:
        latest_date = str(selected["trade_date"].max())
    latest_selection = format_latest_selection(selected, top_n=top_n, as_of_date=latest_date)
    report = format_strategy_report(
        latest_selection,
        performance,
        config={
            "universe": resolved_ts_codes,
            "universe_name": universe_name or "custom",
            "benchmark_code": benchmark_code,
            "selection_logic": "等权综合因子打分后按日排序取Top N",
            "factor_config": factor_config,
            "signal_filters": signal_filters or [],
            "market_regime_filter": market_regime_filter or {},
            "external_regime_filter": external_regime_filter or {},
            "rebalance_frequency": rebalance_frequency,
            "backtest_config": backtest_config,
        },
    )
    report_text = render_strategy_report_text(report)

    result = {
        "factor_data": factor_panel,
        "filtered_factor_data": filtered_factor_panel,
        "scored_signals": scored,
        "selected_signals": selected,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "returns_with_benchmark": returns_with_benchmark,
        "market_regime": market_regime,
        "external_regime": external_regime,
        "holdings": holdings,
        "performance": performance,
        "latest_selection": latest_selection,
        "report": report,
        "report_text": report_text,
    }
    if enable_data_diagnostics:
        raw_factor_panel = _build_raw_factor_panel(
            ts_codes=resolved_ts_codes,
            start_date=start_date,
            end_date=end_date,
            factor_config=panel_factor_config,
        )
        result["diagnostics"] = _build_data_diagnostics_outputs(
            raw_factor_panel=raw_factor_panel,
            factor_names=list(panel_factor_config.keys()),
        )
    if enable_factor_diagnostics:
        factor_diagnostics, factor_report_text = _build_factor_diagnostics_outputs(
            factor_panel=factor_panel,
            market_panel=market_panel,
            factor_names=list(panel_factor_config.keys()),
            analysis_horizons=analysis_horizons,
        )
        result["factor_diagnostics"] = factor_diagnostics
        result["factor_report_text"] = factor_report_text

    return result


def run_recommendation_pipeline(
    *,
    ts_codes: list[str] | None = None,
    start_date: str,
    end_date: str,
    top_n: int = 20,
    universe_name: str | None = None,
    factor_config: dict[str, float] | None = None,
    signal_filters: list[dict[str, object]] | None = None,
    benchmark_code: str = "000300.SH",
    market_regime_filter: dict[str, object] | None = None,
    external_regime_filter: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run a daily recommendation pipeline using the latest available trading date.

    This path is intentionally separate from the monthly backtest pipeline:
    - backtest uses monthly rebalance signal dates
    - recommendation uses the latest available daily factor cross-section as of end_date
    """

    factor_config = factor_config or {
        "return_20d": 1.0,
        "volatility_20d": -1.0,
        "turnover_mean_20d": 1.0,
    }
    panel_factor_config = _merge_factor_configs(factor_config, signal_filters)
    resolved_ts_codes = _resolve_ts_codes(
        ts_codes=ts_codes,
        universe_name=universe_name,
        as_of_date=end_date,
        signal_dates=None,
    )
    factor_panel = build_factor_panel(
        ts_codes=resolved_ts_codes,
        start_date=start_date,
        end_date=end_date,
        factor_config=panel_factor_config,
    )
    filtered_factor_panel = _apply_signal_filters(factor_panel, signal_filters)
    scored = combine_factor_scores(
        filtered_factor_panel,
        factor_cols=list(factor_config.keys()),
        directions={factor_name: 1 if weight >= 0 else -1 for factor_name, weight in factor_config.items()},
        weights={factor_name: abs(weight) for factor_name, weight in factor_config.items()},
    )
    ranked = rank_signal(scored, score_col="score")
    selected = top_n_selection(ranked, top_n=top_n)
    market_regime = pd.DataFrame(columns=["trade_date", "market_regime_allowed"])
    if market_regime_filter and market_regime_filter.get("enabled", True) is not False:
        benchmark_data = get_a_share_index_daily(
            ts_code=benchmark_code,
            start_date=start_date,
            end_date=end_date,
        )
        market_regime_reference_data = _load_market_regime_reference_data(
            market_regime_filter,
            start_date=start_date,
            end_date=end_date,
        )
        selected, market_regime = _apply_market_regime_filter(
            selected,
            benchmark_data,
            market_regime_filter,
            reference_index_data=market_regime_reference_data,
        )
    selected, external_regime = _apply_external_regime_filter(selected, external_regime_filter)

    latest_date = None
    if not selected.empty:
        latest_date = str(selected["trade_date"].astype(str).max())
    latest_selection = format_latest_selection(selected, top_n=top_n, as_of_date=latest_date)

    return {
        "factor_data": factor_panel,
        "filtered_factor_data": filtered_factor_panel,
        "scored_signals": scored,
        "ranked_signals": ranked,
        "selected_signals": selected,
        "market_regime": market_regime,
        "external_regime": external_regime,
        "latest_selection": latest_selection,
    }


def _resolve_ts_codes(
    *,
    ts_codes: list[str] | None,
    universe_name: str | None,
    as_of_date: str,
    signal_dates: list[str] | None = None,
) -> list[str]:
    if ts_codes:
        return sorted(set(ts_codes))
    if not universe_name:
        raise ValueError("Either ts_codes or universe_name must be provided.")
    if signal_dates:
        universe = get_universe_history(universe_name, signal_dates)
    else:
        universe = get_universe(universe_name, as_of_date)
    return universe["ts_code"].drop_duplicates().sort_values().tolist()


def _filter_ranked_by_universe_history(
    ranked: pd.DataFrame,
    universe_name: str,
    signal_dates: list[str],
) -> pd.DataFrame:
    if not signal_dates:
        return ranked
    universe_history = get_universe_history(universe_name, signal_dates)
    allowed = universe_history.loc[:, ["as_of_date", "ts_code"]].rename(columns={"as_of_date": "trade_date"})
    filtered = ranked.merge(allowed, on=["trade_date", "ts_code"], how="inner")
    return filtered.sort_values(["trade_date", "rank", "ts_code"]).reset_index(drop=True)


def _build_factor_diagnostics_outputs(
    *,
    factor_panel: pd.DataFrame,
    market_panel: pd.DataFrame,
    factor_names: list[str],
    analysis_horizons: tuple[int, ...],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    unique_horizons = tuple(sorted({int(horizon) for horizon in analysis_horizons if int(horizon) > 0}))
    if not unique_horizons:
        raise ValueError("analysis_horizons must contain at least one positive integer.")

    if market_panel.empty:
        forward_returns = calc_forward_returns(
            pd.DataFrame(columns=["trade_date", "ts_code", "close"]),
            list(unique_horizons),
        )
    else:
        forward_returns = calc_forward_returns(
            market_panel.loc[:, ["trade_date", "ts_code", "close"]].copy(),
            list(unique_horizons),
        )

    diagnostics: dict[str, list[dict[str, object]]] = {}
    report_text: dict[str, str] = {}
    for factor_name in factor_names:
        if factor_name not in factor_panel.columns:
            diagnostics[factor_name] = []
            report_text[factor_name] = ""
            continue

        factor_data = (
            factor_panel.loc[:, ["trade_date", "ts_code", factor_name]]
            .rename(columns={factor_name: "factor_value"})
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )
        if factor_data.empty or factor_data["factor_value"].dropna().empty:
            diagnostics[factor_name] = []
            report_text[factor_name] = ""
            continue

        coverage_data = calc_factor_coverage(factor_data)
        ic_data = calc_ic(factor_data, forward_returns)
        rank_ic_data = calc_rank_ic(factor_data, forward_returns)
        quantile_returns = calc_quantile_returns(factor_data, forward_returns)

        factor_reports = [
            build_factor_diagnostics_report(
                factor_name=factor_name,
                horizon=horizon,
                ic_data=ic_data,
                rank_ic_data=rank_ic_data,
                coverage_data=coverage_data,
                quantile_returns=quantile_returns,
            )
            for horizon in unique_horizons
        ]
        diagnostics[factor_name] = factor_reports
        report_text[factor_name] = "\n\n".join(
            render_factor_report_text(report) for report in factor_reports
        )

    return diagnostics, report_text


def _build_data_diagnostics_outputs(
    *,
    raw_factor_panel: pd.DataFrame,
    factor_names: list[str],
) -> dict[str, object]:
    return {
        "notes": {
            "original_universe_count": "Per-date unique ts_code count in the raw outer-merged factor panel before complete-case filtering.",
            "factor_available_count": "Per-date non-null observations for each configured factor within the raw outer-merged factor panel.",
            "complete_case_count": "Per-date ts_code count where all configured factors are non-null; this is the sample pool entering score combination.",
        },
        "universe_count_by_date": summarize_universe_count_by_date(raw_factor_panel),
        "factor_coverage": summarize_factor_coverage(raw_factor_panel, factor_cols=factor_names),
        "complete_case_count_by_date": summarize_complete_case_count_by_date(
            raw_factor_panel,
            factor_cols=factor_names,
        ),
    }
