"""Minimal end-to-end quant pipeline."""

from __future__ import annotations

from functools import reduce

import pandas as pd

from analysis import (
    build_factor_diagnostics_report,
    calc_factor_coverage,
    calc_forward_returns,
    calc_ic,
    calc_quantile_returns,
    calc_rank_ic,
    render_factor_report_text,
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


def _build_market_panel(
    *,
    ts_codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    market_frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        data = get_a_share_daily_prices(ts_code=ts_code, start_date=start_date, end_date=end_date)
        market_frames.append(data.loc[:, ["trade_date", "ts_code", "open", "close"]].copy())

    if not market_frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "open", "close"])
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
    enable_factor_diagnostics: bool = False,
    analysis_horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, object]:
    benchmark_data = get_a_share_index_daily(
        ts_code=benchmark_code,
        start_date=start_date,
        end_date=end_date,
    )
    trade_dates = benchmark_data["trade_date"].astype(str).drop_duplicates().sort_values().tolist()
    rebalance_schedule = get_rebalance_schedule(trade_dates)

    factor_config = factor_config or {
        "return_20d": 1.0,
        "volatility_20d": -1.0,
        "turnover_mean_20d": 1.0,
    }
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
        factor_config=factor_config,
    )
    scored = combine_factor_scores(
        factor_panel,
        factor_cols=list(factor_config.keys()),
        directions={factor_name: 1 if weight >= 0 else -1 for factor_name, weight in factor_config.items()},
        weights={factor_name: abs(weight) for factor_name, weight in factor_config.items()},
    )
    ranked = rank_signal(scored, score_col="score")
    if universe_name and not ts_codes:
        ranked = _filter_ranked_by_universe_history(ranked, universe_name, rebalance_schedule["signal_date"].tolist())
    selected = top_n_selection(ranked, top_n=top_n)

    market_panel = _build_market_panel(ts_codes=resolved_ts_codes, start_date=start_date, end_date=end_date)
    strategy_returns, holdings = run_backtest(
        signals=selected.loc[:, ["trade_date", "ts_code", "selected"]],
        market_data=market_panel,
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
        },
    )
    report_text = render_strategy_report_text(report)

    result = {
        "factor_data": factor_panel,
        "scored_signals": scored,
        "selected_signals": selected,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "returns_with_benchmark": returns_with_benchmark,
        "holdings": holdings,
        "performance": performance,
        "latest_selection": latest_selection,
        "report": report,
        "report_text": report_text,
    }
    if enable_factor_diagnostics:
        factor_diagnostics, factor_report_text = _build_factor_diagnostics_outputs(
            factor_panel=factor_panel,
            market_panel=market_panel,
            factor_names=list(factor_config.keys()),
            analysis_horizons=analysis_horizons,
        )
        result["factor_diagnostics"] = factor_diagnostics
        result["factor_report_text"] = factor_report_text

    return result


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
