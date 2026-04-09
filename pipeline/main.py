"""Minimal end-to-end quant pipeline."""

from __future__ import annotations

from functools import reduce

import pandas as pd

from backtest import attach_benchmark, calc_benchmark_returns, calc_performance, calc_relative_performance, run_backtest
from data import get_a_share_daily_prices, get_a_share_index_daily
from factors import return_20d_factor, turnover_mean_20d_factor, volatility_20d_factor
from reports import format_latest_selection, format_strategy_report, render_strategy_report_text
from signals import combine_factor_scores, rank_signal, top_n_selection
from universe import get_universe


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
) -> pd.DataFrame:
    factor_frames: list[pd.DataFrame] = []
    for ts_code in ts_codes:
        frames = [
            _factor_to_wide(
                return_20d_factor,
                factor_name="return_20d",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            ),
            _factor_to_wide(
                volatility_20d_factor,
                factor_name="volatility_20d",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            ),
            _factor_to_wide(
                turnover_mean_20d_factor,
                factor_name="turnover_mean_20d",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            ),
        ]
        merged = reduce(
            lambda left, right: left.merge(right, on=["trade_date", "ts_code"], how="inner"),
            frames,
        )
        factor_frames.append(merged)

    if not factor_frames:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "return_20d", "volatility_20d", "turnover_mean_20d"]
        )
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
) -> dict[str, object]:
    resolved_ts_codes = _resolve_ts_codes(
        ts_codes=ts_codes,
        universe_name=universe_name,
        as_of_date=end_date,
    )
    factor_panel = build_factor_panel(ts_codes=resolved_ts_codes, start_date=start_date, end_date=end_date)
    scored = combine_factor_scores(
        factor_panel,
        factor_cols=["return_20d", "volatility_20d", "turnover_mean_20d"],
        directions={"return_20d": 1, "volatility_20d": -1, "turnover_mean_20d": 1},
    )
    ranked = rank_signal(scored, score_col="score")
    selected = top_n_selection(ranked, top_n=top_n)

    market_panel = _build_market_panel(ts_codes=resolved_ts_codes, start_date=start_date, end_date=end_date)
    strategy_returns, holdings = run_backtest(
        signals=selected.loc[:, ["trade_date", "ts_code", "selected"]],
        market_data=market_panel,
    )
    benchmark_data = get_a_share_index_daily(
        ts_code=benchmark_code,
        start_date=start_date,
        end_date=end_date,
    )
    benchmark_returns = calc_benchmark_returns(benchmark_data)
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
            "factor_config": {
                "return_20d": 1.0,
                "volatility_20d": -1.0,
                "turnover_mean_20d": 1.0,
            },
        },
    )
    report_text = render_strategy_report_text(report)

    return {
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


def _resolve_ts_codes(
    *,
    ts_codes: list[str] | None,
    universe_name: str | None,
    as_of_date: str,
) -> list[str]:
    if ts_codes:
        return sorted(set(ts_codes))
    if not universe_name:
        raise ValueError("Either ts_codes or universe_name must be provided.")
    universe = get_universe(universe_name, as_of_date)
    return universe["ts_code"].drop_duplicates().sort_values().tolist()
