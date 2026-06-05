"""Active-stock pullback recommendation entrypoint."""

from __future__ import annotations

from datetime import datetime, timedelta

from pipeline import run_recommendation_pipeline


DEFAULT_ACTIVE_PULLBACK_FACTOR_CONFIG = {
    "pullback_after_trend_60d": 0.45,
    "return_5d_negative": 0.25,
    "down_day_absorption_20d": 0.20,
    "close_to_high_20d": 0.10,
}


def build_active_pullback_signal_filters(
    *,
    return_quantile: float = 0.70,
    turnover_quantile: float = 0.70,
    amount_quantile: float | None = None,
    money_flow_quantile: float | None = None,
) -> list[dict[str, object]]:
    """Build active-stock pool filters for recent strength and activity."""

    _validate_quantile(return_quantile, "return_quantile")
    _validate_quantile(turnover_quantile, "turnover_quantile")
    filters: list[dict[str, object]] = [
        {"factor": "return_20d", "op": "quantile_gte", "value": return_quantile},
        {"factor": "turnover_mean_20d", "op": "quantile_gte", "value": turnover_quantile},
    ]
    if amount_quantile is not None:
        _validate_quantile(amount_quantile, "amount_quantile")
        filters.append({"factor": "amount_mean_20d", "op": "quantile_gte", "value": amount_quantile})
    if money_flow_quantile is not None:
        _validate_quantile(money_flow_quantile, "money_flow_quantile")
        filters.append({"factor": "money_flow_strength_20d", "op": "quantile_gte", "value": money_flow_quantile})
    return filters


def generate_active_pullback_recommendations(
    *,
    as_of_date: str,
    start_date: str | None = None,
    universe_name: str = "zz1000",
    top_n: int = 30,
    benchmark_code: str = "000852.SH",
    return_quantile: float = 0.70,
    turnover_quantile: float = 0.70,
    amount_quantile: float | None = None,
    money_flow_quantile: float | None = None,
    factor_config: dict[str, float] | None = None,
) -> dict[str, object]:
    """Generate active recent-winner pullback candidates using the daily recommendation path.

    Time alignment:
    - Factors are computed as of ``as_of_date`` using available historical daily data.
    - This entrypoint does not change the repository's backtest/rebalance logic.
    - A production trading workflow should execute after confirming the next A-share
      trading session is tradable under suspension/limit constraints.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    start_date = start_date or _calendar_days_before(as_of_date, 240)
    signal_filters = build_active_pullback_signal_filters(
        return_quantile=return_quantile,
        turnover_quantile=turnover_quantile,
        amount_quantile=amount_quantile,
        money_flow_quantile=money_flow_quantile,
    )
    factor_config = factor_config or DEFAULT_ACTIVE_PULLBACK_FACTOR_CONFIG

    pipeline_result = run_recommendation_pipeline(
        universe_name=universe_name,
        start_date=start_date,
        end_date=as_of_date,
        top_n=top_n,
        benchmark_code=benchmark_code,
        factor_config=factor_config,
        signal_filters=signal_filters,
    )
    latest_selection = pipeline_result["latest_selection"]

    research_experiment = {
        "name": f"active_pullback_{universe_name}_top{top_n}",
        "universe_name": universe_name,
        "benchmark_code": benchmark_code,
        "top_n": top_n,
        "signal_filters": signal_filters,
        "factor_config": factor_config,
    }
    return {
        "as_of_date": as_of_date,
        "start_date": start_date,
        "universe_name": universe_name,
        "benchmark_code": benchmark_code,
        "top_n": top_n,
        "signal_filters": signal_filters,
        "factor_config": factor_config,
        "research_experiment": research_experiment,
        "latest_selection": latest_selection,
        "top_stocks": latest_selection.get("top_stocks", []),
    }


def render_active_pullback_text(report: dict[str, object]) -> str:
    """Render active pullback candidates as compact text."""

    lines = [
        "active pullback candidates:",
        f"universe={report.get('universe_name')} benchmark={report.get('benchmark_code')}",
        f"as_of_date={report.get('as_of_date')} start_date={report.get('start_date')} top_n={report.get('top_n')}",
        "",
        "active pool filters:",
    ]
    for rule in report.get("signal_filters", []):
        lines.append(f"{rule['factor']} {rule['op']} {rule['value']}")

    lines.extend(["", "ranking factors:"])
    factor_config = report.get("factor_config", {})
    for factor_name, weight in factor_config.items():
        lines.append(f"{factor_name}: {weight}")

    lines.extend(["", "top stocks:"])
    top_stocks = report.get("top_stocks", [])
    if not top_stocks:
        lines.append("N/A")
    else:
        displayed = 0
        for item in top_stocks:
            selected = bool(item.get("selected", False))
            if not selected:
                continue
            displayed += 1
            lines.append(
                f"{displayed}. {item['ts_code']} rank={item['rank']} "
                f"score={float(item['score']):.6f}"
            )
        if displayed == 0:
            lines.append("N/A")
    lines.extend(
        [
            "",
            "research experiment config:",
            str(report.get("research_experiment", {})),
        ]
    )
    return "\n".join(lines)


def _validate_quantile(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _calendar_days_before(date: str, days: int) -> str:
    timestamp = datetime.strptime(date, "%Y%m%d") - timedelta(days=days)
    return timestamp.strftime("%Y%m%d")
