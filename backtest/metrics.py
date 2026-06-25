"""Performance metrics for daily strategy return series."""

from __future__ import annotations

import math

import pandas as pd


FORWARD_MAX_GAIN_WINDOWS = {
    "1d": 1,
    "3d": 3,
    "7d": 7,
    "1w": 5,
}


def _monthly_compounded_returns(
    data: pd.DataFrame,
    *,
    date_col: str,
    return_col: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["month", return_col])

    monthly = data.loc[:, [date_col, return_col]].copy()
    monthly[date_col] = pd.to_datetime(monthly[date_col].astype(str))
    monthly["month"] = monthly[date_col].dt.to_period("M").astype(str)
    monthly = (
        monthly.groupby("month", sort=True)[return_col]
        .apply(lambda values: float((1.0 + values.fillna(0.0)).prod() - 1.0))
        .reset_index()
    )
    return monthly


def _build_rolling_5m_metrics(
    monthly_returns: pd.DataFrame,
    *,
    return_col: str,
) -> pd.DataFrame:
    if monthly_returns.empty or len(monthly_returns) < 5:
        return pd.DataFrame(columns=["month", "rolling_5m_excess_return", "rolling_5m_sharpe"])

    rows: list[dict[str, float | str]] = []
    for end_idx in range(4, len(monthly_returns)):
        window = monthly_returns.iloc[end_idx - 4 : end_idx + 1].copy()
        window_returns = window[return_col].fillna(0.0)
        rolling_excess_return = float((1.0 + window_returns).prod() - 1.0)

        volatility = float(window_returns.std(ddof=0))
        rolling_sharpe = 0.0
        if volatility > 0:
            rolling_sharpe = float((window_returns.mean() / volatility) * math.sqrt(12))

        rows.append(
            {
                "month": str(window["month"].iloc[-1]),
                "rolling_5m_excess_return": rolling_excess_return,
                "rolling_5m_sharpe": rolling_sharpe,
            }
        )

    return pd.DataFrame(rows)


def _summarize_rolling_metric(
    rolling_metrics: pd.DataFrame,
    *,
    value_col: str,
    prefix: str,
) -> dict[str, float]:
    if rolling_metrics.empty:
        return {
            f"latest_{prefix}": 0.0,
            f"mean_{prefix}": 0.0,
            f"worst_{prefix}": 0.0,
        }

    values = rolling_metrics[value_col].fillna(0.0)
    return {
        f"latest_{prefix}": float(values.iloc[-1]),
        f"mean_{prefix}": float(values.mean()),
        f"worst_{prefix}": float(values.min()),
    }


def _build_forward_max_gain_metrics(
    data: pd.DataFrame,
    *,
    date_col: str,
    return_col: str,
) -> pd.DataFrame:
    """Build ex-post forward max gain diagnostics from strategy returns."""
    columns = ["trade_date"] + [
        f"forward_{label}_max_gain"
        for label in FORWARD_MAX_GAIN_WINDOWS
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    returns = data[return_col].fillna(0.0).reset_index(drop=True)
    dates = data[date_col].astype(str).reset_index(drop=True)
    rows: list[dict[str, float | str]] = []

    for idx, trade_date in dates.items():
        row: dict[str, float | str] = {"trade_date": trade_date}
        for label, window in FORWARD_MAX_GAIN_WINDOWS.items():
            future_returns = returns.iloc[idx + 1 : idx + window + 1]
            value = math.nan
            if len(future_returns) == window:
                cumulative_returns = [
                    float((1.0 + future_returns.iloc[: step]).prod() - 1.0)
                    for step in range(1, window + 1)
                ]
                value = max(cumulative_returns)
            row[f"forward_{label}_max_gain"] = value
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _summarize_forward_max_gain(forward_metrics: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for label in FORWARD_MAX_GAIN_WINDOWS:
        col = f"forward_{label}_max_gain"
        values = forward_metrics[col].dropna() if col in forward_metrics.columns else pd.Series(dtype="float64")
        if values.empty:
            summary[f"latest_{col}"] = 0.0
            summary[f"mean_{col}"] = 0.0
            summary[f"best_{col}"] = 0.0
            continue
        summary[f"latest_{col}"] = float(values.iloc[-1])
        summary[f"mean_{col}"] = float(values.mean())
        summary[f"best_{col}"] = float(values.max())
    return summary


def _records_with_none_for_nan(data: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in data.to_dict(orient="records"):
        records.append(
            {
                key: None if pd.isna(value) else value
                for key, value in record.items()
            }
        )
    return records


def calc_performance(
    strategy_returns: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    return_col: str = "strategy_return",
) -> dict[str, object]:
    if strategy_returns.empty:
        return {
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "mean_daily_turnover": 0.0,
            "mean_rebalance_turnover": 0.0,
            "median_rebalance_turnover": 0.0,
            "max_rebalance_turnover": 0.0,
            "positive_month_ratio": 0.0,
            "forward_max_gain": [],
            **_summarize_forward_max_gain(pd.DataFrame()),
        }

    data = strategy_returns.loc[:, [date_col, return_col]].copy()
    data = data.sort_values(date_col).reset_index(drop=True)
    daily_returns = data[return_col].fillna(0.0)

    equity = (1.0 + daily_returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    num_days = len(daily_returns)
    annual_return = float(equity.iloc[-1] ** (252 / num_days) - 1.0) if num_days > 0 else 0.0

    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    volatility = float(daily_returns.std(ddof=0))
    sharpe = 0.0
    if volatility > 0:
        sharpe = float((daily_returns.mean() / volatility) * math.sqrt(252))

    turnover_series = (
        strategy_returns["turnover"].fillna(0.0)
        if "turnover" in strategy_returns.columns
        else pd.Series(0.0, index=strategy_returns.index, dtype="float64")
    )
    rebalance_turnover = turnover_series.loc[turnover_series > 0]

    monthly_returns = _monthly_compounded_returns(data, date_col=date_col, return_col=return_col)
    positive_month_ratio = 0.0
    if not monthly_returns.empty:
        positive_month_ratio = float((monthly_returns[return_col] > 0).mean())

    forward_max_gain = _build_forward_max_gain_metrics(
        data,
        date_col=date_col,
        return_col=return_col,
    )

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "mean_daily_turnover": float(turnover_series.mean()) if not turnover_series.empty else 0.0,
        "mean_rebalance_turnover": float(rebalance_turnover.mean()) if not rebalance_turnover.empty else 0.0,
        "median_rebalance_turnover": float(rebalance_turnover.median()) if not rebalance_turnover.empty else 0.0,
        "max_rebalance_turnover": float(rebalance_turnover.max()) if not rebalance_turnover.empty else 0.0,
        "positive_month_ratio": positive_month_ratio,
        "forward_max_gain": _records_with_none_for_nan(forward_max_gain),
        **_summarize_forward_max_gain(forward_max_gain),
    }


def calc_relative_performance(
    returns_with_benchmark: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    benchmark_col: str = "benchmark_return",
    excess_col: str = "excess_return",
) -> dict[str, object]:
    if returns_with_benchmark.empty:
        return {
            "benchmark_cumulative_return": 0.0,
            "excess_cumulative_return": 0.0,
            "excess_annual_return": 0.0,
            "positive_excess_month_ratio": 0.0,
            "rolling_5m_excess_return": [],
            "latest_rolling_5m_excess_return": 0.0,
            "mean_rolling_5m_excess_return": 0.0,
            "worst_rolling_5m_excess_return": 0.0,
            "rolling_5m_sharpe": [],
            "latest_rolling_5m_sharpe": 0.0,
            "mean_rolling_5m_sharpe": 0.0,
            "worst_rolling_5m_sharpe": 0.0,
        }

    data = returns_with_benchmark.loc[:, [date_col, benchmark_col, excess_col]].copy()
    data = data.sort_values(date_col).reset_index(drop=True)

    benchmark_equity = (1.0 + data[benchmark_col].fillna(0.0)).cumprod()
    excess_equity = (1.0 + data[excess_col].fillna(0.0)).cumprod()

    benchmark_cumulative_return = float(benchmark_equity.iloc[-1] - 1.0)
    excess_cumulative_return = float(excess_equity.iloc[-1] - 1.0)
    num_days = len(data)
    excess_annual_return = float(excess_equity.iloc[-1] ** (252 / num_days) - 1.0) if num_days > 0 else 0.0
    monthly_excess_returns = _monthly_compounded_returns(data, date_col=date_col, return_col=excess_col)
    positive_excess_month_ratio = 0.0
    if not monthly_excess_returns.empty:
        positive_excess_month_ratio = float((monthly_excess_returns[excess_col] > 0).mean())

    rolling_5m_metrics = _build_rolling_5m_metrics(monthly_excess_returns, return_col=excess_col)
    excess_summary = _summarize_rolling_metric(
        rolling_5m_metrics,
        value_col="rolling_5m_excess_return",
        prefix="rolling_5m_excess_return",
    )
    sharpe_summary = _summarize_rolling_metric(
        rolling_5m_metrics,
        value_col="rolling_5m_sharpe",
        prefix="rolling_5m_sharpe",
    )

    return {
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "excess_cumulative_return": excess_cumulative_return,
        "excess_annual_return": excess_annual_return,
        "positive_excess_month_ratio": positive_excess_month_ratio,
        "rolling_5m_excess_return": rolling_5m_metrics.loc[:, ["month", "rolling_5m_excess_return"]].to_dict(orient="records"),
        "rolling_5m_sharpe": rolling_5m_metrics.loc[:, ["month", "rolling_5m_sharpe"]].to_dict(orient="records"),
        **excess_summary,
        **sharpe_summary,
    }
