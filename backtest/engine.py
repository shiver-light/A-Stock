"""Minimal daily backtest engine."""

from __future__ import annotations

import pandas as pd


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(market_data, ["trade_date", "ts_code", "open", "close"])
    data = market_data.loc[:, ["trade_date", "ts_code", "open", "close"]].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data = data.sort_values(["trade_date", "ts_code"]).drop_duplicates(subset=["trade_date", "ts_code"])
    return data.reset_index(drop=True)


def _prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(signals, ["trade_date", "ts_code", "selected"])
    data = signals.loc[:, ["trade_date", "ts_code", "selected"]].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["selected"] = data["selected"].astype(bool)
    data = data.sort_values(["trade_date", "ts_code"]).drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    return data.reset_index(drop=True)


def _monthly_first_trade_dates(trade_dates: list[str]) -> list[str]:
    series = pd.Series(sorted(trade_dates), name="trade_date")
    months = series.str.slice(0, 6)
    return series.groupby(months).first().tolist()


def _execution_schedule(trade_dates: list[str]) -> pd.DataFrame:
    monthly_exec_dates = _monthly_first_trade_dates(trade_dates)
    order_map = {trade_date: index for index, trade_date in enumerate(sorted(trade_dates))}
    schedule_rows: list[dict[str, str]] = []
    for exec_date in monthly_exec_dates:
        exec_index = order_map[exec_date]
        if exec_index == 0:
            continue
        signal_date = sorted(trade_dates)[exec_index - 1]
        schedule_rows.append({"signal_date": signal_date, "execution_date": exec_date})
    return pd.DataFrame(schedule_rows)


def get_rebalance_schedule(trade_dates: list[str]) -> pd.DataFrame:
    """Return the monthly execution schedule under the current backtest assumptions."""
    return _execution_schedule(trade_dates)


def generate_weights(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
    selected_col: str = "selected",
) -> pd.DataFrame:
    market = _prepare_market_data(market_data).rename(columns={"trade_date": date_col, "ts_code": asset_col})
    signal_data = _prepare_signals(signals).rename(columns={"trade_date": date_col, "ts_code": asset_col, "selected": selected_col})
    trade_dates = market[date_col].drop_duplicates().sort_values().tolist()
    schedule = _execution_schedule(trade_dates)
    if schedule.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "target_weight", "signal_date"])

    selected_signals = signal_data.loc[signal_data[selected_col], [date_col, asset_col]].copy()
    weight_rows: list[dict[str, object]] = []
    for row in schedule.itertuples(index=False):
        picks = selected_signals.loc[selected_signals[date_col] == row.signal_date, asset_col].tolist()
        if not picks:
            continue
        weight = 1.0 / len(picks)
        for ts_code in picks:
            weight_rows.append(
                {
                    "trade_date": row.execution_date,
                    "ts_code": ts_code,
                    "target_weight": weight,
                    "signal_date": row.signal_date,
                }
            )
    return pd.DataFrame(weight_rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def run_backtest(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    fee_bps: float = 10.0,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _prepare_market_data(market_data).rename(columns={"trade_date": date_col, "ts_code": asset_col})
    weights = generate_weights(signals, market.rename(columns={date_col: "trade_date", asset_col: "ts_code"}))

    trade_dates = market[date_col].drop_duplicates().sort_values().tolist()
    close_wide = market.pivot(index=date_col, columns=asset_col, values="close").sort_index()
    open_wide = market.pivot(index=date_col, columns=asset_col, values="open").sort_index()

    weight_by_exec = {}
    for exec_date, group in weights.groupby("trade_date"):
        weight_by_exec[exec_date] = dict(zip(group["ts_code"], group["target_weight"]))

    previous_close_weights: dict[str, float] = {}
    records: list[dict[str, float | str]] = []
    holdings_rows: list[dict[str, float | str]] = []

    for idx, trade_date in enumerate(trade_dates):
        exec_weights = weight_by_exec.get(trade_date)
        turnover = 0.0
        cost = 0.0
        gross_return = 0.0

        if exec_weights is not None:
            all_assets = sorted(set(previous_close_weights) | set(exec_weights))
            turnover = sum(abs(exec_weights.get(asset, 0.0) - previous_close_weights.get(asset, 0.0)) for asset in all_assets)
            cost = turnover * fee_bps / 10000.0

            for asset, weight in exec_weights.items():
                asset_open = open_wide.loc[trade_date, asset]
                asset_close = close_wide.loc[trade_date, asset]
                if pd.isna(asset_open) or pd.isna(asset_close) or asset_open == 0:
                    continue
                gross_return += weight * (asset_close / asset_open - 1.0)
            end_of_day_weights = exec_weights.copy()
        else:
            if idx == 0:
                end_of_day_weights = {}
            else:
                prev_date = trade_dates[idx - 1]
                for asset, weight in previous_close_weights.items():
                    prev_close = close_wide.loc[prev_date, asset] if asset in close_wide.columns else pd.NA
                    today_close = close_wide.loc[trade_date, asset] if asset in close_wide.columns else pd.NA
                    if pd.isna(prev_close) or pd.isna(today_close) or prev_close == 0:
                        continue
                    gross_return += weight * (today_close / prev_close - 1.0)
                end_of_day_weights = previous_close_weights.copy()

        strategy_return = gross_return - cost
        records.append(
            {
                "trade_date": trade_date,
                "strategy_return": strategy_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
            }
        )
        for asset, weight in end_of_day_weights.items():
            holdings_rows.append({"trade_date": trade_date, "ts_code": asset, "weight": weight})
        previous_close_weights = end_of_day_weights

    returns = pd.DataFrame(records)
    holdings = pd.DataFrame(holdings_rows)
    return returns, holdings


def calc_benchmark_returns(
    benchmark_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    _validate_columns(benchmark_data, [date_col, "open", "close", "pre_close"])
    data = benchmark_data.loc[:, [date_col, "open", "close", "pre_close"]].copy()
    data[date_col] = data[date_col].astype(str)
    data = data.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    data["benchmark_return"] = data["close"] / data["pre_close"] - 1.0
    return data.loc[:, [date_col, "benchmark_return"]]


def attach_benchmark(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    _validate_columns(strategy_returns, [date_col, "strategy_return"])
    _validate_columns(benchmark_returns, [date_col, "benchmark_return"])
    merged = strategy_returns.merge(benchmark_returns, on=date_col, how="left")
    merged["benchmark_return"] = merged["benchmark_return"].fillna(0.0)
    merged["excess_return"] = merged["strategy_return"] - merged["benchmark_return"]
    return merged.sort_values(date_col).reset_index(drop=True)
