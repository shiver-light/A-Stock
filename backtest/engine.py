"""Minimal daily backtest engine."""

from __future__ import annotations

import pandas as pd


OPTIONAL_MARKET_COLUMNS = [
    "amount",
    "suspended",
    "is_suspended",
    "paused",
    "up_limit",
    "down_limit",
]


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(market_data, ["trade_date", "ts_code", "open", "close"])
    keep_columns = ["trade_date", "ts_code", "open", "close"] + [
        column for column in OPTIONAL_MARKET_COLUMNS if column in market_data.columns
    ]
    data = market_data.loc[:, keep_columns].copy()
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


def _prepare_exit_signals(exit_signals: pd.DataFrame | None) -> pd.DataFrame:
    if exit_signals is None:
        return pd.DataFrame(columns=["trade_date", "ts_code", "selected"])
    return _prepare_signals(exit_signals)


def _monthly_first_trade_dates(trade_dates: list[str]) -> list[str]:
    series = pd.Series(sorted(trade_dates), name="trade_date")
    months = series.str.slice(0, 6)
    return series.groupby(months).first().tolist()


def _weekly_first_trade_dates(trade_dates: list[str]) -> list[str]:
    series = pd.Series(sorted(trade_dates), name="trade_date")
    parsed_dates = pd.to_datetime(series, format="%Y%m%d")
    weeks = parsed_dates.dt.strftime("%G%V")
    return series.groupby(weeks).first().tolist()


def _execution_dates(trade_dates: list[str], rebalance_frequency: str) -> list[str]:
    sorted_dates = sorted(trade_dates)
    if rebalance_frequency == "monthly":
        return _monthly_first_trade_dates(sorted_dates)
    if rebalance_frequency == "weekly":
        return _weekly_first_trade_dates(sorted_dates)
    if rebalance_frequency == "daily":
        return sorted_dates
    raise ValueError(f"Unsupported rebalance_frequency: {rebalance_frequency}")


def _execution_schedule(trade_dates: list[str], rebalance_frequency: str = "monthly") -> pd.DataFrame:
    exec_dates = _execution_dates(trade_dates, rebalance_frequency)
    order_map = {trade_date: index for index, trade_date in enumerate(sorted(trade_dates))}
    schedule_rows: list[dict[str, str]] = []
    for exec_date in exec_dates:
        exec_index = order_map[exec_date]
        if exec_index == 0:
            continue
        signal_date = sorted(trade_dates)[exec_index - 1]
        schedule_rows.append({"signal_date": signal_date, "execution_date": exec_date})
    return pd.DataFrame(schedule_rows)


def get_rebalance_schedule(trade_dates: list[str], rebalance_frequency: str = "monthly") -> pd.DataFrame:
    """Return execution schedule for the requested rebalance frequency."""

    return _execution_schedule(trade_dates, rebalance_frequency=rebalance_frequency)


def _market_row(market_indexed: pd.DataFrame, trade_date: str, asset: str) -> pd.Series | None:
    key = (trade_date, asset)
    if key not in market_indexed.index:
        return None
    row = market_indexed.loc[key]
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def _is_suspended(row: pd.Series | None) -> bool:
    if row is None:
        return False
    for column in ["suspended", "is_suspended", "paused"]:
        if column in row.index and not pd.isna(row[column]):
            return bool(row[column])
    return False


def _is_trade_allowed(
    row: pd.Series | None,
    *,
    side: str,
    block_suspended: bool,
    block_limit_up_buy: bool,
    block_limit_down_sell: bool,
    min_amount: float | None,
) -> bool:
    if row is None:
        return False

    asset_open = row.get("open")
    if pd.isna(asset_open) or asset_open == 0:
        return False

    if min_amount is not None and "amount" in row.index and not pd.isna(row["amount"]) and float(row["amount"]) < min_amount:
        return False

    if block_suspended and _is_suspended(row):
        return False

    if side == "buy" and block_limit_up_buy and "up_limit" in row.index and not pd.isna(row["up_limit"]):
        return float(asset_open) < float(row["up_limit"])

    if side == "sell" and block_limit_down_sell and "down_limit" in row.index and not pd.isna(row["down_limit"]):
        return float(asset_open) > float(row["down_limit"])

    return True


def _apply_execution_constraints(
    previous_close_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    trade_date: str,
    market_indexed: pd.DataFrame,
    block_suspended: bool,
    block_limit_up_buy: bool,
    block_limit_down_sell: bool,
    min_amount: float | None,
) -> dict[str, float]:
    if not previous_close_weights and not target_weights:
        return {}

    current_weights = {asset: float(weight) for asset, weight in previous_close_weights.items() if weight > 0}
    cash_weight = max(0.0, 1.0 - sum(current_weights.values()))

    sell_assets = sorted(set(current_weights) | set(target_weights))
    for asset in sell_assets:
        current_weight = current_weights.get(asset, 0.0)
        target_weight = float(target_weights.get(asset, 0.0))
        if target_weight >= current_weight:
            continue
        row = _market_row(market_indexed, trade_date, asset)
        if _is_trade_allowed(
            row,
            side="sell",
            block_suspended=block_suspended,
            block_limit_up_buy=block_limit_up_buy,
            block_limit_down_sell=block_limit_down_sell,
            min_amount=min_amount,
        ):
            cash_weight += current_weight - target_weight
            if target_weight > 0:
                current_weights[asset] = target_weight
            else:
                current_weights.pop(asset, None)

    buy_demands: list[tuple[str, float]] = []
    for asset in sorted(set(target_weights) | set(current_weights)):
        current_weight = current_weights.get(asset, 0.0)
        target_weight = float(target_weights.get(asset, 0.0))
        if target_weight <= current_weight:
            continue
        row = _market_row(market_indexed, trade_date, asset)
        if _is_trade_allowed(
            row,
            side="buy",
            block_suspended=block_suspended,
            block_limit_up_buy=block_limit_up_buy,
            block_limit_down_sell=block_limit_down_sell,
            min_amount=min_amount,
        ):
            buy_demands.append((asset, target_weight - current_weight))

    total_buy_demand = sum(weight for _, weight in buy_demands)
    if total_buy_demand > 0 and cash_weight > 0:
        scale = min(1.0, cash_weight / total_buy_demand)
        for asset, demand in buy_demands:
            executed_weight = demand * scale
            current_weights[asset] = current_weights.get(asset, 0.0) + executed_weight
            cash_weight -= executed_weight

    return {asset: weight for asset, weight in current_weights.items() if weight > 0}


def _position_exit_policy_enabled(position_exit_policy: dict[str, object] | None) -> bool:
    return bool(position_exit_policy and position_exit_policy.get("enabled", False))


def _get_hold_allowed_assets_by_date(exit_signals: pd.DataFrame) -> dict[str, set[str]]:
    if exit_signals.empty:
        return {}
    allowed = exit_signals.loc[exit_signals["selected"], ["trade_date", "ts_code"]].copy()
    return {
        str(trade_date): set(group["ts_code"].astype(str).tolist())
        for trade_date, group in allowed.groupby("trade_date", sort=True)
    }


def _apply_position_exit_policy(
    previous_close_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    signal_date: str | None,
    holding_days: dict[str, int],
    hold_allowed_assets_by_date: dict[str, set[str]],
    position_exit_policy: dict[str, object] | None,
) -> dict[str, float]:
    if not _position_exit_policy_enabled(position_exit_policy):
        return target_weights

    policy = position_exit_policy or {}
    min_hold_days = int(policy.get("min_hold_days", 0))
    max_hold_days = policy.get("max_hold_days")
    max_hold_days = int(max_hold_days) if max_hold_days is not None else None
    hold_allowed_assets = hold_allowed_assets_by_date.get(str(signal_date), set()) if signal_date else set()
    has_exit_signal_rules = bool(policy.get("exit_filters"))

    retained_assets: set[str] = set()
    for asset in previous_close_weights:
        days_held = int(holding_days.get(asset, 0))
        if max_hold_days is not None and days_held >= max_hold_days:
            continue
        if days_held < min_hold_days:
            retained_assets.add(asset)
            continue
        if not has_exit_signal_rules or asset in hold_allowed_assets:
            retained_assets.add(asset)

    forced_exit_assets = set(previous_close_weights) - retained_assets
    target_assets = {asset for asset, weight in target_weights.items() if weight > 0 and asset not in forced_exit_assets}
    final_assets = sorted(retained_assets | target_assets)
    if not final_assets:
        return {}

    weight = 1.0 / len(final_assets)
    return {asset: weight for asset in final_assets}


def _normalize_asset_gross_return(gross_return: float | int | None) -> float:
    if gross_return is None or pd.isna(gross_return) or gross_return <= 0:
        return 1.0
    return float(gross_return)


def _drift_weights_to_close(
    start_weights: dict[str, float],
    asset_gross_returns: dict[str, float],
) -> tuple[float, dict[str, float]]:
    cash_weight = max(0.0, 1.0 - sum(float(weight) for weight in start_weights.values()))
    asset_values: dict[str, float] = {}
    for asset, start_weight in start_weights.items():
        gross = _normalize_asset_gross_return(asset_gross_returns.get(asset))
        asset_values[asset] = float(start_weight) * gross

    portfolio_close_value = cash_weight + sum(asset_values.values())
    if portfolio_close_value <= 0:
        return 0.0, {}

    gross_return = portfolio_close_value - 1.0
    close_weights = {
        asset: asset_value / portfolio_close_value
        for asset, asset_value in asset_values.items()
        if asset_value > 0
    }
    return float(gross_return), close_weights


def generate_weights(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    rebalance_frequency: str = "monthly",
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
    selected_col: str = "selected",
) -> pd.DataFrame:
    market = _prepare_market_data(market_data).rename(columns={"trade_date": date_col, "ts_code": asset_col})
    signal_data = _prepare_signals(signals).rename(columns={"trade_date": date_col, "ts_code": asset_col, "selected": selected_col})
    trade_dates = market[date_col].drop_duplicates().sort_values().tolist()
    schedule = _execution_schedule(trade_dates, rebalance_frequency=rebalance_frequency)
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
    if not weight_rows:
        return pd.DataFrame(columns=["trade_date", "ts_code", "target_weight", "signal_date"])
    return pd.DataFrame(weight_rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def run_backtest(
    signals: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    rebalance_frequency: str = "monthly",
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    block_suspended: bool = False,
    block_limit_up_buy: bool = False,
    block_limit_down_sell: bool = False,
    min_amount: float | None = None,
    cash_on_empty_signal: bool = False,
    exit_signals: pd.DataFrame | None = None,
    position_exit_policy: dict[str, object] | None = None,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the current backtest with optional execution constraints.

    Constraint flags only take effect when the required market_data columns are present.
    When optional fields such as ``amount`` / ``up_limit`` / ``down_limit`` /
    ``suspended`` are absent, the engine falls back to the unconstrained behavior
    for that rule instead of failing.
    """
    market = _prepare_market_data(market_data).rename(columns={"trade_date": date_col, "ts_code": asset_col})
    signal_data = _prepare_signals(signals).rename(
        columns={"trade_date": date_col, "ts_code": asset_col, "selected": "selected"}
    )
    exit_signal_data = _prepare_exit_signals(exit_signals)
    weights = generate_weights(
        signals,
        market.rename(columns={date_col: "trade_date", asset_col: "ts_code"}),
        rebalance_frequency=rebalance_frequency,
    )

    trade_dates = market[date_col].drop_duplicates().sort_values().tolist()
    close_wide = market.pivot(index=date_col, columns=asset_col, values="close").sort_index()
    open_wide = market.pivot(index=date_col, columns=asset_col, values="open").sort_index()
    market_indexed = market.set_index([date_col, asset_col]).sort_index()
    schedule = _execution_schedule(trade_dates, rebalance_frequency=rebalance_frequency)
    signal_date_by_exec = {
        row.execution_date: row.signal_date
        for row in schedule.itertuples(index=False)
    }
    hold_allowed_assets_by_date = _get_hold_allowed_assets_by_date(exit_signal_data)

    weight_by_exec = {}
    for exec_date, group in weights.groupby("trade_date"):
        weight_by_exec[exec_date] = dict(zip(group["ts_code"], group["target_weight"]))
    if cash_on_empty_signal or _position_exit_policy_enabled(position_exit_policy):
        selected_by_date = (
            signal_data.groupby(date_col)["selected"].any()
            if not signal_data.empty
            else pd.Series(dtype=bool)
        )
        for row in schedule.itertuples(index=False):
            has_selected = bool(selected_by_date.loc[row.signal_date]) if row.signal_date in selected_by_date.index else False
            if not has_selected:
                weight_by_exec[row.execution_date] = {}

    previous_close_weights: dict[str, float] = {}
    holding_days: dict[str, int] = {}
    records: list[dict[str, float | str]] = []
    holdings_rows: list[dict[str, float | str]] = []

    for idx, trade_date in enumerate(trade_dates):
        exec_weights = weight_by_exec.get(trade_date)
        turnover = 0.0
        cost = 0.0
        gross_return = 0.0

        if exec_weights is not None:
            exec_weights = _apply_position_exit_policy(
                previous_close_weights,
                exec_weights,
                signal_date=signal_date_by_exec.get(trade_date),
                holding_days=holding_days,
                hold_allowed_assets_by_date=hold_allowed_assets_by_date,
                position_exit_policy=position_exit_policy,
            )
            realized_exec_weights = _apply_execution_constraints(
                previous_close_weights,
                exec_weights,
                trade_date=trade_date,
                market_indexed=market_indexed,
                block_suspended=block_suspended,
                block_limit_up_buy=block_limit_up_buy,
                block_limit_down_sell=block_limit_down_sell,
                min_amount=min_amount,
            )
            all_assets = sorted(set(previous_close_weights) | set(realized_exec_weights))
            turnover = sum(
                abs(realized_exec_weights.get(asset, 0.0) - previous_close_weights.get(asset, 0.0))
                for asset in all_assets
            )
            cost = turnover * (fee_bps + slippage_bps) / 10000.0
            asset_gross_returns: dict[str, float] = {}
            for asset in realized_exec_weights:
                asset_open = open_wide.loc[trade_date, asset]
                asset_close = close_wide.loc[trade_date, asset]
                if pd.isna(asset_open) or pd.isna(asset_close) or asset_open == 0 or asset_close <= 0:
                    asset_gross_returns[asset] = 1.0
                else:
                    asset_gross_returns[asset] = float(asset_close / asset_open)
            gross_return, end_of_day_weights = _drift_weights_to_close(realized_exec_weights, asset_gross_returns)
        else:
            if idx == 0:
                end_of_day_weights = {}
            else:
                prev_date = trade_dates[idx - 1]
                asset_gross_returns = {}
                for asset in previous_close_weights:
                    prev_close = close_wide.loc[prev_date, asset] if asset in close_wide.columns else pd.NA
                    today_close = close_wide.loc[trade_date, asset] if asset in close_wide.columns else pd.NA
                    if pd.isna(prev_close) or pd.isna(today_close) or prev_close == 0 or today_close <= 0:
                        asset_gross_returns[asset] = 1.0
                    else:
                        asset_gross_returns[asset] = float(today_close / prev_close)
                gross_return, end_of_day_weights = _drift_weights_to_close(previous_close_weights, asset_gross_returns)

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
        holding_days = {
            asset: holding_days.get(asset, 0) + 1
            for asset in end_of_day_weights
        }

    returns = pd.DataFrame(records)
    holdings = pd.DataFrame(holdings_rows)
    return returns, holdings


def calc_benchmark_returns(
    benchmark_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    execution_dates: list[str] | None = None,
) -> pd.DataFrame:
    _validate_columns(benchmark_data, [date_col, "open", "close", "pre_close"])
    data = benchmark_data.loc[:, [date_col, "open", "close", "pre_close"]].copy()
    data[date_col] = data[date_col].astype(str)
    data = data.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    if data.empty:
        return data.loc[:, [date_col]].assign(benchmark_return=pd.Series(dtype="float64"))

    execution_date_set = {str(trade_date) for trade_date in execution_dates or []}
    close_to_close_return = data["close"] / data["pre_close"] - 1.0
    open_to_close_return = data["close"] / data["open"] - 1.0

    data["benchmark_return"] = close_to_close_return
    exec_mask = data[date_col].isin(execution_date_set) & data["open"].notna() & (data["open"] != 0)
    data.loc[exec_mask, "benchmark_return"] = open_to_close_return.loc[exec_mask]
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
