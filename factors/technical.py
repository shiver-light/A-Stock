"""Technical factor implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.daily_market import AShareDailyMarketService, DailyMarketRequest
from data.valuation_daily import get_a_share_daily_valuation
from factors.base import build_factor_output, validate_factor_input


def _buffered_start_date(start_date: str, lookback_days: int) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")


def _clip_dates(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
    return data.loc[mask].reset_index(drop=True)


def _load_qfq_market_data(
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool,
    lookback_days: int,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    service = AShareDailyMarketService()
    request_fields = tuple(dict.fromkeys(("trade_date", "ts_code", *fields)))
    request = DailyMarketRequest(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, lookback_days),
        end_date=end_date,
        adjust="qfq",
        fields=request_fields,
        refresh=refresh,
    )
    return service.get_daily(request)


def _load_qfq_daily(
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    return _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days,
        fields=("close",),
    )


def _load_turnover_data(
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    return get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, lookback_days),
        end_date=end_date,
        refresh=refresh,
    )


def _merge_market_and_turnover(
    market_data: pd.DataFrame,
    turnover_data: pd.DataFrame,
    *,
    turnover_col: str = "turnover_rate_f",
) -> pd.DataFrame:
    merged = market_data.merge(
        turnover_data.loc[:, ["trade_date", "ts_code", turnover_col]],
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    validate_factor_input(merged, ["trade_date", "ts_code", turnover_col])
    return merged


def return_5d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=15)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["return_5d"] = data.groupby("ts_code")["close"].pct_change(5)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "return_5d", "return_5d")


def return_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=45)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["return_20d"] = data.groupby("ts_code")["close"].pct_change(20)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "return_20d", "return_20d")


def return_60d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """60-day qfq return, kept separate from momentum_60d for research config readability."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["return_60d"] = data.groupby("ts_code")["close"].pct_change(60)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "return_60d", "return_60d")


def return_120d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """120-day qfq return."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=160)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["return_120d"] = data.groupby("ts_code")["close"].pct_change(120)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "return_120d", "return_120d")


def volatility_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=45)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    returns = data.groupby("ts_code")["close"].pct_change()
    data["volatility_20d"] = returns.groupby(data["ts_code"]).rolling(20).std().reset_index(level=0, drop=True)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_20d", "volatility_20d")


def volatility_60d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """60-day rolling volatility of daily qfq close returns."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    returns = data.groupby("ts_code")["close"].pct_change()
    data["volatility_60d"] = returns.groupby(data["ts_code"]).rolling(60).std().reset_index(level=0, drop=True)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_60d", "volatility_60d")


def return_5d_negative_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Negative 5-day return for short-term overheating penalty."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=15)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["return_5d_negative"] = -data.groupby("ts_code")["close"].pct_change(5)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "return_5d_negative", "return_5d_negative")


def volatility_20d_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 20-day volatility so larger values mean lower risk."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=45)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    returns = data.groupby("ts_code")["close"].pct_change()
    data["volatility_20d_negative"] = -returns.groupby(data["ts_code"]).rolling(20).std().reset_index(
        level=0, drop=True
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_20d_negative", "volatility_20d_negative")


def volatility_60d_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 60-day volatility so larger values mean lower risk."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    returns = data.groupby("ts_code")["close"].pct_change()
    data["volatility_60d_negative"] = -returns.groupby(data["ts_code"]).rolling(60).std().reset_index(
        level=0, drop=True
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_60d_negative", "volatility_60d_negative")


def max_drawdown_60d_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 60-day maximum drawdown magnitude so larger values mean lower drawdown."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["max_drawdown_60d_negative"] = (
        data.groupby("ts_code")["close"]
        .rolling(60)
        .apply(_window_max_drawdown, raw=False)
        .reset_index(level=0, drop=True)
        * -1.0
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "max_drawdown_60d_negative", "max_drawdown_60d_negative")


def price_rank_60d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])

    grouped_close = data.groupby("ts_code")["close"]
    rolling_min = grouped_close.rolling(60).min().reset_index(level=0, drop=True)
    rolling_max = grouped_close.rolling(60).max().reset_index(level=0, drop=True)
    denominator = rolling_max - rolling_min
    data["price_rank_60d"] = np.where(denominator == 0, np.nan, (data["close"] - rolling_min) / denominator)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "price_rank_60d", "price_rank_60d")


def momentum_60d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """60-day momentum defined as qfq close-to-close return over the past 60 trading days."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["momentum_60d"] = data.groupby("ts_code")["close"].pct_change(60)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "momentum_60d", "momentum_60d")


def reversal_5d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """5-day reversal defined as the negative of recent 5-day qfq return."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=15)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data["reversal_5d"] = -data.groupby("ts_code")["close"].pct_change(5)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "reversal_5d", "reversal_5d")


def turnover_volatility_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day volatility of free-float turnover rate using daily_basic.turnover_rate_f."""

    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, 45),
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "turnover_rate_f"])
    data["turnover_volatility_20d"] = data.groupby("ts_code")["turnover_rate_f"].rolling(20).std().reset_index(
        level=0,
        drop=True,
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "turnover_volatility_20d", "turnover_volatility_20d")


def amount_mean_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """20-day average daily amount."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("amount",),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "amount"])
    data["amount_mean_20d"] = data.groupby("ts_code")["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "amount_mean_20d", "amount_mean_20d")


def illiq_negative_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Negative 20-day Amihud illiquidity so larger values mean better liquidity."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("close", "amount"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "close", "amount"])
    returns = data.groupby("ts_code")["close"].pct_change().abs()
    daily_illiq = np.where(data["amount"] > 0, returns / data["amount"], np.nan)
    data["illiq_negative"] = -pd.Series(daily_illiq, index=data.index).groupby(data["ts_code"]).rolling(20).mean().reset_index(
        level=0, drop=True
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "illiq_negative", "illiq_negative")


def money_flow_strength_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day mean close-location-value times amount as a daily money flow proxy."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close", "amount"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close", "amount"])
    price_range = data["high"] - data["low"]
    clv = np.where(
        price_range == 0,
        0.0,
        ((data["close"] - data["low"]) - (data["high"] - data["close"])) / price_range,
    )
    daily_flow = clv * data["amount"]
    data["money_flow_strength_20d"] = (
        pd.Series(daily_flow, index=data.index).groupby(data["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "money_flow_strength_20d", "money_flow_strength_20d")


def high_turnover_low_range_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day mean turnover divided by 20-day mean amplitude as a stealth accumulation proxy."""

    market_data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "pre_close"),
    )
    turnover_data = _load_turnover_data(ts_code, start_date, end_date, refresh, lookback_days=45)
    data = _merge_market_and_turnover(market_data, turnover_data)
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "pre_close", "turnover_rate_f"])
    daily_amplitude = np.where(data["pre_close"] == 0, np.nan, (data["high"] - data["low"]) / data["pre_close"])
    amplitude_mean = (
        pd.Series(daily_amplitude, index=data.index).groupby(data["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    turnover_mean = data.groupby("ts_code")["turnover_rate_f"].rolling(20).mean().reset_index(level=0, drop=True)
    data["high_turnover_low_range_20d"] = np.where(amplitude_mean == 0, np.nan, turnover_mean / amplitude_mean)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "high_turnover_low_range_20d", "high_turnover_low_range_20d")


def price_suppression_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day mean turnover scaled by muted 20-day return magnitude as a price suppression proxy."""

    market_data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    turnover_data = _load_turnover_data(ts_code, start_date, end_date, refresh, lookback_days=90)
    data = _merge_market_and_turnover(market_data, turnover_data)
    validate_factor_input(data, ["trade_date", "ts_code", "close", "turnover_rate_f"])
    turnover_mean = data.groupby("ts_code")["turnover_rate_f"].rolling(20).mean().reset_index(level=0, drop=True)
    return_20d = data.groupby("ts_code")["close"].pct_change(20).abs()
    data["price_suppression_20d"] = turnover_mean / (1.0 + return_20d)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "price_suppression_20d", "price_suppression_20d")


def down_day_support_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day mean close-from-low ratio on down days as a trading support proxy."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close", "pre_close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close", "pre_close"])
    price_range = data["high"] - data["low"]
    recovery = np.where(price_range == 0, np.nan, (data["close"] - data["low"]) / price_range)
    down_day_support = np.where(data["close"] < data["pre_close"], recovery, np.nan)
    down_day_series = pd.Series(down_day_support, index=data.index, dtype="float64")
    data["down_day_support_20d"] = (
        down_day_series.groupby(data["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "down_day_support_20d", "down_day_support_20d")


def position_safety_60d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Distance below the rolling 60-day high so larger values mean less overbought positioning."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=90,
        fields=("close", "high"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "close", "high"])
    rolling_high = data.groupby("ts_code")["high"].rolling(60).max().reset_index(level=0, drop=True)
    data["position_safety_60d"] = np.where(rolling_high == 0, np.nan, 1.0 - (data["close"] / rolling_high))
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "position_safety_60d", "position_safety_60d")


def small_body_high_turnover_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day turnover intensity relative to average candle body size.

    Larger values indicate repeated turnover without large daily body expansion,
    which serves as a daily proxy for stealth accumulation via smaller orders.
    """

    market_data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("open", "close", "pre_close"),
    )
    turnover_data = _load_turnover_data(ts_code, start_date, end_date, refresh, lookback_days=45)
    data = _merge_market_and_turnover(market_data, turnover_data)
    validate_factor_input(data, ["trade_date", "ts_code", "open", "close", "pre_close", "turnover_rate_f"])
    body_ratio = np.where(data["pre_close"] == 0, np.nan, (data["close"] - data["open"]).abs() / data["pre_close"])
    body_mean = (
        pd.Series(body_ratio, index=data.index).groupby(data["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    turnover_mean = data.groupby("ts_code")["turnover_rate_f"].rolling(20).mean().reset_index(level=0, drop=True)
    data["small_body_high_turnover_20d"] = turnover_mean / (body_mean + 1e-6)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "small_body_high_turnover_20d", "small_body_high_turnover_20d")


def close_near_high_on_high_amount_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day mean close-to-high strength weighted by trading amount.

    Larger values indicate repeated heavy trading on sessions that close near
    the intraday high, which is a daily proxy for persistent buying support.
    """

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close", "amount"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close", "amount"])
    price_range = data["high"] - data["low"]
    close_location = np.where(price_range == 0, 0.0, (data["close"] - data["low"]) / price_range)
    weighted_strength = close_location * np.log1p(data["amount"].clip(lower=0.0))
    data["close_near_high_on_high_amount_20d"] = (
        pd.Series(weighted_strength, index=data.index)
        .groupby(data["ts_code"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(
        data,
        "close_near_high_on_high_amount_20d",
        "close_near_high_on_high_amount_20d",
    )


def down_day_absorption_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day weighted support on down days as an absorption proxy.

    Larger values indicate that down days still show good recovery from intraday
    lows while trading amount remains meaningful.
    """

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close", "pre_close", "amount"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close", "pre_close", "amount"])
    price_range = data["high"] - data["low"]
    recovery = np.where(price_range == 0, np.nan, (data["close"] - data["low"]) / price_range)
    down_day_weighted = np.where(
        data["close"] < data["pre_close"],
        recovery * np.log1p(data["amount"].clip(lower=0.0)),
        np.nan,
    )
    data["down_day_absorption_20d"] = (
        pd.Series(down_day_weighted, index=data.index)
        .groupby(data["ts_code"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "down_day_absorption_20d", "down_day_absorption_20d")


def amplitude_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """20-day mean daily amplitude, where daily amplitude is (high - low) / pre_close on qfq prices."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "pre_close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "pre_close"])
    daily_amplitude = np.where(data["pre_close"] == 0, np.nan, (data["high"] - data["low"]) / data["pre_close"])
    data["amplitude_20d"] = (
        pd.Series(daily_amplitude, index=data.index)
        .groupby(data["ts_code"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "amplitude_20d", "amplitude_20d")


def _window_max_drawdown(window: pd.Series) -> float:
    if window.empty:
        return np.nan
    running_peak = window.cummax()
    drawdown = 1.0 - (window / running_peak)
    return float(drawdown.max())


def close_to_high_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Distance to 20-day high defined as qfq close divided by rolling 20-day qfq high."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("close", "high"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "close", "high"])
    rolling_high = data.groupby("ts_code")["high"].rolling(20).max().reset_index(level=0, drop=True)
    data["close_to_high_20d"] = np.where(rolling_high == 0, np.nan, data["close"] / rolling_high)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "close_to_high_20d", "close_to_high_20d")
