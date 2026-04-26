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
