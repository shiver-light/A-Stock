"""Technical factor implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.daily_market import AShareDailyMarketService, DailyMarketRequest
from factors.base import build_factor_output, validate_factor_input


def _buffered_start_date(start_date: str, lookback_days: int) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")


def _clip_dates(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
    return data.loc[mask].reset_index(drop=True)


def _load_qfq_daily(
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    service = AShareDailyMarketService()
    request = DailyMarketRequest(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, lookback_days),
        end_date=end_date,
        adjust="qfq",
        fields=("trade_date", "ts_code", "close"),
        refresh=refresh,
    )
    return service.get_daily(request)


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


def volatility_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=45)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    returns = data.groupby("ts_code")["close"].pct_change()
    data["volatility_20d"] = returns.groupby(data["ts_code"]).rolling(20).std().reset_index(level=0, drop=True)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_20d", "volatility_20d")


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
