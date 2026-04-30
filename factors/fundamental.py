"""Fundamental and valuation factor implementations."""

from __future__ import annotations

import pandas as pd

from data.fundamental_daily import get_a_share_fundamental_daily
from data.valuation_daily import get_a_share_daily_valuation
from factors.base import build_factor_output, validate_factor_input


def _buffered_start_date(start_date: str, lookback_days: int) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")


def _clip_dates(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
    return data.loc[mask].reset_index(drop=True)


def turnover_mean_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, 45),
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "turnover_rate_f"])
    data["turnover_mean_20d"] = data.groupby("ts_code")["turnover_rate_f"].rolling(20).mean().reset_index(
        level=0, drop=True
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "turnover_mean_20d", "turnover_mean_20d")


def pe_ttm_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "pe_ttm"])
    return build_factor_output(data, "pe_ttm", "pe_ttm")


def ep_ttm_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Earnings yield defined as 1 / pe_ttm, with invalid pe_ttm mapped to NaN."""

    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "pe_ttm"])
    valid_pe = pd.Series(data["pe_ttm"]).where(data["pe_ttm"] > 0)
    data["ep_ttm"] = 1.0 / valid_pe
    return build_factor_output(data, "ep_ttm", "ep_ttm")


def pb_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "pb"])
    return build_factor_output(data, "pb", "pb")


def bp_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Book-to-price proxy defined as 1 / pb, with invalid pb mapped to NaN."""

    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "pb"])
    valid_pb = pd.Series(data["pb"]).where(data["pb"] > 0)
    data["bp"] = 1.0 / valid_pb
    return build_factor_output(data, "bp", "bp")


def roe_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    data = get_a_share_fundamental_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "roe"])
    return build_factor_output(data, "roe", "roe")


def roe_ttm_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """First-pass alias for the current aligned ROE field."""

    data = get_a_share_fundamental_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "roe"])
    data["roe_ttm"] = data["roe"]
    return build_factor_output(data, "roe_ttm", "roe_ttm")


def revenue_growth_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    data = get_a_share_fundamental_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "revenue_growth"])
    return build_factor_output(data, "revenue_growth", "revenue_growth")
