"""Fundamental and valuation factor implementations."""

from __future__ import annotations

import pandas as pd

from data.fundamental_daily import get_a_share_fundamental_daily
from data.holder_number import get_a_share_holder_number_daily
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


def holder_num_change_ratio_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative shareholder-number change ratio aligned by ann_date; larger means larger holder count decline."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "holder_num_change_ratio_negative"])
    return build_factor_output(
        data,
        "holder_num_change_ratio_negative",
        "holder_num_change_ratio_negative",
    )


def holder_num_change_ratio_negative_fresh_3d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Holder-count decline factor, valid only when ann_date is less than 3 trading days after end_date."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(
        data,
        ["trade_date", "ts_code", "holder_num_change_ratio_negative", "holder_report_lag_trading_days"],
    )
    lag = pd.to_numeric(data["holder_report_lag_trading_days"], errors="coerce")
    data["holder_num_change_ratio_negative_fresh_3d"] = data["holder_num_change_ratio_negative"].where(lag < 3)
    return build_factor_output(
        data,
        "holder_num_change_ratio_negative_fresh_3d",
        "holder_num_change_ratio_negative_fresh_3d",
    )


def _build_fresh_holder_decline_decay_factor(data: pd.DataFrame, *, valid_days: int, factor_name: str) -> pd.DataFrame:
    validate_factor_input(
        data,
        [
            "trade_date",
            "ts_code",
            "holder_num_change_ratio_negative",
            "holder_report_lag_trading_days",
            "holder_announcement_age_trading_days",
        ],
    )
    lag = pd.to_numeric(data["holder_report_lag_trading_days"], errors="coerce")
    age = pd.to_numeric(data["holder_announcement_age_trading_days"], errors="coerce")
    decay_weight = ((float(valid_days) - age + 1.0) / float(valid_days + 1)).clip(lower=0.0, upper=1.0)
    data[factor_name] = data["holder_num_change_ratio_negative"].where((lag <= 3) & (age <= valid_days)) * decay_weight
    return build_factor_output(data, factor_name, factor_name)


def holder_num_change_ratio_negative_fresh_3d_decay_3d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fresh holder-count decline factor with a 3-trading-day post-announcement linear decay."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return _build_fresh_holder_decline_decay_factor(
        data,
        valid_days=3,
        factor_name="holder_num_change_ratio_negative_fresh_3d_decay_3d",
    )


def holder_num_change_ratio_negative_fresh_3d_decay_5d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fresh holder-count decline factor with a 5-trading-day post-announcement linear decay."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return _build_fresh_holder_decline_decay_factor(
        data,
        valid_days=5,
        factor_name="holder_num_change_ratio_negative_fresh_3d_decay_5d",
    )


def holder_num_change_ratio_negative_fresh_3d_decay_10d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fresh holder-count decline factor with a 10-trading-day post-announcement linear decay."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return _build_fresh_holder_decline_decay_factor(
        data,
        valid_days=10,
        factor_name="holder_num_change_ratio_negative_fresh_3d_decay_10d",
    )


def holder_num_change_ratio_negative_fresh_3d_announced_today_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fresh holder-count decline factor that is only valid on its ann_date."""

    data = get_a_share_holder_number_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(
        data,
        ["trade_date", "ts_code", "ann_date", "holder_num_change_ratio_negative", "holder_report_lag_trading_days"],
    )
    lag = pd.to_numeric(data["holder_report_lag_trading_days"], errors="coerce")
    announced_today = data["trade_date"].astype(str) == data["ann_date"].astype(str)
    data["holder_num_change_ratio_negative_fresh_3d_announced_today"] = data[
        "holder_num_change_ratio_negative"
    ].where((lag < 3) & announced_today)
    return build_factor_output(
        data,
        "holder_num_change_ratio_negative_fresh_3d_announced_today",
        "holder_num_change_ratio_negative_fresh_3d_announced_today",
    )
