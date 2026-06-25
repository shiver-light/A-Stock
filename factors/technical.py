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


def turnover_stability_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 20-day turnover volatility so larger values mean more stable turnover."""

    data = get_a_share_daily_valuation(
        ts_code=ts_code,
        start_date=_buffered_start_date(start_date, 45),
        end_date=end_date,
        refresh=refresh,
    )
    validate_factor_input(data, ["trade_date", "ts_code", "turnover_rate_f"])
    data["turnover_stability_20d"] = -data.groupby("ts_code")["turnover_rate_f"].rolling(20).std().reset_index(
        level=0,
        drop=True,
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "turnover_stability_20d", "turnover_stability_20d")


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


def price_new_low_20d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Closeness to a 20-day closing low, where larger values mean closer to a stage low."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=45)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    rolling_low = data.groupby("ts_code")["close"].rolling(20).min().reset_index(level=0, drop=True)
    data["price_new_low_20d"] = np.where(data["close"] == 0, np.nan, rolling_low / data["close"])
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "price_new_low_20d", "price_new_low_20d")


def kdj_bullish_divergence_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """KDJ bullish divergence near a 20-day low.

    Larger values indicate price is at/near a 20-day low while J is above its
    own recent low. This uses only same-day and historical OHLC data.
    """

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=70,
        fields=("high", "low", "close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close"])
    data = _append_kdj_columns(data)
    grouped = data.groupby("ts_code")
    rolling_close_low = grouped["close"].rolling(20).min().reset_index(level=0, drop=True)
    rolling_j_low = grouped["kdj_j"].rolling(20).min().reset_index(level=0, drop=True)
    price_near_low = np.where(data["close"] == 0, np.nan, rolling_close_low / data["close"])
    j_divergence = (data["kdj_j"] - rolling_j_low) / 100.0
    data["kdj_bullish_divergence_20d"] = np.where(price_near_low >= 0.98, price_near_low + j_divergence, np.nan)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "kdj_bullish_divergence_20d", "kdj_bullish_divergence_20d")


def kdj_j_turn_up_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """J value turning up from a low zone, where larger values mean stronger low-level rebound."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close"])
    data = _append_kdj_columns(data)
    previous_j = data.groupby("ts_code")["kdj_j"].shift(1)
    j_delta = data["kdj_j"] - previous_j
    data["kdj_j_turn_up"] = np.where((previous_j <= 30.0) & (j_delta > 0), j_delta / 100.0, np.nan)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "kdj_j_turn_up", "kdj_j_turn_up")


def kdj_j_turn_up_3d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Trailing 3-day low-zone J turn-up signal with recency decay and no look-ahead."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close"])
    data = _append_kdj_columns(data)
    previous_j = data.groupby("ts_code")["kdj_j"].shift(1)
    j_delta = data["kdj_j"] - previous_j
    raw_signal = pd.Series(
        np.where((previous_j <= 30.0) & (j_delta > 0), j_delta / 100.0, np.nan),
        index=data.index,
        dtype="float64",
    )
    data["kdj_j_turn_up_3d"] = _trailing_decay_max(raw_signal, data["ts_code"])
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "kdj_j_turn_up_3d", "kdj_j_turn_up_3d")


def kdj_golden_cross_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """KDJ K crossing above D on the current day, represented as 1.0 when true."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close"])
    data = _append_kdj_columns(data)
    previous_k = data.groupby("ts_code")["kdj_k"].shift(1)
    previous_d = data.groupby("ts_code")["kdj_d"].shift(1)
    cross = (previous_k <= previous_d) & (data["kdj_k"] > data["kdj_d"])
    data["kdj_golden_cross"] = np.where(cross, 1.0, np.nan)
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "kdj_golden_cross", "kdj_golden_cross")


def kdj_golden_cross_3d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Trailing 3-day KDJ golden-cross event with recency decay and no look-ahead."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close"])
    data = _append_kdj_columns(data)
    previous_k = data.groupby("ts_code")["kdj_k"].shift(1)
    previous_d = data.groupby("ts_code")["kdj_d"].shift(1)
    cross = (previous_k <= previous_d) & (data["kdj_k"] > data["kdj_d"])
    raw_signal = pd.Series(np.where(cross, 1.0, np.nan), index=data.index, dtype="float64")
    data["kdj_golden_cross_3d"] = _trailing_decay_max(raw_signal, data["ts_code"])
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "kdj_golden_cross_3d", "kdj_golden_cross_3d")


def amount_mild_expansion_5d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Mild amount expansion: highest near 5-day amount at 1.1-1.8x the 20-day mean."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("amount",),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "amount"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    amount = pd.to_numeric(data["amount"], errors="coerce")
    grouped = data["ts_code"]
    amount_5d = amount.groupby(grouped).rolling(5).mean().reset_index(level=0, drop=True)
    amount_20d = amount.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    ratio = np.where(amount_20d == 0, np.nan, amount_5d / amount_20d)
    data["amount_mild_expansion_5d"] = np.where(
        (ratio >= 1.0) & (ratio <= 2.2),
        1.0 - (np.abs(ratio - 1.4) / 1.4),
        np.nan,
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "amount_mild_expansion_5d", "amount_mild_expansion_5d")


def ma5_ma10_breakout_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Close above both 5-day and 10-day moving averages, scaled by the smaller distance."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=30)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped_close = data.groupby("ts_code")["close"]
    ma5 = grouped_close.rolling(5).mean().reset_index(level=0, drop=True)
    ma10 = grouped_close.rolling(10).mean().reset_index(level=0, drop=True)
    distance5 = np.where(ma5 == 0, np.nan, (data["close"] / ma5) - 1.0)
    distance10 = np.where(ma10 == 0, np.nan, (data["close"] / ma10) - 1.0)
    data["ma5_ma10_breakout"] = np.where(
        (distance5 > 0) & (distance10 > 0),
        np.minimum(distance5, distance10),
        np.nan,
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "ma5_ma10_breakout", "ma5_ma10_breakout")


def ma5_ma10_breakout_3d_factor(*, ts_code: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """Trailing 3-day close-above-MA5/MA10 signal with recency decay and no look-ahead."""

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=30)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped_close = data.groupby("ts_code")["close"]
    ma5 = grouped_close.rolling(5).mean().reset_index(level=0, drop=True)
    ma10 = grouped_close.rolling(10).mean().reset_index(level=0, drop=True)
    distance5 = np.where(ma5 == 0, np.nan, (data["close"] / ma5) - 1.0)
    distance10 = np.where(ma10 == 0, np.nan, (data["close"] / ma10) - 1.0)
    raw_signal = pd.Series(
        np.where((distance5 > 0) & (distance10 > 0), np.minimum(distance5, distance10), np.nan),
        index=data.index,
        dtype="float64",
    )
    data["ma5_ma10_breakout_3d"] = _trailing_decay_max(raw_signal, data["ts_code"])
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "ma5_ma10_breakout_3d", "ma5_ma10_breakout_3d")


def gap_risk_20d_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 20-day mean absolute open gap so larger values mean lower gap risk."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("open", "pre_close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "open", "pre_close"])
    abs_gap = ((data["open"] / data["pre_close"]) - 1.0).abs().where(data["pre_close"] != 0)
    data["gap_risk_20d_negative"] = -(
        pd.Series(abs_gap, index=data.index).groupby(data["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "gap_risk_20d_negative", "gap_risk_20d_negative")


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


def low_range_high_amount_days_ratio_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day ratio of quiet-range but high-amount sessions as an accumulation proxy."""

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("high", "low", "pre_close", "amount"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "pre_close", "amount"])
    amplitude = ((data["high"] - data["low"]) / data["pre_close"]).where(data["pre_close"] != 0)
    amount = pd.to_numeric(data["amount"], errors="coerce")
    grouped = data["ts_code"]
    rolling_amplitude_mean = (
        pd.Series(amplitude, index=data.index).groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    )
    rolling_amount_mean = amount.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    quiet_with_amount = ((amplitude <= rolling_amplitude_mean) & (amount >= rolling_amount_mean)).astype("float64")
    data["low_range_high_amount_days_ratio_20d"] = (
        quiet_with_amount.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(
        data,
        "low_range_high_amount_days_ratio_20d",
        "low_range_high_amount_days_ratio_20d",
    )


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
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    price_range = pd.to_numeric(data["high"] - data["low"], errors="coerce")
    recovery = ((data["close"] - data["low"]) / price_range).where(price_range != 0)
    amount_weight = pd.to_numeric(data["amount"], errors="coerce").clip(lower=0.0).map(np.log1p)
    down_day_weighted = (recovery * amount_weight).where(data["close"] < data["pre_close"])
    down_day_weighted = pd.Series(down_day_weighted, index=data.index, dtype="float64")
    data["down_day_absorption_20d"] = (
        down_day_weighted.groupby(data["ts_code"])
        .rolling(20, min_periods=3)
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


def _append_kdj_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()
    grouped = result.groupby("ts_code", group_keys=False)
    low_9 = grouped["low"].rolling(9).min().reset_index(level=0, drop=True)
    high_9 = grouped["high"].rolling(9).max().reset_index(level=0, drop=True)
    denominator = high_9 - low_9
    result["kdj_rsv"] = np.where(denominator == 0, 50.0, ((result["close"] - low_9) / denominator) * 100.0)
    result["kdj_k"] = grouped["kdj_rsv"].transform(lambda series: series.ewm(alpha=1.0 / 3.0, adjust=False).mean())
    result["kdj_d"] = grouped["kdj_k"].transform(lambda series: series.ewm(alpha=1.0 / 3.0, adjust=False).mean())
    result["kdj_j"] = (3.0 * result["kdj_k"]) - (2.0 * result["kdj_d"])
    return result


def _trailing_decay_max(
    values: pd.Series,
    group_keys: pd.Series,
    *,
    decay_weights: tuple[float, ...] = (1.0, 0.7, 0.4),
) -> pd.Series:
    weighted_signals = []
    grouped = values.groupby(group_keys)
    for offset, weight in enumerate(decay_weights):
        weighted_signals.append(grouped.shift(offset) * weight)
    combined = pd.concat(weighted_signals, axis=1)
    return combined.max(axis=1, skipna=True)


def pullback_after_trend_60d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """60-day trend minus recent 5-day return, only when price stays above its 60-day mean.

    Larger values favor stocks with a positive medium-term trend and a recent
    pullback, avoiding pure short-term reversal candidates without trend support.
    """

    data = _load_qfq_daily(ts_code, start_date, end_date, refresh, lookback_days=90)
    validate_factor_input(data, ["trade_date", "ts_code", "close"])
    grouped_close = data.groupby("ts_code")["close"]
    return_60d = grouped_close.pct_change(60)
    return_5d = grouped_close.pct_change(5)
    ma_60d = grouped_close.rolling(60).mean().reset_index(level=0, drop=True)
    data["pullback_after_trend_60d"] = np.where(
        (return_60d > 0) & (data["close"] >= ma_60d),
        return_60d - return_5d,
        np.nan,
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "pullback_after_trend_60d", "pullback_after_trend_60d")


def distribution_risk_20d_negative_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 20-day distribution-risk proxy, so larger values mean less suspected distribution.

    The raw risk proxy combines high-turnover down days, amount-weighted upper
    shadows, and high turnover without price progress. It uses only same-day
    and historical OHLCV/turnover information.
    """

    market_data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=45,
        fields=("open", "high", "low", "close", "pre_close", "amount"),
    )
    turnover_data = _load_turnover_data(ts_code, start_date, end_date, refresh, lookback_days=45)
    data = _merge_market_and_turnover(market_data, turnover_data)
    validate_factor_input(
        data,
        ["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "amount", "turnover_rate_f"],
    )
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    price_range = data["high"] - data["low"]
    upper_shadow = (data["high"] - data[["open", "close"]].max(axis=1)) / price_range.replace(0, np.nan)
    daily_return = ((data["close"] / data["pre_close"]) - 1.0).where(data["pre_close"] != 0)
    amount_weight = pd.to_numeric(data["amount"], errors="coerce").clip(lower=0.0).map(np.log1p)
    turnover = pd.to_numeric(data["turnover_rate_f"], errors="coerce").clip(lower=0.0)

    down_turnover = turnover.where(data["close"] < data["pre_close"], 0.0)
    upper_shadow_amount = upper_shadow.clip(lower=0.0) * amount_weight
    turnover_without_progress = turnover / (daily_return.abs() + 0.01)
    raw_risk = down_turnover + upper_shadow_amount + turnover_without_progress
    data["distribution_risk_20d_negative"] = -(
        pd.Series(raw_risk, index=data.index)
        .groupby(data["ts_code"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "distribution_risk_20d_negative", "distribution_risk_20d_negative")


def liquidity_improvement_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """20-day liquidity improvement versus 60-day baseline with unstable turnover penalized."""

    market_data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=90,
        fields=("amount",),
    )
    turnover_data = _load_turnover_data(ts_code, start_date, end_date, refresh, lookback_days=90)
    data = _merge_market_and_turnover(market_data, turnover_data)
    validate_factor_input(data, ["trade_date", "ts_code", "amount", "turnover_rate_f"])
    grouped = data["ts_code"]

    amount = pd.to_numeric(data["amount"], errors="coerce")
    turnover = pd.to_numeric(data["turnover_rate_f"], errors="coerce")
    amount_mean_20 = amount.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    amount_mean_60 = amount.groupby(grouped).rolling(60).mean().reset_index(level=0, drop=True)
    turnover_mean_20 = turnover.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    turnover_mean_60 = turnover.groupby(grouped).rolling(60).mean().reset_index(level=0, drop=True)
    turnover_std_20 = turnover.groupby(grouped).rolling(20).std().reset_index(level=0, drop=True)

    amount_ratio = np.where(amount_mean_60 == 0, np.nan, amount_mean_20 / amount_mean_60)
    turnover_ratio = np.where(turnover_mean_60 == 0, np.nan, turnover_mean_20 / turnover_mean_60)
    turnover_instability = turnover_std_20 / (turnover_mean_20.abs() + 1e-6)
    data["liquidity_improvement_20d"] = (0.5 * amount_ratio) + (0.5 * turnover_ratio) - turnover_instability
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "liquidity_improvement_20d", "liquidity_improvement_20d")


def volatility_contraction_20d_factor(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Negative 20-day volatility/amplitude versus 60-day baseline.

    Larger values indicate stronger recent volatility contraction and are meant
    to be used with trend or buying-support factors, not as a standalone alpha.
    """

    data = _load_qfq_market_data(
        ts_code,
        start_date,
        end_date,
        refresh,
        lookback_days=90,
        fields=("high", "low", "close", "pre_close"),
    )
    validate_factor_input(data, ["trade_date", "ts_code", "high", "low", "close", "pre_close"])
    grouped = data["ts_code"]
    daily_amplitude = pd.Series(
        np.where(data["pre_close"] == 0, np.nan, (data["high"] - data["low"]) / data["pre_close"]),
        index=data.index,
    )
    returns = data.groupby("ts_code")["close"].pct_change()
    amplitude_20 = daily_amplitude.groupby(grouped).rolling(20).mean().reset_index(level=0, drop=True)
    amplitude_60 = daily_amplitude.groupby(grouped).rolling(60).mean().reset_index(level=0, drop=True)
    volatility_20 = returns.groupby(grouped).rolling(20).std().reset_index(level=0, drop=True)
    volatility_60 = returns.groupby(grouped).rolling(60).std().reset_index(level=0, drop=True)

    amplitude_ratio = np.where(amplitude_60 == 0, np.nan, amplitude_20 / amplitude_60)
    volatility_ratio = np.where(volatility_60 == 0, np.nan, volatility_20 / volatility_60)
    data["volatility_contraction_20d"] = -((0.5 * amplitude_ratio) + (0.5 * volatility_ratio))
    data = _clip_dates(data, start_date, end_date)
    return build_factor_output(data, "volatility_contraction_20d", "volatility_contraction_20d")
