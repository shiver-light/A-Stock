"""Low-position accumulation pattern research utilities.

This module separates three time domains:
- features: only use trade date T and historical rows before T
- labels: use future rows after T for research evaluation only
- scoring: weights are fitted on a time-bounded training set, then applied out of sample
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = ["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount"]
TURNOVER_COLUMNS = ["trade_date", "ts_code", "turnover_rate_f"]
LABEL_COLUMNS = ["future_20d_max_gain", "future_20d_max_drawdown", "label"]


def build_accumulation_dataset(
    market_data: pd.DataFrame,
    turnover_data: pd.DataFrame,
    *,
    industry_map: pd.DataFrame | None = None,
    benchmark_returns: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build features and future labels for low-position accumulation research."""

    market = _prepare_market_data(market_data)
    turnover = _prepare_turnover_data(turnover_data)
    data = market.merge(turnover, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    data["daily_return"] = np.where(data["pre_close"] == 0, np.nan, data["close"] / data["pre_close"] - 1.0)
    data["amplitude"] = np.where(data["pre_close"] == 0, np.nan, (data["high"] - data["low"]) / data["pre_close"])
    intraday_range = data["high"] - data["low"]
    data["close_position"] = np.where(intraday_range == 0, np.nan, (data["close"] - data["low"]) / intraday_range)
    data["upper_shadow"] = np.where(
        intraday_range == 0,
        np.nan,
        (data["high"] - data[["open", "close"]].max(axis=1)) / intraday_range,
    )
    data["lower_shadow"] = np.where(
        intraday_range == 0,
        np.nan,
        (data[["open", "close"]].min(axis=1) - data["low"]) / intraday_range,
    )

    data = _append_price_features(data)
    data = _append_ma_features(data)
    data = _append_volume_features(data)
    data = _append_turnover_features(data)
    data = _append_efficiency_features(data)
    data = _append_absorption_features(data)
    data = _append_relative_strength_features(data, benchmark_returns or {})
    data = _append_industry_features(data, industry_map)
    data = _append_labels(data)
    data["low_position_filter"] = (data["price_position_120d"] < 0.30) | (data["drawdown_from_120d_high"] <= -0.20)
    return data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def filter_research_samples(data: pd.DataFrame, *, low_position_only: bool = True) -> pd.DataFrame:
    """Return non-ambiguous labeled samples, optionally restricted to low-position rows."""

    if data.empty:
        return data.copy()
    result = data.loc[data["label"].isin([0, 1])].copy()
    if low_position_only and "low_position_filter" in result.columns:
        result = result.loc[result["low_position_filter"]].copy()
    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def split_by_time(
    data: pd.DataFrame,
    *,
    train_end: str = "20211231",
    validation_end: str = "20231231",
) -> dict[str, pd.DataFrame]:
    """Split samples by date into train, validation and test sets."""

    if data.empty:
        return {"train": data.copy(), "validation": data.copy(), "test": data.copy()}
    result = data.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    return {
        "train": result.loc[result["trade_date"] <= train_end].copy(),
        "validation": result.loc[(result["trade_date"] > train_end) & (result["trade_date"] <= validation_end)].copy(),
        "test": result.loc[result["trade_date"] > validation_end].copy(),
    }


def single_factor_analysis(
    samples: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Analyze each feature by quantile, label hit rate and forward returns."""

    if samples.empty or not feature_cols:
        return pd.DataFrame(
            columns=[
                "feature",
                "quantile",
                "sample_count",
                "positive_ratio",
                "mean_forward_5d",
                "median_forward_5d",
                "mean_forward_10d",
                "median_forward_10d",
                "mean_forward_20d",
                "median_forward_20d",
                "mean_future_20d_max_drawdown",
            ]
        )
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        if feature not in samples.columns:
            continue
        data = samples.loc[:, ["trade_date", "ts_code", "label", feature, *_forward_columns(samples)]].dropna(
            subset=[feature, "label"]
        )
        if data.empty:
            continue
        data["quantile"] = _cross_section_quantile(data, feature, n_quantiles)
        data = data.dropna(subset=["quantile"])
        for quantile, group in data.groupby("quantile", sort=True):
            row = {
                "feature": feature,
                "quantile": int(quantile),
                "sample_count": int(len(group)),
                "positive_ratio": float(group["label"].mean()),
                "mean_future_20d_max_drawdown": _safe_mean(group.get("future_20d_max_drawdown")),
            }
            for horizon in [5, 10, 20]:
                column = f"forward_return_{horizon}d"
                row[f"mean_forward_{horizon}d"] = _safe_mean(group.get(column))
                row[f"median_forward_{horizon}d"] = _safe_median(group.get(column))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature", "quantile"]).reset_index(drop=True)


def summarize_feature_power(quantile_stats: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature discrimination from quantile-level statistics."""

    if quantile_stats.empty:
        return pd.DataFrame(columns=["feature", "sample_count", "positive_ratio_spread", "return_20d_spread", "score"])
    rows = []
    for feature, group in quantile_stats.groupby("feature", sort=True):
        ordered = group.sort_values("quantile")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        positive_spread = float(high["positive_ratio"] - low["positive_ratio"])
        return_spread = float(high["mean_forward_20d"] - low["mean_forward_20d"])
        score = max(positive_spread, 0.0) + max(return_spread, 0.0)
        rows.append(
            {
                "feature": feature,
                "sample_count": int(group["sample_count"].sum()),
                "positive_ratio_spread": positive_spread,
                "return_20d_spread": return_spread,
                "score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_condition_ladder(samples: pd.DataFrame, conditions: list[dict[str, object]]) -> pd.DataFrame:
    """Evaluate incremental predictive value as conditions are added one by one."""

    rows = []
    current = samples.copy()
    rows.append(_condition_row("baseline", current))
    for condition in conditions:
        before = current.copy()
        current = _apply_condition(current, condition)
        row = _condition_row(str(condition.get("name") or condition.get("feature")), current)
        row["previous_sample_count"] = int(len(before))
        row["sample_delta"] = int(len(current) - len(before))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_accumulation_score_model(
    train_samples: pd.DataFrame,
    feature_power: pd.DataFrame,
    *,
    max_features: int = 10,
) -> dict[str, object]:
    """Fit a simple 0-100 accumulation score model from training feature power."""

    if train_samples.empty or feature_power.empty:
        return {"features": [], "weights": {}, "feature_directions": {}, "reason": "empty training data"}
    selected = feature_power.loc[feature_power["score"] > 0].head(max_features).copy()
    if selected.empty:
        return {"features": [], "weights": {}, "feature_directions": {}, "reason": "no positive feature power"}
    total = float(selected["score"].sum())
    weights = {row.feature: float(row.score / total) for row in selected.itertuples(index=False)}
    directions = {
        row.feature: 1 if float(row.positive_ratio_spread) >= 0 else -1 for row in selected.itertuples(index=False)
    }
    return {
        "features": list(weights.keys()),
        "weights": weights,
        "feature_directions": directions,
        "reason": "weights are proportional to training-set positive-rate and 20d-return quantile spreads",
    }


def apply_accumulation_score(data: pd.DataFrame, model: dict[str, object]) -> pd.DataFrame:
    """Apply a fitted 0-100 accumulation score model to samples."""

    if data.empty:
        result = data.copy()
        result["accumulation_score"] = pd.Series(dtype="float64")
        return result
    features = list(model.get("features", []))
    weights = dict(model.get("weights", {}))
    directions = dict(model.get("feature_directions", {}))
    result = data.copy()
    if not features:
        result["accumulation_score"] = np.nan
        return result
    score = pd.Series(0.0, index=result.index)
    for feature in features:
        if feature not in result.columns:
            continue
        ranks = result.groupby("trade_date")[feature].rank(pct=True, method="average")
        direction = float(directions.get(feature, 1))
        component = ranks if direction >= 0 else 1.0 - ranks
        score = score + float(weights.get(feature, 0.0)) * component.fillna(0.0)
    result["accumulation_score"] = (score * 100.0).clip(0.0, 100.0)
    return result


def score_bucket_analysis(scored: pd.DataFrame) -> pd.DataFrame:
    """Analyze forward outcomes by accumulation score bucket."""

    if scored.empty or "accumulation_score" not in scored.columns:
        return pd.DataFrame(columns=["score_bucket", "sample_count", "positive_ratio", "mean_forward_20d"])
    data = scored.dropna(subset=["accumulation_score", "label"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["score_bucket", "sample_count", "positive_ratio", "mean_forward_20d"])
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    data["score_bucket"] = pd.cut(data["accumulation_score"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for bucket, group in data.groupby("score_bucket", observed=True):
        row = {
            "score_bucket": str(bucket),
            "sample_count": int(len(group)),
            "positive_ratio": float(group["label"].mean()),
            "mean_future_20d_max_drawdown": _safe_mean(group.get("future_20d_max_drawdown")),
        }
        for horizon in [5, 10, 20]:
            row[f"mean_forward_{horizon}d"] = _safe_mean(group.get(f"forward_return_{horizon}d"))
            row[f"median_forward_{horizon}d"] = _safe_median(group.get(f"forward_return_{horizon}d"))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


DEFAULT_ACCUMULATION_FEATURES = [
    "price_position_60d",
    "price_position_120d",
    "price_position_250d",
    "drawdown_from_120d_high",
    "amplitude_mean_10d",
    "amplitude_mean_20d",
    "amplitude_mean_30d",
    "new_low_10d",
    "new_low_20d",
    "second_bottom_20d",
    "second_bottom_volume_shrink_20d",
    "ma5_distance",
    "ma10_distance",
    "ma20_distance",
    "ma60_distance",
    "ma20_slope",
    "ma20_slope_change",
    "ma_convergence_5_10_20",
    "volume_ratio_3_10",
    "volume_ratio_5_20",
    "volume_trend_20d",
    "down_day_volume_mean_20d",
    "up_day_volume_mean_20d",
    "up_down_volume_ratio_20d",
    "shrink_down_count_20d",
    "expand_up_count_20d",
    "expand_no_down_count_20d",
    "turnover_mean_3d",
    "turnover_mean_5d",
    "turnover_mean_10d",
    "turnover_mean_20d",
    "turnover_ratio_to_20d",
    "up_day_turnover_mean_20d",
    "down_day_turnover_mean_20d",
    "turnover_trend_20d",
    "up_efficiency_3d",
    "down_efficiency_3d",
    "efficiency_ratio_3d",
    "up_efficiency_5d",
    "down_efficiency_5d",
    "efficiency_ratio_5d",
    "up_efficiency_10d",
    "down_efficiency_10d",
    "efficiency_ratio_10d",
    "up_efficiency_20d",
    "down_efficiency_20d",
    "efficiency_ratio_20d",
    "up_efficiency_trend_3_20",
    "down_efficiency_trend_3_20",
    "efficiency_ratio_trend_3_20",
    "volume_no_down_20d",
    "high_turnover_no_down_20d",
    "long_lower_shadow_20d",
    "recover_previous_low_20d",
    "second_bottom_not_break_20d",
    "benchmark_down_relative_return",
    "relative_hs300_20d",
    "relative_zz1000_20d",
    "relative_industry_20d",
]


def _prepare_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(market_data, BASE_COLUMNS)
    data = market_data.loc[:, BASE_COLUMNS].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["ts_code"] = data["ts_code"].astype(str)
    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _prepare_turnover_data(turnover_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(turnover_data, TURNOVER_COLUMNS)
    data = turnover_data.loc[:, TURNOVER_COLUMNS].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["ts_code"] = data["ts_code"].astype(str)
    data["turnover_rate_f"] = pd.to_numeric(data["turnover_rate_f"], errors="coerce")
    return (
        data.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )


def _append_price_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped_close = result.groupby("ts_code")["close"]
    for window in [60, 120, 250]:
        rolling_low = grouped_close.rolling(window).min().reset_index(level=0, drop=True)
        rolling_high = grouped_close.rolling(window).max().reset_index(level=0, drop=True)
        result[f"price_position_{window}d"] = np.where(
            rolling_high == rolling_low,
            np.nan,
            (result["close"] - rolling_low) / (rolling_high - rolling_low),
        )
    high_120 = grouped_close.rolling(120).max().reset_index(level=0, drop=True)
    result["drawdown_from_120d_high"] = np.where(high_120 == 0, np.nan, result["close"] / high_120 - 1.0)
    for window in [10, 20, 30]:
        result[f"amplitude_mean_{window}d"] = (
            result.groupby("ts_code")["amplitude"].rolling(window).mean().reset_index(level=0, drop=True)
        )
    for window in [10, 20]:
        rolling_low = grouped_close.rolling(window).min().reset_index(level=0, drop=True)
        result[f"new_low_{window}d"] = (result["close"] <= rolling_low).astype("float64")
    low_20 = grouped_close.rolling(20).min().reset_index(level=0, drop=True)
    prior_low_20 = low_20.groupby(result["ts_code"]).shift(5)
    result["second_bottom_20d"] = (
        (result["close"] <= prior_low_20 * 1.05) & (result["close"] >= prior_low_20 * 0.97)
    ).astype("float64")
    volume_mean_5 = result.groupby("ts_code")["vol"].rolling(5).mean().reset_index(level=0, drop=True)
    prior_volume_mean_5 = volume_mean_5.groupby(result["ts_code"]).shift(5)
    result["second_bottom_volume_shrink_20d"] = (
        (result["second_bottom_20d"] > 0) & (volume_mean_5 < prior_volume_mean_5)
    ).astype("float64")
    return result


def _append_ma_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped_close = result.groupby("ts_code")["close"]
    ma_values = {}
    for window in [5, 10, 20, 60]:
        ma = grouped_close.rolling(window).mean().reset_index(level=0, drop=True)
        ma_values[window] = ma
        result[f"ma{window}_distance"] = np.where(ma == 0, np.nan, result["close"] / ma - 1.0)
    ma20 = ma_values[20]
    prior_ma20 = ma20.groupby(result["ts_code"]).shift(1)
    result["ma20_slope"] = np.where(prior_ma20 == 0, np.nan, ma20 / prior_ma20 - 1.0)
    result["ma20_slope_change"] = result["ma20_slope"] - result.groupby("ts_code")["ma20_slope"].shift(5)
    ma_stack = pd.concat([ma_values[5], ma_values[10], ma_values[20]], axis=1)
    result["ma_convergence_5_10_20"] = ma_stack.std(axis=1) / ma_stack.mean(axis=1)
    return result


def _append_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped_vol = result.groupby("ts_code")["vol"]
    vol3 = grouped_vol.rolling(3).mean().reset_index(level=0, drop=True)
    vol5 = grouped_vol.rolling(5).mean().reset_index(level=0, drop=True)
    vol10 = grouped_vol.rolling(10).mean().reset_index(level=0, drop=True)
    vol20 = grouped_vol.rolling(20).mean().reset_index(level=0, drop=True)
    result["volume_ratio_3_10"] = np.where(vol10 == 0, np.nan, vol3 / vol10)
    result["volume_ratio_5_20"] = np.where(vol20 == 0, np.nan, vol5 / vol20)
    result["volume_trend_20d"] = grouped_vol.rolling(20).apply(_linear_slope, raw=True).reset_index(level=0, drop=True)
    up_volume = result["vol"].where(result["daily_return"] > 0)
    down_volume = result["vol"].where(result["daily_return"] < 0)
    result["up_day_volume_mean_20d"] = up_volume.groupby(result["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    result["down_day_volume_mean_20d"] = (
        down_volume.groupby(result["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    result["up_down_volume_ratio_20d"] = result["up_day_volume_mean_20d"] / result["down_day_volume_mean_20d"]
    shrink_down = ((result["daily_return"] < 0) & (result["vol"] < vol20)).astype("float64")
    expand_up = ((result["daily_return"] > 0) & (result["vol"] > vol20)).astype("float64")
    expand_no_down = ((result["daily_return"] >= -0.005) & (result["vol"] > vol20)).astype("float64")
    result["shrink_down_count_20d"] = shrink_down.groupby(result["ts_code"]).rolling(20).sum().reset_index(level=0, drop=True)
    result["expand_up_count_20d"] = expand_up.groupby(result["ts_code"]).rolling(20).sum().reset_index(level=0, drop=True)
    result["expand_no_down_count_20d"] = (
        expand_no_down.groupby(result["ts_code"]).rolling(20).sum().reset_index(level=0, drop=True)
    )
    return result


def _append_turnover_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped_turnover = result.groupby("ts_code")["turnover_rate_f"]
    for window in [3, 5, 10, 20]:
        result[f"turnover_mean_{window}d"] = (
            grouped_turnover.rolling(window).mean().reset_index(level=0, drop=True)
        )
    result["turnover_ratio_to_20d"] = result["turnover_rate_f"] / result["turnover_mean_20d"]
    result["turnover_trend_20d"] = (
        grouped_turnover.rolling(20).apply(_linear_slope, raw=True).reset_index(level=0, drop=True)
    )
    up_turnover = result["turnover_rate_f"].where(result["daily_return"] > 0)
    down_turnover = result["turnover_rate_f"].where(result["daily_return"] < 0)
    result["up_day_turnover_mean_20d"] = (
        up_turnover.groupby(result["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    result["down_day_turnover_mean_20d"] = (
        down_turnover.groupby(result["ts_code"]).rolling(20).mean().reset_index(level=0, drop=True)
    )
    return result


def _append_efficiency_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    up_return = result["daily_return"].where(result["daily_return"] > 0, 0.0)
    down_return = result["daily_return"].abs().where(result["daily_return"] < 0, 0.0)
    up_turnover = result["turnover_rate_f"].where(result["daily_return"] > 0, 0.0)
    down_turnover = result["turnover_rate_f"].where(result["daily_return"] < 0, 0.0)
    for window in [3, 5, 10, 20]:
        up_ret_sum = up_return.groupby(result["ts_code"]).rolling(window).sum().reset_index(level=0, drop=True)
        down_ret_sum = down_return.groupby(result["ts_code"]).rolling(window).sum().reset_index(level=0, drop=True)
        up_turnover_sum = up_turnover.groupby(result["ts_code"]).rolling(window).sum().reset_index(level=0, drop=True)
        down_turnover_sum = (
            down_turnover.groupby(result["ts_code"]).rolling(window).sum().reset_index(level=0, drop=True)
        )
        result[f"up_efficiency_{window}d"] = np.where(up_turnover_sum == 0, np.nan, up_ret_sum / up_turnover_sum)
        result[f"down_efficiency_{window}d"] = np.where(
            down_turnover_sum == 0,
            np.nan,
            down_ret_sum / down_turnover_sum,
        )
        result[f"efficiency_ratio_{window}d"] = result[f"up_efficiency_{window}d"] / result[f"down_efficiency_{window}d"]
    result["up_efficiency_trend_3_20"] = result["up_efficiency_3d"] - result["up_efficiency_20d"]
    result["down_efficiency_trend_3_20"] = result["down_efficiency_3d"] - result["down_efficiency_20d"]
    result["efficiency_ratio_trend_3_20"] = result["efficiency_ratio_3d"] - result["efficiency_ratio_20d"]
    return result


def _append_absorption_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    vol20 = result.groupby("ts_code")["vol"].rolling(20).mean().reset_index(level=0, drop=True)
    turnover20 = result.groupby("ts_code")["turnover_rate_f"].rolling(20).mean().reset_index(level=0, drop=True)
    result["volume_no_down_20d"] = (
        ((result["vol"] > vol20) & (result["daily_return"] >= -0.005))
        .astype("float64")
        .groupby(result["ts_code"])
        .rolling(20)
        .sum()
        .reset_index(level=0, drop=True)
    )
    result["high_turnover_no_down_20d"] = (
        ((result["turnover_rate_f"] > turnover20) & (result["daily_return"] >= -0.005))
        .astype("float64")
        .groupby(result["ts_code"])
        .rolling(20)
        .sum()
        .reset_index(level=0, drop=True)
    )
    result["long_lower_shadow_20d"] = (
        (result["lower_shadow"] > 0.45).astype("float64").groupby(result["ts_code"]).rolling(20).sum().reset_index(level=0, drop=True)
    )
    prior_low_20 = result.groupby("ts_code")["low"].rolling(20).min().reset_index(level=0, drop=True).groupby(result["ts_code"]).shift(1)
    result["recover_previous_low_20d"] = ((result["low"] < prior_low_20) & (result["close"] > prior_low_20)).astype("float64")
    result["second_bottom_not_break_20d"] = (
        ((result["second_bottom_20d"] > 0) & (result["close"] >= prior_low_20 * 0.98)).astype("float64")
    )
    return result


def _append_relative_strength_features(
    data: pd.DataFrame,
    benchmark_returns: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = data.copy()
    for name, frame in benchmark_returns.items():
        if frame is None or frame.empty or not {"trade_date", "close"}.issubset(frame.columns):
            continue
        benchmark = frame.loc[:, ["trade_date", "close"]].copy()
        benchmark["trade_date"] = benchmark["trade_date"].astype(str)
        benchmark[f"{name}_return_20d"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change(20)
        result = result.merge(benchmark.loc[:, ["trade_date", f"{name}_return_20d"]], on="trade_date", how="left")
        stock_ret_20d = result.groupby("ts_code")["close"].pct_change(20)
        result[f"relative_{name}_20d"] = stock_ret_20d - result[f"{name}_return_20d"]
    down_reference = None
    for candidate in ["hs300", "zz1000"]:
        column = f"{candidate}_return_20d"
        if column in result.columns:
            down_reference = result[column]
            break
    if down_reference is not None:
        result["benchmark_down_relative_return"] = np.where(
            down_reference < 0,
            result.groupby("ts_code")["daily_return"].transform(lambda values: values.rolling(20).mean()) - down_reference,
            np.nan,
        )
    return result


def _append_industry_features(data: pd.DataFrame, industry_map: pd.DataFrame | None) -> pd.DataFrame:
    result = data.copy()
    if industry_map is None or industry_map.empty or not {"ts_code", "industry"}.issubset(industry_map.columns):
        result["industry"] = None
        result["relative_industry_20d"] = np.nan
        return result
    mapping = industry_map.loc[:, ["ts_code", "industry"]].drop_duplicates(subset=["ts_code"], keep="last").copy()
    mapping["ts_code"] = mapping["ts_code"].astype(str)
    result = result.merge(mapping, on="ts_code", how="left")
    stock_ret_20d = result.groupby("ts_code")["close"].pct_change(20)
    industry_daily = (
        result.dropna(subset=["industry"])
        .groupby(["industry", "trade_date"], as_index=False)["daily_return"]
        .mean()
        .sort_values(["industry", "trade_date"])
        .reset_index(drop=True)
    )
    industry_daily["industry_return_20d"] = (
        (1.0 + industry_daily["daily_return"].fillna(0.0))
        .groupby(industry_daily["industry"])
        .rolling(20)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    result = result.merge(
        industry_daily.loc[:, ["industry", "trade_date", "industry_return_20d"]],
        on=["industry", "trade_date"],
        how="left",
    )
    result["relative_industry_20d"] = stock_ret_20d - result["industry_return_20d"]
    return result


def _append_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped_close = result.groupby("ts_code")["close"]
    for horizon in [5, 10, 20]:
        future_close = grouped_close.shift(-horizon)
        result[f"forward_return_{horizon}d"] = np.where(result["close"] == 0, np.nan, future_close / result["close"] - 1.0)
    result["future_20d_max_gain"] = result.groupby("ts_code", group_keys=False).apply(_future_max_gain)
    result["future_20d_max_drawdown"] = result.groupby("ts_code", group_keys=False).apply(_future_max_drawdown)
    positive = (result["future_20d_max_gain"] >= 0.20) & (result["future_20d_max_drawdown"] >= -0.10)
    negative = (result["future_20d_max_gain"] < 0.08) | (result["future_20d_max_drawdown"] <= -0.15)
    result["label"] = np.select([positive, negative], [1, 0], default=np.nan)
    return result


def _future_max_gain(group: pd.DataFrame) -> pd.Series:
    close = group["close"].astype(float)
    future_max = close.iloc[::-1].shift(1).rolling(20, min_periods=20).max().iloc[::-1]
    return pd.Series(np.where(close == 0, np.nan, future_max / close - 1.0), index=group.index)


def _future_max_drawdown(group: pd.DataFrame) -> pd.Series:
    close = group["close"].astype(float)
    future_min = close.iloc[::-1].shift(1).rolling(20, min_periods=20).min().iloc[::-1]
    return pd.Series(np.where(close == 0, np.nan, future_min / close - 1.0), index=group.index)


def _linear_slope(values: np.ndarray) -> float:
    valid = pd.Series(values).dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype="float64")
    return float(np.polyfit(x, valid.to_numpy(dtype="float64"), 1)[0])


def _cross_section_quantile(data: pd.DataFrame, feature: str, n_quantiles: int) -> pd.Series:
    ranks = data.groupby("trade_date")[feature].rank(method="first", pct=True)
    return np.ceil(ranks * n_quantiles).clip(1, n_quantiles).astype("Int64")


def _forward_columns(data: pd.DataFrame) -> list[str]:
    columns = [f"forward_return_{horizon}d" for horizon in [5, 10, 20]]
    columns.extend([column for column in ["future_20d_max_gain", "future_20d_max_drawdown"] if column in data.columns])
    return [column for column in columns if column in data.columns]


def _apply_condition(data: pd.DataFrame, condition: dict[str, object]) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    feature = condition.get("feature")
    op = condition.get("op")
    if not isinstance(feature, str) or feature not in data.columns:
        raise ValueError(f"Unknown condition feature: {feature}")
    values = pd.to_numeric(data[feature], errors="coerce")
    if op == "gte":
        mask = values >= float(condition["value"])
    elif op == "lte":
        mask = values <= float(condition["value"])
    elif op == "quantile_gte":
        threshold = values.groupby(data["trade_date"]).transform(lambda series: series.quantile(float(condition["value"])))
        mask = values >= threshold
    elif op == "quantile_lte":
        threshold = values.groupby(data["trade_date"]).transform(lambda series: series.quantile(float(condition["value"])))
        mask = values <= threshold
    else:
        raise ValueError(f"Unsupported condition op: {op}")
    return data.loc[mask.fillna(False)].copy()


def _condition_row(name: str, data: pd.DataFrame) -> dict[str, object]:
    return {
        "condition": name,
        "sample_count": int(len(data)),
        "positive_ratio": _safe_mean(data.get("label")),
        "mean_forward_20d": _safe_mean(data.get("forward_return_20d")),
        "median_forward_20d": _safe_median(data.get("forward_return_20d")),
        "mean_future_20d_max_drawdown": _safe_mean(data.get("future_20d_max_drawdown")),
    }


def _safe_mean(series) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_median(series) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _validate_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
