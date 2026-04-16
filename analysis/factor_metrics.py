"""Cross-sectional factor analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = ["trade_date", "ts_code", "factor_value"]


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _prepare_factor_data(factor_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(factor_data, FACTOR_COLUMNS)
    return (
        factor_data.loc[:, FACTOR_COLUMNS]
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )


def _prepare_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(price_data, ["trade_date", "ts_code", "close"])
    return (
        price_data.loc[:, ["trade_date", "ts_code", "close"]]
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )


def _forward_return_columns(horizons: list[int]) -> list[str]:
    return [f"forward_return_{horizon}d" for horizon in horizons]


def _resolve_forward_return_columns(forward_returns: pd.DataFrame) -> list[str]:
    return [column for column in forward_returns.columns if column.startswith("forward_return_")]


def _calc_group_correlation(group: pd.DataFrame, return_column: str, method: str) -> float | None:
    valid = group.loc[:, ["factor_value", return_column]].dropna()
    if len(valid) < 2:
        return None

    correlation = valid["factor_value"].corr(valid[return_column], method=method)
    if pd.isna(correlation):
        return None
    return float(correlation)


def _extract_horizon(column_name: str) -> int:
    return int(column_name.removeprefix("forward_return_").removesuffix("d"))


def calc_forward_returns(price_data: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Calculate per-stock forward close-to-close returns for the requested horizons."""

    if not horizons:
        raise ValueError("horizons must not be empty.")

    unique_horizons = sorted(set(horizons))
    if any(horizon <= 0 for horizon in unique_horizons):
        raise ValueError("horizons must contain positive integers only.")

    if price_data.empty:
        return _empty_frame(["trade_date", "ts_code"] + _forward_return_columns(unique_horizons))

    prepared = _prepare_price_data(price_data)
    result = prepared.loc[:, ["trade_date", "ts_code"]].copy()

    grouped_close = prepared.groupby("ts_code")["close"]
    for horizon in unique_horizons:
        column = f"forward_return_{horizon}d"
        future_close = grouped_close.shift(-horizon)
        result[column] = future_close.div(prepared["close"]).sub(1.0)

    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def calc_ic(
    factor_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate cross-sectional IC by trade date and forward-return horizon."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'.")

    if factor_data.empty or forward_returns.empty:
        return _empty_frame(["trade_date", "horizon", "ic", "n_obs", "method"])

    prepared_factor = _prepare_factor_data(factor_data)
    _validate_columns(forward_returns, ["trade_date", "ts_code"])
    return_columns = _resolve_forward_return_columns(forward_returns)
    if not return_columns:
        raise ValueError("forward_returns must contain at least one forward_return_* column.")

    prepared_returns = (
        forward_returns.loc[:, ["trade_date", "ts_code"] + return_columns]
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )
    merged = prepared_factor.merge(prepared_returns, on=["trade_date", "ts_code"], how="inner")
    if merged.empty:
        return _empty_frame(["trade_date", "horizon", "ic", "n_obs", "method"])

    rows: list[dict[str, object]] = []
    for return_column in return_columns:
        valid = merged.loc[:, ["trade_date", "factor_value", return_column]].dropna()
        if valid.empty:
            continue

        for trade_date, group in valid.groupby("trade_date", sort=True):
            n_obs = len(group)
            ic_value = _calc_group_correlation(group, return_column, method=method)
            if ic_value is None:
                continue

            rows.append(
                {
                    "trade_date": trade_date,
                    "horizon": _extract_horizon(return_column),
                    "ic": ic_value,
                    "n_obs": int(n_obs),
                    "method": method,
                }
            )

    if not rows:
        return _empty_frame(["trade_date", "horizon", "ic", "n_obs", "method"])

    return pd.DataFrame(rows).sort_values(["trade_date", "horizon"]).reset_index(drop=True)


def calc_rank_ic(factor_data: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-sectional rank IC using Spearman correlation."""

    return calc_ic(factor_data, forward_returns, method="spearman")


def calc_quantile_groups(factor_data: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Assign cross-sectional quantile labels to each factor observation by date."""

    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be a positive integer.")

    if factor_data.empty:
        return _empty_frame(["trade_date", "ts_code", "factor_value", "quantile"])

    prepared = _prepare_factor_data(factor_data)
    result = prepared.copy()
    result["quantile"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    valid_mask = result["factor_value"].notna()
    if valid_mask.any():
        pct_rank = result.loc[valid_mask].groupby("trade_date")["factor_value"].rank(
            method="first",
            pct=True,
            ascending=True,
        )
        quantiles = np.ceil(pct_rank * n_quantiles).clip(1, n_quantiles).astype("int64")
        result.loc[valid_mask, "quantile"] = quantiles.astype("Int64")

    return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def calc_quantile_returns(
    factor_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Calculate mean forward returns for each cross-sectional factor quantile."""

    if factor_data.empty or forward_returns.empty:
        return _empty_frame(["trade_date", "quantile", "horizon", "forward_return", "n_obs"])

    quantiles = calc_quantile_groups(factor_data, n_quantiles=n_quantiles)
    _validate_columns(forward_returns, ["trade_date", "ts_code"])
    return_columns = _resolve_forward_return_columns(forward_returns)
    if not return_columns:
        raise ValueError("forward_returns must contain at least one forward_return_* column.")

    prepared_returns = (
        forward_returns.loc[:, ["trade_date", "ts_code"] + return_columns]
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )
    merged = quantiles.merge(prepared_returns, on=["trade_date", "ts_code"], how="inner")
    merged = merged.dropna(subset=["quantile"])
    if merged.empty:
        return _empty_frame(["trade_date", "quantile", "horizon", "forward_return", "n_obs"])

    rows: list[dict[str, object]] = []
    for return_column in return_columns:
        subset = merged.loc[:, ["trade_date", "quantile", return_column]].dropna()
        if subset.empty:
            continue

        grouped = (
            subset.groupby(["trade_date", "quantile"], sort=True)[return_column]
            .agg(["mean", "count"])
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "trade_date": row.trade_date,
                    "quantile": int(row.quantile),
                    "horizon": _extract_horizon(return_column),
                    "forward_return": float(row.mean),
                    "n_obs": int(row.count),
                }
            )

    if not rows:
        return _empty_frame(["trade_date", "quantile", "horizon", "forward_return", "n_obs"])

    return (
        pd.DataFrame(rows)
        .sort_values(["trade_date", "horizon", "quantile"])
        .reset_index(drop=True)
    )


def calc_factor_coverage(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-date factor non-null coverage after de-duplicating keys."""

    if factor_data.empty:
        return _empty_frame(["trade_date", "coverage", "non_null_count", "total_count"])

    prepared = _prepare_factor_data(factor_data)
    coverage = (
        prepared.groupby("trade_date", sort=True)["factor_value"]
        .agg(non_null_count=lambda values: int(values.notna().sum()), total_count="size")
        .reset_index()
    )
    coverage["coverage"] = coverage["non_null_count"] / coverage["total_count"]
    return coverage.loc[:, ["trade_date", "coverage", "non_null_count", "total_count"]]
