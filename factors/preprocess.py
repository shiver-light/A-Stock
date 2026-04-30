"""Cross-sectional factor preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_factor_panel(
    data: pd.DataFrame,
    *,
    factor_cols: list[str],
    date_col: str = "trade_date",
    industry_col: str | None = None,
    market_cap_col: str | None = None,
    directions: dict[str, int] | None = None,
    neutralize_industry: bool = False,
    neutralize_market_cap: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Preprocess factor columns cross-sectionally by date.

    Steps:
    1. fill missing by industry median, then market median
    2. MAD 3x clipping
    3. z-score standardization
    4. direction alignment so larger is always better
    5. reserve neutralization interface and emit warnings when not applied
    """

    if data.empty:
        result = data.copy()
        for factor in factor_cols:
            if factor not in result.columns:
                result[factor] = pd.Series(dtype="float64")
        return result, []

    working = data.copy()
    warnings: list[str] = []
    directions = directions or {}

    for factor in factor_cols:
        if factor not in working.columns:
            raise ValueError(f"Missing factor column: {factor}")

    grouped_frames: list[pd.DataFrame] = []
    for _, group in working.groupby(date_col, sort=False):
        group = group.copy()
        trade_date = str(group[date_col].iloc[0])
        for factor in factor_cols:
            group[factor] = _fill_missing(group, factor, industry_col)
            group[factor] = _mad_clip(group[factor])
            standardized, warning = _zscore(group[factor], factor_name=factor, trade_date=trade_date)
            group[factor] = standardized
            if warning:
                warnings.append(warning)
            direction = int(directions.get(factor, 1))
            group[factor] = group[factor] * (1 if direction >= 0 else -1)

        if neutralize_industry and not industry_col:
            warnings.append("industry neutralization requested but industry_col is unavailable; skipped.")
        if neutralize_market_cap and not market_cap_col:
            warnings.append("market_cap neutralization requested but market_cap_col is unavailable; skipped.")
        grouped_frames.append(group)

    result = pd.concat(grouped_frames, ignore_index=True)
    return result, sorted(set(warnings))


def _fill_missing(group: pd.DataFrame, factor_col: str, industry_col: str | None) -> pd.Series:
    values = group[factor_col].copy()
    if industry_col and industry_col in group.columns:
        industry_median = group.groupby(industry_col)[factor_col].transform("median")
        values = values.fillna(industry_median)
    values = values.fillna(values.median())
    return values


def _mad_clip(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    median = series.median()
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return series
    scale = 1.4826 * mad
    lower = median - 3.0 * scale
    upper = median + 3.0 * scale
    return series.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series, *, factor_name: str, trade_date: str) -> tuple[pd.Series, str | None]:
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        fallback = pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
        warning = f"std is zero for factor={factor_name} on trade_date={trade_date}; filled standardized values with 0.0."
        return fallback, warning
    return (series - mean) / std, None
