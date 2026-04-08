"""Minimal ranking-based stock selection logic."""

from __future__ import annotations

import pandas as pd


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def combine_factor_scores(
    factor_data: pd.DataFrame,
    *,
    factor_cols: list[str],
    weights: dict[str, float] | None = None,
    directions: dict[str, int] | None = None,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
    dropna: bool = True,
) -> pd.DataFrame:
    """Combine multiple factor columns into a single equal-weight or weighted score.

    The score is computed cross-sectionally per date using percentile ranks so factors
    with different scales can be averaged without changing the data layer.
    """

    _validate_columns(factor_data, [date_col, asset_col] + factor_cols)
    if not factor_cols:
        raise ValueError("factor_cols must not be empty.")

    weights = weights or {column: 1.0 for column in factor_cols}
    directions = directions or {}

    missing_weights = [column for column in factor_cols if column not in weights]
    if missing_weights:
        raise ValueError(f"Missing weights for factor columns: {missing_weights}")

    result = factor_data.loc[:, [date_col, asset_col] + factor_cols].copy()
    if dropna:
        result = result.dropna(subset=factor_cols).reset_index(drop=True)

    total_weight = sum(abs(weights[column]) for column in factor_cols)
    if total_weight == 0:
        raise ValueError("weights must not all be zero.")

    score = pd.Series(0.0, index=result.index, dtype="float64")
    for column in factor_cols:
        ascending = directions.get(column, 1) < 0
        ranked = result.groupby(date_col)[column].rank(method="average", pct=True, ascending=ascending)
        score = score + ranked * weights[column]

    result["score"] = score / total_weight
    return (
        result.loc[:, [date_col, asset_col, "score"]]
        .sort_values([date_col, asset_col])
        .reset_index(drop=True)
    )


def rank_signal(
    factor_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
    score_col: str = "factor_value",
    ascending: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """Rank stocks independently for each date."""

    _validate_columns(factor_data, [date_col, asset_col, score_col])
    result = factor_data.loc[:, [date_col, asset_col, score_col]].copy()
    result = result.rename(columns={score_col: "score"})
    if dropna:
        result = result.dropna(subset=["score"]).reset_index(drop=True)

    result = result.sort_values([date_col, "score", asset_col], ascending=[True, ascending, True]).reset_index(drop=True)
    result["rank"] = result.groupby(date_col)["score"].rank(method="first", ascending=ascending)
    result["rank"] = result["rank"].astype(int)
    return result.loc[:, [date_col, asset_col, "score", "rank"]]


def top_n_selection(
    ranked_data: pd.DataFrame,
    *,
    top_n: int,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
    score_col: str = "score",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Select top N stocks for each date from ranked signal data."""

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    _validate_columns(ranked_data, [date_col, asset_col, score_col, rank_col])
    result = ranked_data.loc[:, [date_col, asset_col, score_col, rank_col]].copy()
    result = result.sort_values([date_col, rank_col, asset_col]).reset_index(drop=True)
    result["selected"] = result[rank_col] <= top_n
    return result.loc[:, [date_col, asset_col, score_col, rank_col, "selected"]]
