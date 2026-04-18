"""Structured data quality checks for research inputs and outputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _detect_date_column(data: pd.DataFrame) -> str:
    for column in ["trade_date", "as_of_date"]:
        if column in data.columns:
            return column
    raise ValueError("Expected either 'trade_date' or 'as_of_date' in input data.")


def check_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> dict[str, object]:
    """Report duplicate key groups and sample duplicate rows."""

    if not keys:
        raise ValueError("keys must not be empty.")
    _validate_columns(df, keys)

    if df.empty:
        return {
            "keys": keys,
            "row_count": 0,
            "duplicate_row_count": 0,
            "duplicate_group_count": 0,
            "has_duplicates": False,
            "duplicate_samples": [],
        }

    duplicated_mask = df.duplicated(subset=keys, keep=False)
    duplicates = df.loc[duplicated_mask, keys].copy()
    duplicate_groups = duplicates.drop_duplicates(subset=keys)

    return {
        "keys": keys,
        "row_count": int(len(df)),
        "duplicate_row_count": int(duplicated_mask.sum()),
        "duplicate_group_count": int(len(duplicate_groups)),
        "has_duplicates": bool(duplicated_mask.any()),
        "duplicate_samples": duplicate_groups.head(10).to_dict(orient="records"),
    }


def check_missing_ratio_by_date(df: pd.DataFrame, value_cols: list[str]) -> dict[str, object]:
    """Summarize daily sample counts and missing ratios for selected value columns."""

    if not value_cols:
        raise ValueError("value_cols must not be empty.")
    date_col = _detect_date_column(df)
    _validate_columns(df, [date_col] + value_cols)

    if df.empty:
        return {
            "date_col": date_col,
            "value_cols": value_cols,
            "row_count": 0,
            "date_count": 0,
            "per_date": [],
            "column_summary": [],
        }

    data = df.loc[:, [date_col] + value_cols].copy()
    data[date_col] = data[date_col].astype(str)
    grouped = data.groupby(date_col, sort=True)
    sample_count = grouped.size().rename("sample_count")
    per_date = sample_count.reset_index()
    per_date["sample_count_change"] = per_date["sample_count"].diff().fillna(0).astype(int)
    per_date["sample_count_change_ratio"] = (
        per_date["sample_count"].pct_change().replace([pd.NA, pd.NaT], 0.0).fillna(0.0)
    )

    column_summary: list[dict[str, object]] = []
    for column in value_cols:
        missing_count = grouped[column].apply(lambda values: int(values.isna().sum())).rename(f"{column}_missing_count")
        missing_ratio = grouped[column].apply(lambda values: float(values.isna().mean())).rename(f"{column}_missing_ratio")
        per_date = per_date.merge(missing_count.reset_index(), on=date_col, how="left")
        per_date = per_date.merge(missing_ratio.reset_index(), on=date_col, how="left")

        ratio_series = per_date[f"{column}_missing_ratio"]
        anomaly_mask = ratio_series > 0.5
        column_summary.append(
            {
                "value_col": column,
                "mean_missing_ratio": float(ratio_series.mean()) if not ratio_series.empty else 0.0,
                "max_missing_ratio": float(ratio_series.max()) if not ratio_series.empty else 0.0,
                "anomaly_dates": per_date.loc[anomaly_mask, date_col].astype(str).tolist(),
            }
        )

    return {
        "date_col": date_col,
        "value_cols": value_cols,
        "row_count": int(len(df)),
        "date_count": int(per_date[date_col].nunique()),
        "per_date": per_date.to_dict(orient="records"),
        "column_summary": column_summary,
    }


def check_factor_signal_alignment(
    factor_df: pd.DataFrame,
    execution_dates: pd.DataFrame | Sequence[str],
) -> dict[str, object]:
    """Check whether factor dates overlap execution dates or miss required signal dates."""

    _validate_columns(factor_df, ["trade_date"])
    factor_dates = pd.Series(factor_df["trade_date"].astype(str).dropna().unique()).sort_values().tolist()

    if isinstance(execution_dates, pd.DataFrame):
        if "execution_date" not in execution_dates.columns:
            raise ValueError("execution_dates DataFrame must contain 'execution_date'.")
        execution_date_list = execution_dates["execution_date"].astype(str).dropna().drop_duplicates().sort_values().tolist()
        signal_date_list = (
            execution_dates["signal_date"].astype(str).dropna().drop_duplicates().sort_values().tolist()
            if "signal_date" in execution_dates.columns
            else []
        )
    else:
        execution_date_list = sorted({str(value) for value in execution_dates})
        signal_date_list = []

    overlap_dates = sorted(set(factor_dates) & set(execution_date_list))
    missing_signal_dates = sorted(set(signal_date_list) - set(factor_dates))

    return {
        "factor_date_count": int(len(factor_dates)),
        "execution_date_count": int(len(execution_date_list)),
        "signal_date_count": int(len(signal_date_list)),
        "overlap_with_execution_dates": overlap_dates,
        "missing_signal_dates": missing_signal_dates,
        "has_alignment_risk": bool(overlap_dates or missing_signal_dates),
    }


def check_universe_stability(universe_data: pd.DataFrame) -> dict[str, object]:
    """Track daily universe size changes and membership turnover across dates."""

    date_col = _detect_date_column(universe_data)
    _validate_columns(universe_data, [date_col, "ts_code"])

    if universe_data.empty:
        return {
            "date_col": date_col,
            "date_count": 0,
            "per_date": [],
            "summary": {
                "mean_constituent_count": 0.0,
                "max_constituent_count": 0,
                "min_constituent_count": 0,
                "mean_turnover_ratio": 0.0,
            },
        }

    prepared = (
        universe_data.loc[:, [date_col, "ts_code"]]
        .copy()
        .assign(**{date_col: universe_data[date_col].astype(str)})
        .drop_duplicates(subset=[date_col, "ts_code"], keep="last")
        .sort_values([date_col, "ts_code"])
        .reset_index(drop=True)
    )

    date_to_assets = {
        date: set(group["ts_code"].astype(str).tolist())
        for date, group in prepared.groupby(date_col, sort=True)
    }

    rows: list[dict[str, object]] = []
    previous_assets: set[str] | None = None
    for date, assets in date_to_assets.items():
        additions = sorted(assets - previous_assets) if previous_assets is not None else []
        removals = sorted(previous_assets - assets) if previous_assets is not None else []
        denominator = max(len(previous_assets or set()), len(assets)) if previous_assets is not None else 0
        turnover_ratio = ((len(additions) + len(removals)) / denominator) if denominator else 0.0
        rows.append(
            {
                date_col: date,
                "constituent_count": int(len(assets)),
                "constituent_count_change": int(len(assets) - len(previous_assets or set())),
                "additions": additions[:20],
                "removals": removals[:20],
                "turnover_ratio": float(turnover_ratio),
            }
        )
        previous_assets = assets

    per_date = pd.DataFrame(rows)
    return {
        "date_col": date_col,
        "date_count": int(len(per_date)),
        "per_date": per_date.to_dict(orient="records"),
        "summary": {
            "mean_constituent_count": float(per_date["constituent_count"].mean()),
            "max_constituent_count": int(per_date["constituent_count"].max()),
            "min_constituent_count": int(per_date["constituent_count"].min()),
            "mean_turnover_ratio": float(per_date["turnover_ratio"].mean()),
        },
    }
