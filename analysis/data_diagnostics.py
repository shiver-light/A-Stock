"""Lightweight diagnostics for factor sample pool and missingness."""

from __future__ import annotations

import pandas as pd


def _validate_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _resolve_factor_cols(data: pd.DataFrame, factor_cols: list[str] | None) -> list[str]:
    if factor_cols is not None:
        return factor_cols
    return [column for column in data.columns if column not in {"trade_date", "ts_code"}]


def _prepare_factor_data(
    data: pd.DataFrame,
    *,
    factor_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    resolved_factor_cols = _resolve_factor_cols(data, factor_cols)
    if not resolved_factor_cols:
        raise ValueError("factor_cols must not be empty.")

    _validate_columns(data, ["trade_date", "ts_code", *resolved_factor_cols])
    if data.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", *resolved_factor_cols]), resolved_factor_cols

    prepared = (
        data.loc[:, ["trade_date", "ts_code", *resolved_factor_cols]]
        .copy()
        .assign(trade_date=data["trade_date"].astype(str), ts_code=data["ts_code"].astype(str))
        .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )
    return prepared, resolved_factor_cols


def summarize_universe_count_by_date(
    factor_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    asset_col: str = "ts_code",
) -> dict[str, object]:
    """Summarize the per-date raw sample pool before complete-case filtering."""

    _validate_columns(factor_data, [date_col, asset_col])
    if factor_data.empty:
        return {
            "date_col": date_col,
            "asset_col": asset_col,
            "date_count": 0,
            "summary": {
                "mean_universe_count": 0.0,
                "max_universe_count": 0,
                "min_universe_count": 0,
                "latest_universe_count": 0,
            },
            "per_date": [],
        }

    prepared = (
        factor_data.loc[:, [date_col, asset_col]]
        .copy()
        .assign(**{date_col: factor_data[date_col].astype(str), asset_col: factor_data[asset_col].astype(str)})
        .drop_duplicates(subset=[date_col, asset_col], keep="last")
        .sort_values([date_col, asset_col])
        .reset_index(drop=True)
    )
    counts = prepared.groupby(date_col, sort=True)[asset_col].nunique().rename("universe_count").reset_index()
    counts["count_change"] = counts["universe_count"].diff().fillna(0).astype(int)
    counts["count_change_ratio"] = counts["universe_count"].pct_change().fillna(0.0)

    return {
        "date_col": date_col,
        "asset_col": asset_col,
        "date_count": int(len(counts)),
        "summary": {
            "mean_universe_count": float(counts["universe_count"].mean()),
            "max_universe_count": int(counts["universe_count"].max()),
            "min_universe_count": int(counts["universe_count"].min()),
            "latest_universe_count": int(counts["universe_count"].iloc[-1]),
        },
        "per_date": counts.to_dict(orient="records"),
    }


def summarize_factor_coverage(
    factor_data: pd.DataFrame,
    *,
    factor_cols: list[str] | None = None,
) -> dict[str, object]:
    """Summarize per-factor available counts and missing ratios by date."""

    prepared, resolved_factor_cols = _prepare_factor_data(factor_data, factor_cols=factor_cols)
    if prepared.empty:
        return {
            "factor_cols": resolved_factor_cols,
            "date_count": 0,
            "factor_summary": [],
            "per_date": [],
        }

    grouped = prepared.groupby("trade_date", sort=True)
    per_date: list[dict[str, object]] = []
    factor_summary: list[dict[str, object]] = []

    for trade_date, group in grouped:
        universe_count = int(group["ts_code"].nunique())
        coverage_by_factor: dict[str, dict[str, object]] = {}
        for factor_name in resolved_factor_cols:
            available_count = int(group[factor_name].notna().sum())
            missing_count = int(group[factor_name].isna().sum())
            coverage_ratio = (available_count / universe_count) if universe_count else 0.0
            coverage_by_factor[factor_name] = {
                "available_count": available_count,
                "missing_count": missing_count,
                "coverage_ratio": float(coverage_ratio),
            }
        per_date.append(
            {
                "trade_date": str(trade_date),
                "universe_count": universe_count,
                "coverage_by_factor": coverage_by_factor,
            }
        )

    for factor_name in resolved_factor_cols:
        available = grouped[factor_name].apply(lambda values: int(values.notna().sum()))
        missing = grouped[factor_name].apply(lambda values: int(values.isna().sum()))
        universe = grouped["ts_code"].nunique()
        coverage_ratio = (available / universe).fillna(0.0)
        factor_summary.append(
            {
                "factor_name": factor_name,
                "mean_available_count": float(available.mean()) if not available.empty else 0.0,
                "mean_missing_count": float(missing.mean()) if not missing.empty else 0.0,
                "mean_coverage_ratio": float(coverage_ratio.mean()) if not coverage_ratio.empty else 0.0,
                "min_coverage_ratio": float(coverage_ratio.min()) if not coverage_ratio.empty else 0.0,
                "max_coverage_ratio": float(coverage_ratio.max()) if not coverage_ratio.empty else 0.0,
            }
        )

    return {
        "factor_cols": resolved_factor_cols,
        "date_count": int(prepared["trade_date"].nunique()),
        "factor_summary": factor_summary,
        "per_date": per_date,
    }


def summarize_complete_case_count_by_date(
    factor_data: pd.DataFrame,
    *,
    factor_cols: list[str] | None = None,
) -> dict[str, object]:
    """Summarize the per-date sample pool after requiring all factors to be non-null."""

    prepared, resolved_factor_cols = _prepare_factor_data(factor_data, factor_cols=factor_cols)
    if prepared.empty:
        return {
            "factor_cols": resolved_factor_cols,
            "date_count": 0,
            "summary": {
                "mean_complete_case_count": 0.0,
                "min_complete_case_count": 0,
                "max_complete_case_count": 0,
                "mean_complete_case_ratio": 0.0,
                "latest_complete_case_count": 0,
            },
            "per_date": [],
        }

    prepared["is_complete_case"] = prepared[resolved_factor_cols].notna().all(axis=1)
    grouped = prepared.groupby("trade_date", sort=True)
    rows: list[dict[str, object]] = []
    for trade_date, group in grouped:
        universe_count = int(group["ts_code"].nunique())
        complete_case_count = int(group["is_complete_case"].sum())
        incomplete_case_count = int(universe_count - complete_case_count)
        complete_case_ratio = (complete_case_count / universe_count) if universe_count else 0.0
        rows.append(
            {
                "trade_date": str(trade_date),
                "universe_count": universe_count,
                "complete_case_count": complete_case_count,
                "incomplete_case_count": incomplete_case_count,
                "complete_case_ratio": float(complete_case_ratio),
            }
        )

    per_date = pd.DataFrame(rows)
    return {
        "factor_cols": resolved_factor_cols,
        "date_count": int(len(per_date)),
        "summary": {
            "mean_complete_case_count": float(per_date["complete_case_count"].mean()),
            "min_complete_case_count": int(per_date["complete_case_count"].min()),
            "max_complete_case_count": int(per_date["complete_case_count"].max()),
            "mean_complete_case_ratio": float(per_date["complete_case_ratio"].mean()),
            "latest_complete_case_count": int(per_date["complete_case_count"].iloc[-1]),
        },
        "per_date": per_date.to_dict(orient="records"),
    }
