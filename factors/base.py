"""Shared helpers for factor outputs."""

from __future__ import annotations

import pandas as pd


def validate_factor_input(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_factor_output(data: pd.DataFrame, factor_name: str, value_column: str) -> pd.DataFrame:
    validate_factor_input(data, ["trade_date", "ts_code", value_column])
    result = data.loc[:, ["trade_date", "ts_code", value_column]].copy()
    result = result.rename(columns={value_column: "factor_value"})
    result["factor_name"] = factor_name
    return result.loc[:, ["trade_date", "ts_code", "factor_name", "factor_value"]].reset_index(drop=True)
