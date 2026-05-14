"""External market regime helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_external_regime_data(path: str | Path) -> pd.DataFrame:
    """Load pre-aligned external regime data from a CSV file."""

    return pd.read_csv(path, dtype={"trade_date": str})


def build_external_regime_flags(
    regime_data: pd.DataFrame,
    external_regime_filter: dict[str, object] | None,
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Build per-date external regime flags from pre-aligned columns.

    The input must already be aligned to A-share signal dates. For US market
    signals, this means the row for an A-share signal date should only contain
    US information known before that signal is generated.
    """

    if not external_regime_filter or external_regime_filter.get("enabled", True) is False:
        return pd.DataFrame(columns=[date_col, "external_regime_allowed"])
    if regime_data.empty:
        return pd.DataFrame(columns=[date_col, "external_regime_allowed"])
    if date_col not in regime_data.columns:
        raise ValueError(f"external_regime_filter requires {date_col} column.")

    data = regime_data.copy()
    data[date_col] = data[date_col].astype(str)
    data = data.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)

    allowed_col = external_regime_filter.get("allowed_col")
    rules = external_regime_filter.get("rules", [])
    if allowed_col:
        if not isinstance(allowed_col, str):
            raise ValueError("external_regime_filter allowed_col must be a string.")
        if allowed_col not in data.columns:
            raise ValueError(f"external_regime_filter allowed_col not found: {allowed_col}")
        allowed = _to_bool_series(data[allowed_col])
    elif rules:
        allowed = pd.Series(True, index=data.index)
    else:
        raise ValueError("external_regime_filter requires allowed_col or rules.")

    if rules:
        if not isinstance(rules, list):
            raise ValueError("external_regime_filter rules must be a list.")
        for rule in rules:
            allowed = allowed & _evaluate_rule(data, rule)

    return data.loc[:, [date_col]].assign(external_regime_allowed=allowed.fillna(False).astype(bool))


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _evaluate_rule(data: pd.DataFrame, rule: object) -> pd.Series:
    if not isinstance(rule, dict):
        raise ValueError("Each external regime rule must be a dictionary.")
    column = rule.get("column")
    op = rule.get("op")
    value = rule.get("value")
    if not isinstance(column, str) or not column:
        raise ValueError("Each external regime rule must contain a non-empty column.")
    if column not in data.columns:
        raise ValueError(f"External regime rule column not found: {column}")

    if op in {"is_true", "is_false"}:
        series = _to_bool_series(data[column])
        return series if op == "is_true" else ~series

    numeric_series = pd.to_numeric(data[column], errors="coerce")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"External regime rule value must be numeric: {value}") from exc

    if op == "gt":
        return numeric_series > numeric_value
    if op == "gte":
        return numeric_series >= numeric_value
    if op == "lt":
        return numeric_series < numeric_value
    if op == "lte":
        return numeric_series <= numeric_value
    if op == "eq":
        return numeric_series == numeric_value
    if op == "ne":
        return numeric_series != numeric_value
    raise ValueError(f"Unsupported external regime rule op: {op}")
