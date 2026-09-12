"""Load stock-level theme labels from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from market_ai.models import LimitStock, StrongStock, ThemeNormalization


REQUIRED_COLUMNS = ["stock_code", "primary_theme"]
OPTIONAL_COLUMNS = [
    "trade_date",
    "secondary_themes",
    "related_theme",
    "related_entities",
    "source",
    "confidence",
    "event_driven",
    "effective_start",
    "effective_end",
]


def load_stock_theme_labels(
    path: str | Path | None,
    *,
    trade_date: str,
    stock_codes: Iterable[str],
    min_confidence: float = 0.0,
) -> list[ThemeNormalization]:
    """Load theme labels for stock_codes that are effective on trade_date."""
    if path is None:
        return []
    label_path = Path(path)
    if not label_path.exists():
        return []

    data = pd.read_csv(label_path)
    data = _normalize_label_frame(data)
    if data.empty:
        return []

    allowed_codes = {str(code).strip() for code in stock_codes if str(code).strip()}
    data = data.loc[data["stock_code"].isin(allowed_codes)].copy()
    data = _filter_trade_date(data, str(trade_date))
    data = data.loc[data["confidence"].ge(float(min_confidence))].copy()
    if data.empty:
        return []

    data = data.sort_values(["confidence", "source", "stock_code"], ascending=[False, True, True])
    data = data.drop_duplicates(subset=["stock_code"], keep="first")
    return [_normalization_from_row(row) for row in data.to_dict(orient="records")]


def stock_codes_from_market_rows(limit_stocks: Iterable[LimitStock], strong_stocks: Iterable[StrongStock]) -> list[str]:
    """Return stable unique stock codes from market rows."""
    codes = [stock.stock_code for stock in limit_stocks]
    codes.extend(stock.stock_code for stock in strong_stocks)
    return list(dict.fromkeys(str(code).strip() for code in codes if str(code).strip()))


def merge_theme_normalizations(
    preferred: Iterable[ThemeNormalization],
    fallback: Iterable[ThemeNormalization],
) -> list[ThemeNormalization]:
    """Merge labels with preferred rows taking precedence by stock_code."""
    results: dict[str, ThemeNormalization] = {}
    for item in fallback:
        results[item.stock_code] = item
    for item in preferred:
        results[item.stock_code] = item
    return list(results.values())


def _normalize_label_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Stock theme label CSV missing required columns: {missing}")

    result = data.copy()
    for column in [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]:
        if column not in result.columns:
            result[column] = ""
        if column == "confidence":
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        else:
            result[column] = result[column].fillna("").astype(str).str.strip()
    for column in ["trade_date", "effective_start", "effective_end"]:
        result[column] = result[column].map(_clean_date_value)

    valid = result["stock_code"].ne("") & result["primary_theme"].ne("")
    return result.loc[valid, [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].copy()


def _filter_trade_date(data: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    result = data.copy()
    if "trade_date" in result.columns:
        same_trade_date = result["trade_date"].eq("") | result["trade_date"].eq(trade_date)
        result = result.loc[same_trade_date].copy()
    if "effective_start" in result.columns:
        start_ok = result["effective_start"].eq("") | result["effective_start"].le(trade_date)
        result = result.loc[start_ok].copy()
    if "effective_end" in result.columns:
        end_ok = result["effective_end"].eq("") | result["effective_end"].ge(trade_date)
        result = result.loc[end_ok].copy()
    return result


def _normalization_from_row(row: dict[str, object]) -> ThemeNormalization:
    return ThemeNormalization(
        stock_code=str(row["stock_code"]),
        primary_theme=str(row["primary_theme"]),
        secondary_themes=_split_labels(row.get("secondary_themes")),
        related_entities=_split_labels(row.get("related_entities")) + _split_labels(row.get("related_theme")),
        event_driven=_bool_value(row.get("event_driven")),
        confidence=float(row.get("confidence") or 0.0),
        source=str(row.get("source") or "stock_theme_label"),
    )


def _split_labels(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = []
    for item in text.replace("，", "|").replace(",", "|").replace("；", "|").replace(";", "|").split("|"):
        item = item.strip()
        if item:
            parts.append(item)
    return list(dict.fromkeys(parts))


def _clean_date_value(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _bool_value(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "event", "事件", "是"}
