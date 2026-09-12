"""Local CSV market provider for offline radar workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_ai.models import LimitStock, StrongStock
from market_ai.providers.market.base import MarketProvider


LIMIT_REQUIRED_COLUMNS = ["trade_date", "stock_code", "stock_name"]
STRONG_REQUIRED_COLUMNS = ["trade_date", "stock_code", "stock_name", "pct_chg"]
OPTIONAL_COLUMNS = [
    "close",
    "pct_chg",
    "amount",
    "turnover_rate",
    "volume_ratio",
    "first_limit_time",
    "last_limit_time",
    "open_count",
    "sealed_amount",
    "consecutive_limit_count",
    "limit_reason",
    "industry",
    "concepts",
    "status",
    "source",
]


class LocalCsvMarketProvider(MarketProvider):
    """Read limit and strong stock radar inputs from local CSV files."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        limit_filename: str = "limit_stocks.csv",
        strong_filename: str = "strong_stocks.csv",
        quotes_filename: str = "daily_quotes.csv",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.limit_path = self.data_dir / limit_filename
        self.strong_path = self.data_dir / strong_filename
        self.quotes_path = self.data_dir / quotes_filename

    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        """Return limit/touched/failed limit stock rows for one trade date."""
        data = _read_csv_or_empty(self.limit_path)
        if data.empty:
            return []
        data = _prepare_frame(data, LIMIT_REQUIRED_COLUMNS)
        data = data.loc[data["trade_date"] == str(trade_date)].copy()
        return [_limit_stock_from_row(row) for row in data.to_dict(orient="records")]

    def get_strong_stocks(self, *, trade_date: str, min_pct_chg: float = 7.0) -> list[StrongStock]:
        """Return strong non-limit rows for one trade date."""
        data = _read_csv_or_empty(self.strong_path)
        if data.empty:
            return []
        data = _prepare_frame(data, STRONG_REQUIRED_COLUMNS)
        data = data.loc[data["trade_date"] == str(trade_date)].copy()
        data["pct_chg"] = pd.to_numeric(data["pct_chg"], errors="coerce")
        data = data.loc[data["pct_chg"].ge(float(min_pct_chg))].copy()
        return [_strong_stock_from_row(row) for row in data.to_dict(orient="records")]

    def get_daily_quotes(self, *, trade_date: str, stock_codes: Iterable[str] | None = None) -> pd.DataFrame:
        """Return raw daily quote rows for one trade date when a CSV is present."""
        data = _read_csv_or_empty(self.quotes_path)
        if data.empty:
            return data
        if "trade_date" not in data.columns:
            raise ValueError("daily_quotes.csv missing required column: trade_date")
        data = data.copy()
        data["trade_date"] = data["trade_date"].astype(str).str.strip()
        data = data.loc[data["trade_date"] == str(trade_date)].copy()
        if stock_codes is not None and "stock_code" in data.columns:
            allowed = {str(code).strip() for code in stock_codes}
            data = data.loc[data["stock_code"].astype(str).str.strip().isin(allowed)].copy()
        return data.reset_index(drop=True)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_frame(data: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Local market CSV missing required columns: {missing}")
    result = data.copy()
    for column in [*required_columns, *OPTIONAL_COLUMNS]:
        if column not in result.columns:
            result[column] = ""
    result["trade_date"] = result["trade_date"].astype(str).str.strip()
    result["stock_code"] = result["stock_code"].astype(str).str.strip()
    result["stock_name"] = result["stock_name"].astype(str).str.strip()
    valid = result["trade_date"].ne("") & result["stock_code"].ne("") & result["stock_name"].ne("")
    return result.loc[valid].reset_index(drop=True)


def _limit_stock_from_row(row: dict[str, object]) -> LimitStock:
    return LimitStock(
        trade_date=str(row["trade_date"]),
        stock_code=str(row["stock_code"]),
        stock_name=str(row["stock_name"]),
        close=_float_or_none(row.get("close")),
        pct_chg=_float_or_none(row.get("pct_chg")),
        amount=_float_or_none(row.get("amount")),
        turnover_rate=_float_or_none(row.get("turnover_rate")),
        volume_ratio=_float_or_none(row.get("volume_ratio")),
        first_limit_time=_text_or_none(row.get("first_limit_time")),
        last_limit_time=_text_or_none(row.get("last_limit_time")),
        open_count=_int_or_none(row.get("open_count")),
        sealed_amount=_float_or_none(row.get("sealed_amount")),
        consecutive_limit_count=_int_or_none(row.get("consecutive_limit_count")),
        limit_reason=_text_or_none(row.get("limit_reason")),
        industry=_text_or_none(row.get("industry")),
        concepts=_parse_concepts(row.get("concepts")),
        status=_text_or_none(row.get("status")) or "limit_up",
        source=_text_or_none(row.get("source")) or "local_csv",
    )


def _strong_stock_from_row(row: dict[str, object]) -> StrongStock:
    return StrongStock(
        trade_date=str(row["trade_date"]),
        stock_code=str(row["stock_code"]),
        stock_name=str(row["stock_name"]),
        pct_chg=float(row["pct_chg"]),
        close=_float_or_none(row.get("close")),
        amount=_float_or_none(row.get("amount")),
        turnover_rate=_float_or_none(row.get("turnover_rate")),
        volume_ratio=_float_or_none(row.get("volume_ratio")),
        industry=_text_or_none(row.get("industry")),
        concepts=_parse_concepts(row.get("concepts")),
        source=_text_or_none(row.get("source")) or "local_csv",
    )


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _int_or_none(value: object) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_concepts(value: object) -> list[str]:
    text = _text_or_none(value)
    if text is None:
        return []
    return [item.strip() for item in re.split(r"[|,，;；]", text) if item.strip()]
