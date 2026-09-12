"""Tushare limit-list market provider for radar workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig
from market_ai.models import LimitStock, StrongStock
from market_ai.providers.market.tushare_daily import TushareDailyMarketProvider


LIMIT_LIST_FIELDS = (
    "trade_date,ts_code,name,close,pct_chg,amount,turnover_ratio,"
    "first_time,last_time,open_times,limit_times,fd_amount,industry,limit"
)


class TushareLimitMarketProvider(TushareDailyMarketProvider):
    """Use Tushare limit_list_d for limit rows and daily bars for strong rows.

    Tushare docs describe limit_list_d as the daily A-share up/down-limit and
    failed-limit list from 2020 onward. Field availability may vary by account
    permission, so this provider maps optional columns defensively.
    """

    def __init__(
        self,
        *,
        client: TushareClient | None = None,
        universe_name: str | None = None,
        cache_dir: str | Path = "data/cache/market_ai/tushare_limit",
        refresh: bool = False,
    ) -> None:
        super().__init__(
            client=client or TushareClient(TushareClientConfig()),
            universe_name=universe_name,
            cache_dir=cache_dir,
            refresh=refresh,
        )

    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        """Return limit-up, touched-limit, and failed-limit rows from limit_list_d."""
        data = self._load_limit_snapshot(trade_date)
        if data.empty:
            return []
        data = self._filter_universe(data, trade_date)
        return [_limit_stock_from_row(row) for row in data.to_dict(orient="records")]

    def _load_limit_snapshot(self, trade_date: str) -> pd.DataFrame:
        cache_path = self.cache_dir / "limit_list_d" / f"{trade_date}.parquet"
        if cache_path.exists() and not self.refresh:
            return pd.read_parquet(cache_path)

        fetched = self.client.limit_list_d(trade_date=trade_date, fields=LIMIT_LIST_FIELDS)
        data = _normalize_limit_list(fetched, trade_date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path, index=False)
        return data


def _normalize_limit_list(data: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=_output_columns())
    result = data.copy()
    if "trade_date" not in result.columns or "ts_code" not in result.columns:
        raise ValueError("limit_list_d result missing required columns: trade_date, ts_code")
    result["trade_date"] = result["trade_date"].astype(str)
    result = result.loc[result["trade_date"] == str(trade_date)].copy()
    result = result.rename(columns={"ts_code": "stock_code", "name": "stock_name"})
    for column in _output_columns():
        if column not in result.columns:
            result[column] = pd.NA
    result["stock_name"] = result["stock_name"].fillna(result["stock_code"]).astype(str)
    result["industry"] = result["industry"].fillna("").astype(str)
    for column in ["close", "pct_chg", "amount", "turnover_ratio", "fd_amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = _filter_up_or_failed_up_rows(result)
    return (
        result.loc[:, _output_columns()]
        .sort_values(["pct_chg", "amount", "stock_code"], ascending=[False, False, True])
        .drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
        .reset_index(drop=True)
    )


def _filter_up_or_failed_up_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    if "limit" in result.columns:
        limit_text = result["limit"].fillna("").astype(str).str.upper()
        has_limit_flag = limit_text.ne("")
        up_like = limit_text.str.contains("U|Z|涨|炸", regex=True)
        if has_limit_flag.any():
            result = result.loc[up_like].copy()
    if "pct_chg" in result.columns:
        result = result.loc[result["pct_chg"].fillna(0.0).ge(0.0)].copy()
    return result


def _limit_stock_from_row(row: dict[str, object]) -> LimitStock:
    return LimitStock(
        trade_date=str(row["trade_date"]),
        stock_code=str(row["stock_code"]),
        stock_name=str(row.get("stock_name") or row["stock_code"]),
        close=_float_or_none(row.get("close")),
        pct_chg=_float_or_none(row.get("pct_chg")),
        amount=_float_or_none(row.get("amount")),
        turnover_rate=_float_or_none(row.get("turnover_ratio")),
        volume_ratio=None,
        first_limit_time=_text_or_none(row.get("first_time")),
        last_limit_time=_text_or_none(row.get("last_time")),
        open_count=_int_or_none(row.get("open_times")),
        sealed_amount=_float_or_none(row.get("fd_amount")),
        consecutive_limit_count=_int_or_none(row.get("limit_times")),
        limit_reason=None,
        industry=_text_or_none(row.get("industry")),
        concepts=[],
        status=_status_from_limit(row.get("limit")),
        source="tushare_limit_list_d",
    )


def _status_from_limit(value: object) -> str:
    text = str(value or "").strip().upper()
    if "Z" in text or "炸" in text:
        return "failed_limit"
    if "T" in text or "触" in text:
        return "touched_limit"
    return "limit_up"


def _output_columns() -> list[str]:
    return [
        "trade_date",
        "stock_code",
        "stock_name",
        "close",
        "pct_chg",
        "amount",
        "turnover_ratio",
        "first_time",
        "last_time",
        "open_times",
        "limit_times",
        "fd_amount",
        "industry",
        "limit",
    ]


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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
