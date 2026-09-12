"""Tushare daily-bar market provider for radar workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig
from universe.provider import get_universe
from market_ai.models import LimitStock, StrongStock
from market_ai.providers.market.base import MarketProvider


DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio"
STOCK_BASIC_FIELDS = "ts_code,name,industry"


class TushareDailyMarketProvider(MarketProvider):
    """Build radar market inputs from Tushare daily and daily_basic data.

    This provider uses only data available for trade_date after close. Limit-up
    rows are approximated by pct_chg >= limit_up_pct_threshold because the
    dedicated limit-list source is not wired in this MVP.
    """

    def __init__(
        self,
        *,
        client: TushareClient | None = None,
        universe_name: str | None = None,
        cache_dir: str | Path = "data/cache/market_ai/tushare_daily",
        refresh: bool = False,
        limit_up_pct_threshold: float = 9.8,
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.universe_name = universe_name
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.limit_up_pct_threshold = float(limit_up_pct_threshold)

    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        """Return approximate final limit-up rows using same-day pct_chg."""
        data = self._load_daily_snapshot(trade_date)
        if data.empty:
            return []
        data = data.loc[data["pct_chg"].ge(self.limit_up_pct_threshold)].copy()
        return [_limit_stock_from_row(row, self.limit_up_pct_threshold) for row in data.to_dict(orient="records")]

    def get_strong_stocks(self, *, trade_date: str, min_pct_chg: float = 7.0) -> list[StrongStock]:
        """Return strong non-limit rows using same-day pct_chg."""
        data = self._load_daily_snapshot(trade_date)
        if data.empty:
            return []
        data = data.loc[
            data["pct_chg"].ge(float(min_pct_chg)) & data["pct_chg"].lt(self.limit_up_pct_threshold)
        ].copy()
        return [_strong_stock_from_row(row) for row in data.to_dict(orient="records")]

    def get_daily_quotes(self, *, trade_date: str, stock_codes: Iterable[str] | None = None) -> pd.DataFrame:
        """Return the normalized same-day Tushare quote snapshot."""
        data = self._load_daily_snapshot(trade_date)
        if stock_codes is not None and not data.empty:
            allowed = {str(code).strip() for code in stock_codes}
            data = data.loc[data["stock_code"].isin(allowed)].copy()
        return data.reset_index(drop=True)

    def _load_daily_snapshot(self, trade_date: str) -> pd.DataFrame:
        cache_path = self.cache_dir / "daily_snapshot" / f"{trade_date}.parquet"
        if cache_path.exists() and not self.refresh:
            return pd.read_parquet(cache_path)

        daily = self.client.daily(trade_date=trade_date, fields=DAILY_FIELDS)
        daily_basic = self.client.daily_basic(trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
        stock_basic = self.client.stock_basic(list_status="L", fields=STOCK_BASIC_FIELDS)
        data = _normalize_snapshot(daily, daily_basic, stock_basic, trade_date)
        data = self._filter_universe(data, trade_date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path, index=False)
        return data

    def _filter_universe(self, data: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        if not self.universe_name or data.empty:
            return data
        universe = get_universe(self.universe_name, trade_date, refresh=self.refresh)
        if universe.empty:
            return data.iloc[0:0].copy()
        allowed = set(universe["ts_code"].astype(str))
        return data.loc[data["stock_code"].isin(allowed)].copy()


def _normalize_snapshot(
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    stock_basic: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    if daily is None or daily.empty:
        return _empty_snapshot()
    data = daily.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data = data.loc[data["trade_date"] == str(trade_date)].copy()
    data = data.rename(columns={"ts_code": "stock_code"})
    for column in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if daily_basic is not None and not daily_basic.empty:
        basic = daily_basic.copy()
        basic["trade_date"] = basic["trade_date"].astype(str)
        basic = basic.loc[basic["trade_date"] == str(trade_date)].copy()
        basic = basic.rename(columns={"ts_code": "stock_code"})
        for column in ["turnover_rate", "volume_ratio"]:
            if column in basic.columns:
                basic[column] = pd.to_numeric(basic[column], errors="coerce")
        data = data.merge(
            basic[["stock_code", "trade_date", "turnover_rate", "volume_ratio"]],
            on=["stock_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    else:
        data["turnover_rate"] = pd.NA
        data["volume_ratio"] = pd.NA

    if stock_basic is not None and not stock_basic.empty:
        names = stock_basic.rename(columns={"ts_code": "stock_code", "name": "stock_name"}).copy()
        data = data.merge(
            names[["stock_code", "stock_name", "industry"]],
            on="stock_code",
            how="left",
            validate="many_to_one",
        )
    else:
        data["stock_name"] = ""
        data["industry"] = ""

    data["stock_name"] = data["stock_name"].fillna(data["stock_code"]).astype(str)
    data["industry"] = data["industry"].fillna("").astype(str)
    data["concepts"] = ""
    return (
        data.sort_values(["pct_chg", "amount", "stock_code"], ascending=[False, False, True])
        .drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
        .reset_index(drop=True)
    )


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "stock_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "vol",
            "amount",
            "turnover_rate",
            "volume_ratio",
            "stock_name",
            "industry",
            "concepts",
        ]
    )


def _limit_stock_from_row(row: dict[str, object], threshold: float) -> LimitStock:
    return LimitStock(
        trade_date=str(row["trade_date"]),
        stock_code=str(row["stock_code"]),
        stock_name=str(row.get("stock_name") or row["stock_code"]),
        close=_float_or_none(row.get("close")),
        pct_chg=_float_or_none(row.get("pct_chg")),
        amount=_float_or_none(row.get("amount")),
        turnover_rate=_float_or_none(row.get("turnover_rate")),
        volume_ratio=_float_or_none(row.get("volume_ratio")),
        consecutive_limit_count=None,
        limit_reason=None,
        industry=_text_or_none(row.get("industry")),
        concepts=[],
        status="limit_up",
        source=f"tushare_daily_pct_gte_{threshold:g}",
    )


def _strong_stock_from_row(row: dict[str, object]) -> StrongStock:
    return StrongStock(
        trade_date=str(row["trade_date"]),
        stock_code=str(row["stock_code"]),
        stock_name=str(row.get("stock_name") or row["stock_code"]),
        pct_chg=float(row["pct_chg"]),
        close=_float_or_none(row.get("close")),
        amount=_float_or_none(row.get("amount")),
        turnover_rate=_float_or_none(row.get("turnover_rate")),
        volume_ratio=_float_or_none(row.get("volume_ratio")),
        industry=_text_or_none(row.get("industry")),
        concepts=[],
        source="tushare_daily",
    )


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
