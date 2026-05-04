"""Stock universe related data access wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig

STOCK_BASIC_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
    "is_hs",
]

INDEX_WEIGHT_FIELDS = ["index_code", "con_code", "trade_date", "weight"]
INDEX_CODE_MAP = {
    "hs300": "399300.SZ",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
    "zz2000": "932000.CSI",
    "sse50": "000016.SH",
}


@dataclass(frozen=True)
class IndexConstituentRequest:
    index_code: str
    as_of_date: str
    refresh: bool = False


class StockUniverseDataService:
    """Minimal cached wrappers for stock_basic and index_weight."""

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stock_basic_history(self, *, refresh: bool = False) -> pd.DataFrame:
        cache_path = self.cache_dir / "stock_basic" / "all_status.parquet"
        cached = self._read_cache(cache_path, date_columns=("list_date", "delist_date"))
        if not refresh and not cached.empty:
            return cached.copy()

        frames: list[pd.DataFrame] = []
        for list_status in ["L", "D", "P"]:
            fetched = self.client.stock_basic(
                list_status=list_status,
                fields=",".join(STOCK_BASIC_FIELDS),
            )
            frames.append(self._normalize_stock_basic(fetched))

        merged = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ts_code", "list_status"])
            .drop_duplicates(subset=["ts_code"], keep="last")
            .reset_index(drop=True)
        )
        self._write_cache(cache_path, merged)
        return merged.copy()

    def get_index_constituents(self, request: IndexConstituentRequest) -> pd.DataFrame:
        cache_path = self.cache_dir / "index_weight" / f"{request.index_code}.parquet"
        cached = self._read_cache(cache_path, date_columns=("trade_date",))

        current_month_start = request.as_of_date[:6] + "01"
        previous_month_start = (pd.Timestamp(current_month_start) - pd.offsets.MonthBegin(1)).strftime("%Y%m%d")
        current_month_end = (pd.Timestamp(current_month_start) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
        need_fetch = request.refresh or cached.empty or not self._cache_covers_window(
            cached,
            previous_month_start,
            current_month_end,
        )

        if need_fetch:
            fetched = self.client.index_weight(
                index_code=request.index_code,
                start_date=previous_month_start,
                end_date=current_month_end,
            )
            fetched = self._normalize_index_weight(fetched)
            combined = self._merge_index_weight(cached, fetched)
            self._write_cache(cache_path, combined)
            cached = combined

        historical = cached.loc[cached["trade_date"] <= request.as_of_date].copy()
        if historical.empty:
            return pd.DataFrame(columns=["as_of_date", "ts_code", "universe_name", "weight", "in_universe"])

        latest_trade_date = str(historical["trade_date"].max())
        result = historical.loc[historical["trade_date"] == latest_trade_date, ["con_code", "weight"]].copy()
        result = result.rename(columns={"con_code": "ts_code"})
        result["as_of_date"] = request.as_of_date
        result["in_universe"] = True
        return result.loc[:, ["as_of_date", "ts_code", "weight", "in_universe"]].reset_index(drop=True)

    def _cache_covers_window(self, cached: pd.DataFrame, start_date: str, end_date: str) -> bool:
        if cached.empty or "trade_date" not in cached.columns:
            return False
        window_data = cached.loc[(cached["trade_date"] >= start_date) & (cached["trade_date"] <= end_date)]
        return not window_data.empty

    def _normalize_stock_basic(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=STOCK_BASIC_FIELDS)
        normalized = data.copy()
        normalized["list_date"] = normalized["list_date"].fillna("").astype(str)
        normalized["delist_date"] = normalized["delist_date"].fillna("").astype(str)
        return (
            normalized[STOCK_BASIC_FIELDS]
            .sort_values(["ts_code", "list_date"])
            .drop_duplicates(subset=["ts_code"], keep="last")
            .reset_index(drop=True)
        )

    def _normalize_index_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=INDEX_WEIGHT_FIELDS)
        normalized = data.copy()
        normalized["trade_date"] = normalized["trade_date"].astype(str)
        return (
            normalized[INDEX_WEIGHT_FIELDS]
            .sort_values(["trade_date", "con_code"])
            .drop_duplicates(subset=["index_code", "con_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _merge_index_weight(self, cached: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
        valid_frames = [frame for frame in [cached, fetched] if frame is not None and not frame.empty]
        if not valid_frames:
            return pd.DataFrame(columns=INDEX_WEIGHT_FIELDS)
        return (
            pd.concat(valid_frames, ignore_index=True)
            .sort_values(["trade_date", "con_code"])
            .drop_duplicates(subset=["index_code", "con_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _read_cache(self, path: Path, date_columns: tuple[str, ...]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        data = pd.read_parquet(path)
        for column in date_columns:
            if column in data.columns:
                data[column] = data[column].fillna("").astype(str)
        return data

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)


def get_stock_basic_history(*, refresh: bool = False, cache_dir: str | Path = "data/cache/tushare") -> pd.DataFrame:
    service = StockUniverseDataService(cache_dir=cache_dir)
    return service.get_stock_basic_history(refresh=refresh)


def get_index_constituents(
    *,
    index_code: str,
    as_of_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    service = StockUniverseDataService(cache_dir=cache_dir)
    request = IndexConstituentRequest(index_code=index_code, as_of_date=as_of_date, refresh=refresh)
    return service.get_index_constituents(request)
