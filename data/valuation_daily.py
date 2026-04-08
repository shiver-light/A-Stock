"""Daily valuation and turnover data for A-share stocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig

VALUATION_FIELDS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "pe_ttm",
    "pb",
]


@dataclass(frozen=True)
class DailyValuationRequest:
    ts_code: str
    start_date: str
    end_date: str
    fields: tuple[str, ...] = tuple(VALUATION_FIELDS)
    refresh: bool = False


class AShareDailyValuationService:
    """Minimal cached wrapper for daily_basic."""

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_daily_valuation(self, request: DailyValuationRequest) -> pd.DataFrame:
        self._validate_request(request)
        cache_path = self.cache_dir / "daily_basic" / f"{request.ts_code}.parquet"
        cached = self._read_cache(cache_path)
        missing_ranges = self._compute_missing_ranges(cached, request.start_date, request.end_date, request.refresh)

        frames = [cached] if not cached.empty else []
        for start_date, end_date in missing_ranges:
            fetched = self.client.daily_basic(
                ts_code=request.ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(VALUATION_FIELDS),
            )
            frames.append(self._normalize_frame(fetched))

        merged = self._merge_frames(frames)
        self._write_cache(cache_path, merged)
        result = self._slice_dates(merged, request.start_date, request.end_date)
        return self._select_fields(result, request.fields)

    def _validate_request(self, request: DailyValuationRequest) -> None:
        if not request.ts_code:
            raise ValueError("ts_code is required.")
        if request.start_date > request.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")

    def _compute_missing_ranges(
        self,
        cached: pd.DataFrame,
        start_date: str,
        end_date: str,
        refresh: bool,
    ) -> list[tuple[str, str]]:
        if refresh or cached.empty:
            return [(start_date, end_date)]

        ranges: list[tuple[str, str]] = []
        min_cached = str(cached["trade_date"].min())
        max_cached = str(cached["trade_date"].max())
        if start_date < min_cached:
            ranges.append((start_date, min_cached))
        if end_date > max_cached:
            ranges.append((max_cached, end_date))
        return ranges

    def _normalize_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        normalized = data.copy()
        normalized["trade_date"] = normalized["trade_date"].astype(str)
        return (
            normalized.sort_values(["trade_date", "ts_code"])
            .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _merge_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        return (
            pd.concat(valid_frames, ignore_index=True)
            .sort_values(["trade_date", "ts_code"])
            .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _slice_dates(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
        return data.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def _select_fields(self, data: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
        missing = [field for field in fields if field not in data.columns]
        if missing:
            raise ValueError(f"Requested fields are unavailable: {missing}")
        return data.loc[:, list(fields)].copy()

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._normalize_frame(pd.read_parquet(path))

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)


def get_a_share_daily_valuation(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    service = AShareDailyValuationService(cache_dir=cache_dir)
    request = DailyValuationRequest(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return service.get_daily_valuation(request)
