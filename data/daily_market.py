"""Unified A-share daily and adjusted market data service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig
from data.tushare_specs import TUSHARE_A_STOCK_DAILY_SPECS

BASE_PRICE_FIELDS = ["open", "high", "low", "close", "pre_close", "change"]
DEFAULT_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
MINIMAL_DAILY_FIELDS = [
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
]


@dataclass(frozen=True)
class DailyMarketRequest:
    ts_code: str
    start_date: str
    end_date: str
    adjust: str | None = None
    fields: tuple[str, ...] = tuple(DEFAULT_FIELDS)
    refresh: bool = False


class AShareDailyMarketService:
    """Daily market data service for A-share research code.

    Time alignment:
    - Data is daily close data for trade_date T.
    - Strategy execution timing is not changed here; callers should keep T close signal,
      T+1 execution assumptions explicitly in research and backtests.
    """

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_daily(self, request: DailyMarketRequest) -> pd.DataFrame:
        self._validate_request(request)
        raw = self._load_or_update_base_cache(request.ts_code, request.start_date, request.end_date, request.refresh)
        data = self._slice_dates(raw, request.start_date, request.end_date)

        if request.adjust in {"qfq", "hfq"}:
            factors = self._load_or_update_factor_cache(
                request.ts_code,
                request.start_date,
                request.end_date,
                request.refresh,
            )
            data = self._apply_adjustment(data, factors, request.adjust, request.end_date)
        elif request.adjust not in {None, ""}:
            raise ValueError("adjust must be one of None, 'qfq', or 'hfq'.")

        return self._select_fields(data, request.fields)

    def get_api_specs(self) -> dict[str, dict[str, object]]:
        return TUSHARE_A_STOCK_DAILY_SPECS

    def fetch_pro_bar(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
        adjust: str,
        fields: Iterable[str] = DEFAULT_FIELDS,
    ) -> pd.DataFrame:
        """Direct SDK wrapper for parity checks against official dynamic adjusted data."""
        data = self.client.pro_bar(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj=adjust,
            freq="D",
            asset="E",
        )
        return self._select_fields(self._normalize_frame(data), tuple(fields))

    def _validate_request(self, request: DailyMarketRequest) -> None:
        if not request.ts_code:
            raise ValueError("ts_code is required.")
        if not request.start_date or not request.end_date:
            raise ValueError("start_date and end_date are required.")
        if request.start_date > request.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")

    def _base_cache_path(self, ts_code: str) -> Path:
        return self.cache_dir / "daily" / f"{ts_code}.parquet"

    def _factor_cache_path(self, ts_code: str) -> Path:
        return self.cache_dir / "adj_factor" / f"{ts_code}.parquet"

    def _load_or_update_base_cache(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool,
    ) -> pd.DataFrame:
        cache_path = self._base_cache_path(ts_code)
        cached = self._read_cache(cache_path)
        missing_ranges = self._compute_missing_ranges(cached, start_date, end_date, refresh)

        frames = [cached] if not cached.empty else []
        for missing_start, missing_end in missing_ranges:
            fetched = self.client.daily(
                ts_code=ts_code,
                start_date=missing_start,
                end_date=missing_end,
                fields=",".join(DEFAULT_FIELDS),
            )
            frames.append(self._normalize_frame(fetched))

        merged = self._merge_frames(frames)
        self._write_cache(cache_path, merged)
        return merged

    def _load_or_update_factor_cache(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool,
    ) -> pd.DataFrame:
        cache_path = self._factor_cache_path(ts_code)
        cached = self._read_cache(cache_path)
        missing_ranges = self._compute_missing_ranges(cached, start_date, end_date, refresh)

        frames = [cached] if not cached.empty else []
        for missing_start, missing_end in missing_ranges:
            fetched = self.client.adj_factor(
                ts_code=ts_code,
                start_date=missing_start,
                end_date=missing_end,
            )
            frames.append(self._normalize_frame(fetched))

        merged = self._merge_frames(frames)
        self._write_cache(cache_path, merged)
        return merged

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

    def _apply_adjustment(
        self,
        daily: pd.DataFrame,
        factors: pd.DataFrame,
        adjust: str,
        anchor_end_date: str,
    ) -> pd.DataFrame:
        merged = daily.merge(
            factors[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        if merged["adj_factor"].isna().any():
            raise ValueError("Missing adj_factor rows after merge; cannot compute adjusted prices.")

        anchor_rows = merged.loc[merged["trade_date"] <= anchor_end_date, ["trade_date", "adj_factor"]]
        if anchor_rows.empty:
            raise ValueError("No adj_factor available on or before end_date.")
        anchor_factor = float(anchor_rows.sort_values("trade_date").iloc[-1]["adj_factor"])

        result = merged.copy()
        if adjust == "qfq":
            scale = result["adj_factor"] / anchor_factor
        else:
            scale = result["adj_factor"]

        for column in BASE_PRICE_FIELDS:
            result[column] = result[column] * scale

        return result

    def _slice_dates(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
        return data.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def _normalize_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        normalized = data.copy()
        if "trade_date" in normalized.columns:
            normalized["trade_date"] = normalized["trade_date"].astype(str)
        return normalized.sort_values(["trade_date", "ts_code"]).drop_duplicates().reset_index(drop=True)

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

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._normalize_frame(pd.read_parquet(path))

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)

    def _select_fields(self, data: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
        missing = [field for field in fields if field not in data.columns]
        if missing:
            raise ValueError(f"Requested fields are unavailable: {missing}")
        return data.loc[:, list(fields)].copy()


def get_a_share_daily_prices(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    """Fetch unified A-share daily market data with a minimal research schema.

    Tushare source:
    - API: daily
    - Doc: https://tushare.pro/document/2?doc_id=27
    - Permission: 120 points and above
    - Limit: 6000 rows per call, about 500 calls/min for base users

    Notes
    -----
    - This function returns unadjusted daily bars.
    - Output is sorted by trade_date, ts_code.
    - Data is trade_date close data and does not change T close / T+1 execution assumptions.
    """

    service = AShareDailyMarketService(cache_dir=cache_dir)
    request = DailyMarketRequest(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields=tuple(MINIMAL_DAILY_FIELDS),
        refresh=refresh,
    )
    return service.get_daily(request)
