"""Tushare major_news provider for market radar news collection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.tushare_client import TushareClient
from market_ai.models import NewsItem
from market_ai.providers.news.base import NewsProvider


CN_TZ = timezone(timedelta(hours=8))
FIELDS = "pub_time,title,content,src"


class TushareMajorNewsProvider(NewsProvider):
    """Fetch major market news from Tushare major_news with local daily cache."""

    def __init__(
        self,
        *,
        client: TushareClient | None = None,
        cache_dir: str | Path = "data/cache/market_ai/news/tushare_major_news",
        refresh: bool = False,
        slice_hours: int = 6,
    ) -> None:
        self.client = client or TushareClient()
        self.cache_dir = Path(cache_dir)
        self.refresh = bool(refresh)
        self.slice_hours = max(1, int(slice_hours))

    def fetch_news(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        sources: Iterable[str] | None = None,
    ) -> list[NewsItem]:
        """Return normalized Tushare major_news items in [start_time, end_time]."""
        start_ts = _to_cn_timestamp(start_time)
        end_ts = _to_cn_timestamp(end_time)
        if start_ts > end_ts:
            raise ValueError("start_time must be earlier than or equal to end_time.")

        data = self.fetch_news_frame(start_time=start_ts.to_pydatetime(), end_time=end_ts.to_pydatetime())
        if data.empty:
            return []
        if sources is not None:
            source_set = {str(source).strip() for source in sources if str(source).strip()}
            if source_set:
                data = data.loc[data["source"].isin(source_set)].copy()
        if data.empty:
            return []
        return [
            NewsItem(
                news_id=str(row.news_id),
                source=str(row.source),
                title=str(row.title),
                published_at=str(row.published_at),
                content=_optional_value(row.content),
            )
            for row in data.itertuples(index=False)
        ]

    def fetch_news_frame(self, *, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch normalized news as a DataFrame for CSV export."""
        start_ts = _to_cn_timestamp(start_time)
        end_ts = _to_cn_timestamp(end_time)
        frames = []
        for slice_start, slice_end in _iter_time_slices(start_ts, end_ts, self.slice_hours):
            frames.append(self._fetch_slice(slice_start=slice_start, slice_end=slice_end))
        if not frames:
            return pd.DataFrame(columns=["news_id", "source", "title", "published_at", "url", "content"])
        data = pd.concat(frames, ignore_index=True)
        data = _normalize_major_news(data)
        if data.empty:
            return data
        published_at = pd.to_datetime(data["published_at"], errors="coerce", utc=True).dt.tz_convert(CN_TZ)
        keep = published_at.notna() & published_at.ge(start_ts) & published_at.le(end_ts)
        data = data.loc[keep].copy()
        data["_published_at"] = published_at.loc[keep]
        data = (
            data.sort_values(["_published_at", "source", "news_id"])
            .drop_duplicates(subset=["news_id", "source"], keep="last")
            .drop(columns=["_published_at"])
            .reset_index(drop=True)
        )
        return data

    def _fetch_slice(self, *, slice_start: pd.Timestamp, slice_end: pd.Timestamp) -> pd.DataFrame:
        cache_path = self._cache_path(slice_start=slice_start, slice_end=slice_end)
        if cache_path.exists() and not self.refresh:
            return pd.read_csv(cache_path)

        data = self.client.major_news(
            start_date=slice_start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=slice_end.strftime("%Y-%m-%d %H:%M:%S"),
            fields=FIELDS,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, index=False)
        return data

    def _cache_path(self, *, slice_start: pd.Timestamp, slice_end: pd.Timestamp) -> Path:
        filename = f"{slice_start.strftime('%Y%m%d_%H%M%S')}_{slice_end.strftime('%Y%m%d_%H%M%S')}.csv"
        return self.cache_dir / filename


def _normalize_major_news(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["news_id", "source", "title", "published_at", "url", "content"]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)
    result = data.copy()
    rename_map = {"src": "source", "pub_time": "published_at"}
    result = result.rename(columns={key: value for key, value in rename_map.items() if key in result.columns})
    for column in columns:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    result = result.loc[result["source"].ne("") & result["title"].ne("") & result["published_at"].ne("")].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)

    parsed = pd.Series([pd.to_datetime(value, errors="coerce") for value in result["published_at"]], index=result.index)
    keep = parsed.notna()
    result = result.loc[keep].copy()
    parsed = parsed.loc[keep]
    normalized_times = []
    for value in parsed:
        if value.tzinfo is None:
            normalized_times.append(value.tz_localize(CN_TZ))
        else:
            normalized_times.append(value.tz_convert(CN_TZ))
    result["published_at"] = [value.isoformat() for value in normalized_times]
    result.loc[result["news_id"].eq(""), "news_id"] = result.loc[result["news_id"].eq("")].apply(
        lambda row: _news_id(row["source"], row["published_at"], row["title"], row["content"]),
        axis=1,
    )
    return result[columns]


def _iter_time_slices(start: pd.Timestamp, end: pd.Timestamp, slice_hours: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    slices = []
    cursor = start
    while cursor <= end:
        slice_end = min(cursor + timedelta(hours=slice_hours), end)
        slices.append((cursor, slice_end))
        cursor = slice_end + timedelta(seconds=1)
    return slices


def _to_cn_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(CN_TZ)
    return timestamp.tz_convert(CN_TZ)


def _news_id(source: str, published_at: str, title: str, content: str) -> str:
    import hashlib

    raw = "|".join([source, published_at, title, content])
    return f"tushare_major_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _optional_value(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None
