"""Local CSV news provider for daily radar workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from market_ai.models import NewsItem
from market_ai.providers.news.base import NewsProvider

REQUIRED_COLUMNS = ["news_id", "source", "title", "published_at"]
OPTIONAL_COLUMNS = ["url", "content", "is_filtered", "filter_reason", "content_hash"]


class LocalCsvNewsProvider(NewsProvider):
    """Read normalized news items from a local CSV file.

    Required columns: news_id, source, title, published_at.
    Optional columns: url, content.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def fetch_news(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        sources: Iterable[str] | None = None,
    ) -> list[NewsItem]:
        """Return normalized news items published in [start_time, end_time]."""
        start_ts = _to_utc_timestamp(start_time)
        end_ts = _to_utc_timestamp(end_time)
        if start_ts > end_ts:
            raise ValueError("start_time must be earlier than or equal to end_time.")
        if not self.csv_path.exists():
            return []

        data = _normalize_news_frame(pd.read_csv(self.csv_path))
        if data.empty:
            return []

        published_at = pd.to_datetime(data["published_at"], errors="coerce", utc=True)
        keep = published_at.notna() & (published_at >= start_ts) & (published_at <= end_ts)
        if "is_filtered" in data.columns:
            keep &= ~data["is_filtered"].map(_bool_value)
        if sources is not None:
            source_set = {str(source).strip() for source in sources if str(source).strip()}
            keep &= data["source"].isin(source_set)

        filtered = data.loc[keep].copy()
        if filtered.empty:
            return []

        filtered["_published_at_ts"] = published_at.loc[keep]
        filtered = (
            filtered.sort_values(["_published_at_ts", "source", "news_id"])
            .drop_duplicates(subset=["news_id", "source"], keep="last")
            .reset_index(drop=True)
        )

        return [
            NewsItem(
                news_id=str(row.news_id),
                source=str(row.source),
                title=str(row.title),
                published_at=str(row.published_at),
                url=_optional_value(getattr(row, "url", None)),
                content=_optional_value(getattr(row, "content", None)),
            )
            for row in filtered.itertuples(index=False)
        ]


def _normalize_news_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Local news CSV missing required columns: {missing}")

    result = data.copy()
    for column in [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()

    valid = (
        result["news_id"].ne("")
        & result["source"].ne("")
        & result["title"].ne("")
        & result["published_at"].ne("")
    )
    return result.loc[valid, [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].copy()


def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _optional_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_value(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "是"}
