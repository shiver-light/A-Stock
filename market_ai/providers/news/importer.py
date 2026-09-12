"""Normalize external news and announcement CSV files for radar reports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["news_id", "source", "title", "published_at", "url", "content"]

ALIASES = {
    "id": "news_id",
    "资讯ID": "news_id",
    "来源": "source",
    "标题": "title",
    "新闻标题": "title",
    "公告标题": "title",
    "发布时间": "published_at",
    "公告时间": "published_at",
    "time": "published_at",
    "datetime": "published_at",
    "link": "url",
    "链接": "url",
    "正文": "content",
    "摘要": "content",
    "内容": "content",
}

CN_TZ = timezone(timedelta(hours=8))


def import_news_csvs(
    input_paths: list[str | Path],
    *,
    output_file: str | Path,
    start_date: str,
    end_date: str,
    lookback_hours: int = 36,
    default_source: str = "manual",
) -> pd.DataFrame:
    """Import external CSV files into the LocalCsvNewsProvider schema."""
    frames = [_read_news_csv(path, default_source=default_source) for path in input_paths]
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = _filter_by_report_window(data, start_date=start_date, end_date=end_date, lookback_hours=lookback_hours)
        data = _deduplicate_news(data)
    else:
        data = pd.DataFrame(columns=OUTPUT_COLUMNS)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def _read_news_csv(path: str | Path, *, default_source: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"News CSV not found: {csv_path}")
    data = pd.read_csv(csv_path)
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    data = _apply_aliases(data)
    for column in OUTPUT_COLUMNS:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str).str.strip()
    if not default_source:
        default_source = "manual"
    data.loc[data["source"].eq(""), "source"] = default_source
    data = data.loc[data["title"].ne("") & data["published_at"].ne("")].copy()
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    normalized_time = _normalize_published_at(data["published_at"])
    data = data.loc[normalized_time.notna()].copy()
    normalized_time = normalized_time.loc[data.index]
    data["published_at"] = normalized_time.map(lambda value: value.isoformat())
    data.loc[data["news_id"].eq(""), "news_id"] = data.loc[data["news_id"].eq("")].apply(_build_news_id, axis=1)
    return data[OUTPUT_COLUMNS].copy()


def _apply_aliases(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for source, target in ALIASES.items():
        if target not in result.columns and source in result.columns:
            result[target] = result[source]
    return result


def _normalize_published_at(values: pd.Series) -> pd.Series:
    results = []
    for raw_value in values:
        value = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(value):
            results.append(pd.NaT)
        elif value.tzinfo is None:
            results.append(value.tz_localize(CN_TZ))
        else:
            results.append(value.tz_convert(CN_TZ))
    return pd.Series(results, index=values.index)


def _filter_by_report_window(
    data: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    lookback_hours: int,
) -> pd.DataFrame:
    if data.empty:
        return data
    start = _trade_date_close_time(start_date) - timedelta(hours=int(lookback_hours))
    end = _trade_date_close_time(end_date)
    published_at = pd.to_datetime(data["published_at"], errors="coerce", utc=True).dt.tz_convert(CN_TZ)
    keep = published_at.notna() & published_at.ge(pd.Timestamp(start)) & published_at.le(pd.Timestamp(end))
    return data.loc[keep].copy()


def _deduplicate_news(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.sort_values(["published_at", "source", "news_id"])
    result = result.drop_duplicates(subset=["news_id", "source"], keep="last")
    return result.reset_index(drop=True)[OUTPUT_COLUMNS]


def _trade_date_close_time(trade_date: str) -> datetime:
    return datetime.strptime(str(trade_date), "%Y%m%d").replace(hour=16, minute=30, tzinfo=CN_TZ)


def _build_news_id(row: pd.Series) -> str:
    raw = "|".join(str(row.get(column, "")) for column in ["source", "published_at", "title", "url"])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"news_{digest}"
