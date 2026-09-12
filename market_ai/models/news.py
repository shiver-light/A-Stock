"""News and event models for market radar research."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from market_ai.models.core import JsonModel


def _require_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _optional_text(value: object) -> str:
    return str(value or "").strip()


def _require_datetime(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if "T" not in text and " " not in text:
        raise ValueError(f"{field_name} must be an ISO datetime string.")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO datetime string.") from exc
    return text


def _text_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _confidence(value: object) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return number


@dataclass
class RawNews(JsonModel):
    """Auditable raw news row before cleaning and deduplication."""

    news_id: str
    source: str
    title: str
    published_at: str
    url: str = ""
    content: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.news_id = _require_text(self.news_id, "news_id")
        self.source = _require_text(self.source, "source")
        self.title = _require_text(self.title, "title")
        self.published_at = _require_datetime(self.published_at, "published_at")
        self.url = _optional_text(self.url)
        self.content = _optional_text(self.content)
        self.raw_payload = dict(self.raw_payload or {})


@dataclass
class NormalizedNews(JsonModel):
    """Cleaned news row with deterministic audit fields."""

    news_id: str
    source: str
    title: str
    published_at: str
    normalized_title: str
    normalized_content: str = ""
    content_hash: str = ""
    url: str = ""
    content: str = ""
    is_filtered: bool = False
    filter_reason: str = ""
    source_news_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.news_id = _require_text(self.news_id, "news_id")
        self.source = _require_text(self.source, "source")
        self.title = _require_text(self.title, "title")
        self.published_at = _require_datetime(self.published_at, "published_at")
        self.normalized_title = _require_text(self.normalized_title, "normalized_title")
        self.normalized_content = _optional_text(self.normalized_content)
        self.content_hash = _optional_text(self.content_hash)
        self.url = _optional_text(self.url)
        self.content = _optional_text(self.content)
        self.is_filtered = bool(self.is_filtered)
        self.filter_reason = _optional_text(self.filter_reason)
        self.source_news_ids = _text_list(self.source_news_ids) or [self.news_id]
        self.sources = _text_list(self.sources) or [self.source]


@dataclass
class EventCluster(JsonModel):
    """Deterministic cluster of normalized news rows for one market event."""

    cluster_id: str
    event_time: str
    title: str
    source_news_ids: list[str]
    sources: list[str] = field(default_factory=list)
    representative_news_id: str = ""

    def __post_init__(self) -> None:
        self.cluster_id = _require_text(self.cluster_id, "cluster_id")
        self.event_time = _require_datetime(self.event_time, "event_time")
        self.title = _require_text(self.title, "title")
        self.source_news_ids = _text_list(self.source_news_ids)
        if not self.source_news_ids:
            raise ValueError("source_news_ids must not be empty.")
        self.sources = _text_list(self.sources)
        self.representative_news_id = _optional_text(self.representative_news_id)


@dataclass
class MarketEvent(JsonModel):
    """Structured market event produced from an event cluster."""

    event_id: str
    event_time: str
    event_summary: str
    event_type: str
    themes: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    direction: str = ""
    scope: str = ""
    confidence: float = 0.0
    source_cluster_id: str = ""

    def __post_init__(self) -> None:
        self.event_id = _require_text(self.event_id, "event_id")
        self.event_time = _require_datetime(self.event_time, "event_time")
        self.event_summary = _require_text(self.event_summary, "event_summary")
        self.event_type = _require_text(self.event_type, "event_type")
        self.themes = _text_list(self.themes)
        self.related_entities = _text_list(self.related_entities)
        self.direction = _optional_text(self.direction)
        self.scope = _optional_text(self.scope)
        self.confidence = _confidence(self.confidence)
        self.source_cluster_id = _optional_text(self.source_cluster_id)
