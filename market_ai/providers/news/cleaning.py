"""Deterministic news cleaning utilities for market radar workflows."""

from __future__ import annotations

import hashlib
import re

from market_ai.models import NormalizedNews, RawNews


QUOTE_BROADCAST_PATTERNS = [
    re.compile(r"^(.*?)(涨|跌)(超|逾)?\d+(\.\d+)?%$"),
    re.compile(r"^(.*?)(涨停|跌停|触及涨停|封涨停)$"),
    re.compile(r"^(.*?)(走高|走低|拉升|跳水)$"),
]


def clean_news_item(item: RawNews) -> NormalizedNews:
    """Clean one raw news item and keep low-value filters as audit fields."""
    normalized_title = normalize_news_text(item.title)
    normalized_content = normalize_news_text(item.content)
    filter_reason = detect_low_value_news_reason(normalized_title, normalized_content)
    return NormalizedNews(
        news_id=item.news_id,
        source=item.source,
        title=item.title,
        published_at=item.published_at,
        normalized_title=normalized_title,
        normalized_content=normalized_content,
        content_hash=content_hash(normalized_title, normalized_content),
        url=item.url,
        content=item.content,
        is_filtered=bool(filter_reason),
        filter_reason=filter_reason,
        source_news_ids=[item.news_id],
        sources=[item.source],
    )


def clean_news_items(items: list[RawNews]) -> list[NormalizedNews]:
    """Clean raw news rows without dropping filtered rows."""
    return [clean_news_item(item) for item in items]


def normalize_news_text(value: object) -> str:
    """Normalize title or content text for deterministic matching."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u3000", " ")
    return text.strip()


def content_hash(normalized_title: str, normalized_content: str) -> str:
    """Return a stable hash for normalized title and content."""
    raw = f"{normalized_title}|{normalized_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def detect_low_value_news_reason(normalized_title: str, normalized_content: str = "") -> str:
    """Return a filter reason for pure quote broadcasts, otherwise empty."""
    text = normalize_news_text(normalized_title)
    if not text:
        return "empty_title"
    if len(text) <= 24 and any(pattern.match(text) for pattern in QUOTE_BROADCAST_PATTERNS):
        return "quote_broadcast"
    if not normalized_content and re.search(r"(涨|跌)(超|逾)?\d+(\.\d+)?%", text) and len(text) <= 18:
        return "quote_broadcast"
    return ""
