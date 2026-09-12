"""Deterministic news deduplication for market radar workflows."""

from __future__ import annotations

import hashlib

from market_ai.models import NormalizedNews


def deduplicate_news_items(items: list[NormalizedNews]) -> list[NormalizedNews]:
    """Deduplicate cleaned news by URL, content hash, then normalized title."""
    groups: dict[str, list[NormalizedNews]] = {}
    for item in items:
        groups.setdefault(_dedup_key(item), []).append(item)

    results = []
    for group in groups.values():
        representative = _choose_representative(group)
        source_news_ids = []
        sources = []
        for item in group:
            source_news_ids.extend(item.source_news_ids)
            sources.extend(item.sources)
        representative.source_news_ids = list(dict.fromkeys(source_news_ids))
        representative.sources = list(dict.fromkeys(sources))
        results.append(representative)
    return sorted(results, key=lambda item: (item.published_at, item.source, item.news_id))


def _dedup_key(item: NormalizedNews) -> str:
    url = str(item.url or "").strip().lower()
    if url:
        return f"url:{url}"
    content_hash = str(item.content_hash or "").strip()
    if content_hash:
        return f"hash:{content_hash}"
    title = str(item.normalized_title or "").strip().lower()
    return "title:" + hashlib.sha1(title.encode("utf-8")).hexdigest()


def _choose_representative(group: list[NormalizedNews]) -> NormalizedNews:
    return sorted(
        group,
        key=lambda item: (
            bool(item.is_filtered),
            -len(str(item.normalized_content or "")),
            item.published_at,
            item.source,
            item.news_id,
        ),
    )[0]
