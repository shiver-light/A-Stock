"""Deterministic event type classification for market news."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

from market_ai.models import NewsEventAnalysis, NewsItem


EVENT_TYPE_SCORES = {
    "policy": 100.0,
    "price_hike": 92.0,
    "order": 88.0,
    "capacity_expansion": 85.0,
    "industry_research": 70.0,
    "earnings": 55.0,
    "overseas": 35.0,
    "market_summary": 25.0,
    "single_stock": 20.0,
    "other": 45.0,
}

EVENT_TYPE_KEYWORDS = {
    "policy": (
        "政策",
        "规划",
        "意见",
        "通知",
        "方案",
        "工信部",
        "发改委",
        "证监会",
        "交易所",
        "国务院",
    ),
    "price_hike": ("涨价", "提价", "报价上调", "价格上调", "价格上涨", "涨幅", "供需偏紧"),
    "order": ("订单", "中标", "签订", "采购", "合同", "交付"),
    "capacity_expansion": ("扩产", "投产", "产能", "新产线", "开工", "建设项目"),
    "industry_research": ("研报", "证券", "投资逻辑", "行业", "产业链", "板块", "专题"),
    "earnings": ("业绩", "利润", "营收", "净利", "预增", "预减", "财报"),
    "overseas": ("美股", "港股", "海外", "特朗普", "英伟达", "nvidia", "meta", "openai"),
    "market_summary": ("早报", "午报", "收评", "收盘", "复盘", "研选日报", "环球市场"),
    "single_stock": ("*st", "st", "控制权", "董事长", "实控人", "股东", "刑事立案", "问询函", "监管函"),
}


@dataclass(frozen=True)
class NewsEventTypeResult:
    """Auditable deterministic event type classification result."""

    primary_type: str
    tags: list[str]
    score: float
    reason: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_type": self.primary_type,
            "tags": self.tags,
            "score": self.score,
            "reason": self.reason,
        }


def classify_news_event_type(event: NewsEventAnalysis, news_item: NewsItem | None = None) -> NewsEventTypeResult:
    """Classify one news event into a stable rule-based event type."""
    text = _event_text(event, news_item)
    hits_by_type = {
        event_type: [keyword for keyword in keywords if keyword in text]
        for event_type, keywords in EVENT_TYPE_KEYWORDS.items()
    }
    tags = [event_type for event_type, hits in hits_by_type.items() if hits]
    if tags:
        primary_type = max(tags, key=lambda event_type: EVENT_TYPE_SCORES[event_type])
    else:
        primary_type = "other"

    reason = []
    for event_type in tags:
        reason.append(f"{event_type}=" + ",".join(hits_by_type[event_type][:5]))
    return NewsEventTypeResult(
        primary_type=primary_type,
        tags=tags,
        score=round(EVENT_TYPE_SCORES[primary_type], 6),
        reason=reason,
    )


def apply_news_event_types(
    events: list[NewsEventAnalysis],
    *,
    news_items: list[NewsItem],
) -> list[NewsEventAnalysis]:
    """Return cloned events with deterministic event_type filled in."""
    news_by_id = {item.news_id: item for item in news_items}
    results = []
    for event in events:
        news_item = _event_news_item(event, news_by_id)
        event_type = classify_news_event_type(event, news_item)
        cloned = copy(event)
        cloned.event_type = event_type.primary_type
        results.append(cloned)
    return results


def event_type_score(event_type: str) -> float:
    """Return deterministic quality score for a classified event type."""
    return EVENT_TYPE_SCORES.get(str(event_type).strip(), EVENT_TYPE_SCORES["other"])


def _event_news_item(event: NewsEventAnalysis, news_by_id: dict[str, NewsItem]) -> NewsItem | None:
    for news_id in event.source_news_ids:
        item = news_by_id.get(news_id)
        if item is not None:
            return item
    return None


def _event_text(event: NewsEventAnalysis, news_item: NewsItem | None) -> str:
    parts = [event.event, event.event_type, " ".join(event.themes)]
    if news_item is not None:
        parts.extend([news_item.title, news_item.content or ""])
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
