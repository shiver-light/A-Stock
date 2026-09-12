"""Deterministic semantic scoring for market news events."""

from __future__ import annotations

from dataclasses import dataclass

from market_ai.models import NewsEventAnalysis, NewsItem


DIRECT_MARKET_KEYWORDS = (
    "a股",
    "沪深",
    "上市公司",
    "板块",
    "多股涨停",
    "涨停潮",
    "产业链",
    "订单",
    "涨价",
    "扩产",
    "政策",
    "工信部",
    "发改委",
    "证监会",
    "交易所",
)
IMPACT_KEYWORDS = (
    "政策",
    "规划",
    "意见",
    "通知",
    "发布",
    "签订",
    "中标",
    "订单",
    "涨价",
    "扩产",
    "突破",
    "首发",
    "放量",
    "多股涨停",
)
LOW_DIRECTNESS_KEYWORDS = (
    "美股",
    "港股",
    "特朗普",
    "海外",
    "环球市场",
    "苹果发布会",
    "meta",
    "openai",
)


@dataclass(frozen=True)
class NewsSemanticScore:
    """Deterministic semantic score with auditable components."""

    directness_score: float
    impact_score: float
    reason: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "directness_score": self.directness_score,
            "impact_score": self.impact_score,
            "reason": self.reason,
        }


def calculate_news_semantic_score(event: NewsEventAnalysis, news_item: NewsItem | None = None) -> NewsSemanticScore:
    """Score whether a news event is directly useful for A-share theme radar."""
    text = _event_text(event, news_item)
    direct_hits = [keyword for keyword in DIRECT_MARKET_KEYWORDS if keyword in text]
    impact_hits = [keyword for keyword in IMPACT_KEYWORDS if keyword in text]
    low_direct_hits = [keyword for keyword in LOW_DIRECTNESS_KEYWORDS if keyword in text]

    directness = min(100.0, 35.0 + 15.0 * len(direct_hits))
    if low_direct_hits and not direct_hits:
        directness = 20.0
    elif low_direct_hits:
        directness = max(35.0, directness - 25.0)

    impact = min(100.0, 30.0 + 12.0 * len(impact_hits) + 8.0 * max(0, len(event.themes) - 1))
    reason = []
    if direct_hits:
        reason.append("direct_market_terms=" + ",".join(direct_hits[:5]))
    if impact_hits:
        reason.append("impact_terms=" + ",".join(impact_hits[:5]))
    if low_direct_hits:
        reason.append("low_directness_terms=" + ",".join(low_direct_hits[:5]))
    return NewsSemanticScore(
        directness_score=round(directness, 6),
        impact_score=round(impact, 6),
        reason=reason,
    )


def _event_text(event: NewsEventAnalysis, news_item: NewsItem | None) -> str:
    parts = [event.event, " ".join(event.themes)]
    if news_item is not None:
        parts.extend([news_item.title, news_item.content or ""])
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
