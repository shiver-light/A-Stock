"""Select core news events for daily market radar reports."""

from __future__ import annotations

from dataclasses import dataclass

from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.scoring.news_authority import calculate_authority_score


@dataclass(frozen=True)
class CoreNewsEventScore:
    """Auditable score for selecting report-level core news."""

    event: NewsEventAnalysis
    score: float
    authority_score: float
    theme_match_score: float
    core_theme_score: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "score": self.score,
            "authority_score": self.authority_score,
            "theme_match_score": self.theme_match_score,
            "core_theme_score": self.core_theme_score,
            "source": self.source,
        }


def rank_core_news_events(
    events: list[NewsEventAnalysis],
    *,
    news_items: list[NewsItem],
    core_themes: list[str],
    authority_scores: dict[str, float],
    default_authority_score: float = 50.0,
    require_core_theme_match: bool = True,
) -> list[CoreNewsEventScore]:
    """Rank normalized news events by authority and overlap with core themes."""
    source_by_id = {item.news_id: item.source for item in news_items}
    core_theme_set = {str(theme).strip() for theme in core_themes if str(theme).strip()}
    results = []
    for event in events:
        event_themes = {str(theme).strip() for theme in event.themes if str(theme).strip()}
        overlap = event_themes & core_theme_set
        if require_core_theme_match and not overlap:
            continue
        source = _event_source(event, source_by_id)
        authority = calculate_authority_score(
            source,
            authority_scores=authority_scores,
            default_score=default_authority_score,
        )
        theme_match_score = min(100.0, 35.0 * len(event_themes))
        core_theme_score = 100.0 if overlap else 0.0
        score = 0.45 * authority.score + 0.35 * core_theme_score + 0.20 * theme_match_score
        results.append(
            CoreNewsEventScore(
                event=event,
                score=round(score, 6),
                authority_score=round(authority.score, 6),
                theme_match_score=round(theme_match_score, 6),
                core_theme_score=round(core_theme_score, 6),
                source=source,
            )
        )
    return sorted(results, key=lambda item: (-item.score, item.event.event_time, item.event.event))


def select_core_news_events(
    events: list[NewsEventAnalysis],
    *,
    news_items: list[NewsItem],
    core_themes: list[str],
    authority_scores: dict[str, float],
    default_authority_score: float = 50.0,
    min_score: float = 60.0,
    top_n: int = 10,
    require_core_theme_match: bool = True,
) -> list[NewsEventAnalysis]:
    """Return the report-level top core news events."""
    ranked = rank_core_news_events(
        events,
        news_items=news_items,
        core_themes=core_themes,
        authority_scores=authority_scores,
        default_authority_score=default_authority_score,
        require_core_theme_match=require_core_theme_match,
    )
    selected = [item.event for item in ranked if item.score >= float(min_score)]
    return selected[: max(0, int(top_n))]


def _event_source(event: NewsEventAnalysis, source_by_id: dict[str, str]) -> str:
    for news_id in event.source_news_ids:
        source = str(source_by_id.get(news_id) or "").strip()
        if source:
            return source
    return ""
