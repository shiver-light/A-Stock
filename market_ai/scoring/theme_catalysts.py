"""Rule-based aggregation of news catalysts by core theme."""

from __future__ import annotations

from market_ai.models import NewsEventAnalysis, ThemeCatalystSummary, ThemeScoreResult
from market_ai.scoring.news_confirmation import CONFIRMED_CATALYST


def aggregate_theme_catalysts(
    *,
    trade_date: str,
    core_themes: list[ThemeScoreResult],
    news_events: list[NewsEventAnalysis],
    max_events_per_theme: int = 3,
) -> list[ThemeCatalystSummary]:
    """Aggregate selected news events into per-theme catalyst summaries."""
    summaries = []
    for theme in sorted(core_themes, key=lambda item: (item.rank or 9999, -item.score, item.theme)):
        related = [event for event in news_events if theme.theme in event.themes]
        if not related:
            summaries.append(
                ThemeCatalystSummary(
                    trade_date=trade_date,
                    theme=theme.theme,
                    conclusion="暂无核心消息直接验证，更多依赖资金行为。",
                )
            )
            continue

        related = sorted(
            related,
            key=lambda event: (
                0 if event.validation_state == CONFIRMED_CATALYST else 1,
                -(event.market_confirm_score or 0.0),
                event.event_time,
                event.event,
            ),
        )
        confirmed_count = sum(1 for event in related if event.validation_state == CONFIRMED_CATALYST)
        unconfirmed_count = len(related) - confirmed_count
        primary_event = related[0].event
        evidence = [event.event for event in related[: max(1, int(max_events_per_theme))]]
        summaries.append(
            ThemeCatalystSummary(
                trade_date=trade_date,
                theme=theme.theme,
                primary_event=primary_event,
                confirmed_event_count=confirmed_count,
                unconfirmed_event_count=unconfirmed_count,
                related_event_count=len(related),
                evidence_events=evidence,
                conclusion=_conclusion(confirmed_count, unconfirmed_count),
            )
        )
    return summaries


def _conclusion(confirmed_count: int, unconfirmed_count: int) -> str:
    if confirmed_count > 0:
        return f"消息与资金形成确认，已确认事件 {confirmed_count} 条。"
    if unconfirmed_count > 0:
        return "有相关消息，但尚未形成强资金确认。"
    return "暂无可验证消息。"
