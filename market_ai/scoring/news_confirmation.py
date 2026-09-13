"""Market confirmation classification for normalized news events."""

from __future__ import annotations

from copy import copy

from market_ai.models import NewsEventAnalysis, ThemeScoreResult


CONFIRMED_CATALYST = "confirmed_catalyst"
UNCONFIRMED_CATALYST = "unconfirmed_catalyst"


def classify_news_market_confirmation(
    events: list[NewsEventAnalysis],
    *,
    core_themes: list[ThemeScoreResult],
    confirmed_score_threshold: float = 50.0,
) -> list[NewsEventAnalysis]:
    """Classify whether news events are confirmed by same-day theme strength."""
    theme_scores = {theme.theme: float(theme.score) for theme in core_themes}
    results = []
    for event in events:
        event_themes = [theme for theme in event.themes if theme in theme_scores]
        score = max([theme_scores[theme] for theme in event_themes] or [0.0])
        if event_themes and score >= float(confirmed_score_threshold):
            state = CONFIRMED_CATALYST
            reason = [f"命中核心题材: {', '.join(event_themes)}", f"最高 ThemeScore={score:.2f}"]
        else:
            state = UNCONFIRMED_CATALYST
            if event_themes:
                reason = [f"命中核心题材但强度不足: {', '.join(event_themes)}", f"最高 ThemeScore={score:.2f}"]
            else:
                reason = ["未命中当日核心题材", "资金确认不足"]
        cloned = copy(event)
        cloned.validation_state = state
        cloned.market_confirm_score = round(score, 6)
        cloned.validation_reason = reason
        results.append(cloned)
    return results
