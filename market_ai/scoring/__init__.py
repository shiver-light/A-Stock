"""Theme scoring utilities for daily market radar workflows."""

from market_ai.scoring.news_authority import AuthorityScoreResult, calculate_authority_score
from market_ai.scoring.news_confirmation import (
    CONFIRMED_CATALYST,
    UNCONFIRMED_CATALYST,
    classify_news_market_confirmation,
)
from market_ai.scoring.news_events import CoreNewsEventScore, rank_core_news_events, select_core_news_events
from market_ai.scoring.news_semantic import NewsSemanticScore, calculate_news_semantic_score
from market_ai.scoring.theme_score import calculate_theme_scores
from market_ai.scoring.theme_catalysts import aggregate_theme_catalysts

__all__ = [
    "AuthorityScoreResult",
    "CoreNewsEventScore",
    "NewsSemanticScore",
    "CONFIRMED_CATALYST",
    "UNCONFIRMED_CATALYST",
    "calculate_authority_score",
    "calculate_news_semantic_score",
    "calculate_theme_scores",
    "classify_news_market_confirmation",
    "aggregate_theme_catalysts",
    "rank_core_news_events",
    "select_core_news_events",
]
