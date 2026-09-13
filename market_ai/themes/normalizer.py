"""Rule-based theme normalization built on top of the taxonomy matcher."""

from __future__ import annotations

import re
from collections.abc import Iterable

from market_ai.models import LimitStock, NewsEventAnalysis, NewsItem, StrongStock, ThemeNormalization
from market_ai.themes.taxonomy import ThemeMatch, ThemeTaxonomy


class RuleBasedThemeNormalizer:
    """Deterministic theme normalizer for market rows and news items."""

    def __init__(self, taxonomy: ThemeTaxonomy) -> None:
        self.taxonomy = taxonomy

    def normalize_text(
        self,
        *,
        stock_code: str,
        text: str,
        source: str = "taxonomy_rule",
        event_driven: bool = False,
    ) -> ThemeNormalization | None:
        """Normalize one raw text into a stock theme normalization result."""
        matches = self.taxonomy.match_text(text)
        if not matches:
            return None
        primary = matches[0]
        return ThemeNormalization(
            stock_code=stock_code,
            primary_theme=primary.theme,
            secondary_themes=_secondary_themes(matches, primary.theme),
            related_entities=[],
            event_driven=event_driven,
            confidence=_rule_confidence(matches),
            source=source,
        )

    def normalize_limit_stock(self, stock: LimitStock) -> ThemeNormalization | None:
        """Normalize a limit-up or touched-limit stock using reason/concepts text."""
        text = _join_text([stock.limit_reason, stock.industry, *stock.concepts])
        return self.normalize_text(
            stock_code=stock.stock_code,
            text=text,
            source=stock.source or "taxonomy_rule",
            event_driven=bool(stock.limit_reason),
        )

    def normalize_strong_stock(self, stock: StrongStock) -> ThemeNormalization | None:
        """Normalize a strong non-limit stock using industry/concepts text."""
        text = _join_text([stock.industry, *stock.concepts])
        return self.normalize_text(
            stock_code=stock.stock_code,
            text=text,
            source=stock.source or "taxonomy_rule",
            event_driven=False,
        )

    def normalize_news_item(self, news: NewsItem) -> NewsEventAnalysis | None:
        """Normalize a news item into a structured rule-based event analysis."""
        text = _join_text([news.title, news.content])
        matches = self.taxonomy.match_text(text)
        if not matches:
            return None
        return NewsEventAnalysis(
            event=news.title,
            event_type="news",
            event_time=news.published_at,
            themes=_unique(match.theme for match in matches),
            related_stocks=[],
            importance=1,
            novelty=1,
            scope="",
            expected_duration="",
            confidence=min(0.8, _rule_confidence(matches)),
            source_news_ids=[news.news_id],
            theme_evidence=_theme_evidence(matches),
        )


def normalize_limit_stocks(
    stocks: Iterable[LimitStock],
    taxonomy: ThemeTaxonomy,
) -> list[ThemeNormalization]:
    """Normalize a batch of limit stock rows and drop unmatched rows."""
    normalizer = RuleBasedThemeNormalizer(taxonomy)
    results = [normalizer.normalize_limit_stock(stock) for stock in stocks]
    return [result for result in results if result is not None]


def normalize_news_items(
    news_items: Iterable[NewsItem],
    taxonomy: ThemeTaxonomy,
) -> list[NewsEventAnalysis]:
    """Normalize a batch of news items and drop unmatched rows."""
    normalizer = RuleBasedThemeNormalizer(taxonomy)
    results = [normalizer.normalize_news_item(news) for news in news_items]
    return [result for result in results if result is not None]


def _secondary_themes(matches: list[ThemeMatch], primary_theme: str) -> list[str]:
    return [theme for theme in _unique(match.theme for match in matches) if theme != primary_theme]


def _rule_confidence(matches: list[ThemeMatch]) -> float:
    if not matches:
        return 0.0
    if len(matches) >= 2:
        return 0.9
    return 0.8


def _theme_evidence(matches: list[ThemeMatch]) -> list[dict[str, object]]:
    evidence = []
    seen = set()
    for match in matches:
        key = (match.theme, match.matched_alias)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "theme": match.theme,
                "matched_keyword": match.matched_alias,
                "source": "taxonomy_rule",
                "source_text": _evidence_excerpt(match.source_text),
                "confidence": match.confidence,
            }
        )
    return evidence


def _evidence_excerpt(text: str, max_length: int = 160) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_length]


def _join_text(values: Iterable[object]) -> str:
    return " ".join(str(value).strip() for value in values if str(value or "").strip())


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
