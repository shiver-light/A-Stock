"""Minimal daily market radar workflow orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from market_ai.config import RadarConfig
from market_ai.lifecycle import classify_theme_lifecycle
from market_ai.models import DailyRadarReport
from market_ai.providers.market import MarketProvider
from market_ai.providers.news import NewsProvider
from market_ai.scoring import calculate_theme_scores
from market_ai.themes import RuleBasedThemeNormalizer, ThemeTaxonomy, normalize_news_items
from market_ai.themes import load_stock_theme_labels, merge_theme_normalizations, stock_codes_from_market_rows


def build_daily_radar_report(
    *,
    trade_date: str,
    market_provider: MarketProvider,
    taxonomy: ThemeTaxonomy,
    config: RadarConfig,
    news_provider: NewsProvider | None = None,
    news_start_time: datetime | None = None,
    news_end_time: datetime | None = None,
    stock_theme_labels_path: str | None = None,
    min_stock_theme_confidence: float = 0.0,
    theme_score_history: pd.DataFrame | None = None,
) -> DailyRadarReport:
    """Build a daily radar report from provider data available by trade date close."""
    limit_stocks = market_provider.get_limit_stocks(trade_date=trade_date)
    strong_stocks = market_provider.get_strong_stocks(
        trade_date=trade_date,
        min_pct_chg=config.strong_stock_min_pct_chg,
    )

    normalizer = RuleBasedThemeNormalizer(taxonomy)
    rule_normalizations = []
    rule_normalizations.extend(
        result
        for result in (normalizer.normalize_limit_stock(stock) for stock in limit_stocks)
        if result is not None
    )
    rule_normalizations.extend(
        result
        for result in (normalizer.normalize_strong_stock(stock) for stock in strong_stocks)
        if result is not None
    )
    label_normalizations = load_stock_theme_labels(
        stock_theme_labels_path,
        trade_date=trade_date,
        stock_codes=stock_codes_from_market_rows(limit_stocks, strong_stocks),
        min_confidence=min_stock_theme_confidence,
    )
    normalizations = merge_theme_normalizations(label_normalizations, rule_normalizations)

    news_events = []
    if news_provider is not None:
        news_end = news_end_time or datetime.now()
        news_start = news_start_time or news_end - timedelta(hours=config.news.lookback_hours)
        news_items = news_provider.fetch_news(start_time=news_start, end_time=news_end)
        news_events = normalize_news_items(news_items, taxonomy)

    core_themes = calculate_theme_scores(
        trade_date=trade_date,
        limit_stocks=limit_stocks,
        strong_stocks=strong_stocks,
        normalizations=normalizations,
        news_events=news_events,
        weights=config.theme_score.weights,
        top_n=config.top_theme_count,
    )
    if theme_score_history is not None:
        core_themes = classify_theme_lifecycle(core_themes, theme_score_history, trade_date=trade_date)
    news_themes = {theme for event in news_events for theme in event.themes}
    unexplained_strength = [theme for theme in core_themes if theme.theme not in news_themes and theme.score >= 50.0]

    return DailyRadarReport(
        trade_date=trade_date,
        core_themes=core_themes,
        news_events=news_events,
        unexplained_strength=unexplained_strength,
        stock_roles=[],
        next_day_observations=_build_observations(core_themes),
        metadata={
            "limit_stock_count": len(limit_stocks),
            "strong_stock_count": len(strong_stocks),
            "normalized_stock_count": len(normalizations),
            "stock_theme_label_count": len(label_normalizations),
            "rule_theme_label_count": len(rule_normalizations),
            "news_event_count": len(news_events),
        },
    )


def _build_observations(core_themes: list) -> list[str]:
    observations = []
    for theme in core_themes[:5]:
        observations.append(f"{theme.theme} 是否继续扩散、龙头是否晋级、成交额是否维持。")
    return observations
