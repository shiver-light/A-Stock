"""Minimal daily market radar workflow orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from market_ai.config import RadarConfig
from market_ai.lifecycle import classify_theme_lifecycle
from market_ai.lifecycle.rules import STAGE_NAMES as LIFECYCLE_STAGE_NAMES
from market_ai.models import DailyRadarReport
from market_ai.providers.market import MarketProvider
from market_ai.providers.news import NewsProvider
from market_ai.scoring import (
    aggregate_theme_catalysts,
    calculate_theme_scores,
    classify_news_market_confirmation,
    select_core_news_events,
)
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
    news_items = []
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
    core_news_events = select_core_news_events(
        news_events,
        news_items=news_items,
        core_themes=[theme.theme for theme in core_themes],
        authority_scores=config.news_scoring.authority_scores,
        default_authority_score=config.news_scoring.default_authority_score,
        min_score=config.news_scoring.min_core_event_score,
        top_n=config.news_scoring.core_event_top_n,
        semantic_weights=config.news_scoring.semantic_weights,
        require_core_theme_match=config.news_scoring.require_core_theme_match,
    )
    core_news_events = classify_news_market_confirmation(core_news_events, core_themes=core_themes)
    theme_catalysts = aggregate_theme_catalysts(
        trade_date=trade_date,
        core_themes=core_themes,
        news_events=core_news_events,
    )

    return DailyRadarReport(
        trade_date=trade_date,
        core_themes=core_themes,
        news_events=core_news_events,
        theme_catalysts=theme_catalysts,
        unexplained_strength=unexplained_strength,
        stock_roles=[],
        next_day_observations=_build_observations(core_themes, theme_catalysts),
        metadata={
            "limit_stock_count": len(limit_stocks),
            "strong_stock_count": len(strong_stocks),
            "normalized_stock_count": len(normalizations),
            "stock_theme_label_count": len(label_normalizations),
            "rule_theme_label_count": len(rule_normalizations),
            "news_event_total_count": len(news_events),
            "core_news_event_count": len(core_news_events),
            "theme_catalyst_count": len(theme_catalysts),
            "news_event_count": len(core_news_events),
        },
    )


def _build_observations(core_themes: list, theme_catalysts: list | None = None) -> list[str]:
    observations = []
    catalysts = {item.theme: item for item in theme_catalysts or []}
    for theme in core_themes[:5]:
        catalyst = catalysts.get(theme.theme)
        observations.append(_theme_observation(theme, catalyst))
    return observations


def _theme_observation(theme, catalyst) -> str:
    theme_name = str(theme.theme)
    stage = theme.lifecycle_stage
    stage_name = LIFECYCLE_STAGE_NAMES.get(stage, "未判定")
    score = float(theme.score)
    rank = theme.rank or "N/A"
    if stage == 1:
        base = f"{theme_name}处于{stage_name}，观察是否从首日启动扩散到更多涨停/强势股"
    elif stage == 2:
        base = f"{theme_name}处于{stage_name}，观察排名和成交额强度是否继续维持"
    elif stage == 3:
        base = f"{theme_name}处于{stage_name}，观察龙头是否晋级、中军是否继续放量承接"
    elif stage == 4:
        base = f"{theme_name}处于{stage_name}，观察炸板率和高位分歧是否扩大"
    elif stage == 5:
        base = f"{theme_name}处于{stage_name}，观察前排是否修复、后排是否继续掉队"
    elif stage == 6:
        base = f"{theme_name}处于{stage_name}，观察是否形成二次走强和资金回补"
    elif stage == 7:
        base = f"{theme_name}处于{stage_name}，观察是否继续降温，原则上降低优先级"
    else:
        base = f"{theme_name}处于{stage_name}，观察是否重新进入活跃扩散"

    confirmation = _confirmation_text(catalyst)
    return f"{base}；当前排名 {rank}，ThemeScore={score:.2f}；{confirmation}。"


def _confirmation_text(catalyst) -> str:
    if catalyst is None:
        return "暂无消息催化聚合"
    if catalyst.confirmed_event_count > 0:
        event = f"，主催化：{catalyst.primary_event}" if catalyst.primary_event else ""
        return f"消息确认 {catalyst.confirmed_event_count} 条{event}"
    if catalyst.related_event_count > 0:
        return "有相关消息但缺少直接确认，需人工复核催化归因"
    return "暂无核心消息直接验证，更多依赖资金行为"
