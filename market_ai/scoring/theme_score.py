"""Configurable ThemeScore calculation from normalized market evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from market_ai.config import DEFAULT_THEME_SCORE_WEIGHTS
from market_ai.models import LimitStock, NewsEventAnalysis, StrongStock, ThemeNormalization, ThemeScoreResult


def calculate_theme_scores(
    *,
    trade_date: str,
    limit_stocks: Iterable[LimitStock],
    strong_stocks: Iterable[StrongStock],
    normalizations: Iterable[ThemeNormalization],
    news_events: Iterable[NewsEventAnalysis] | None = None,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[ThemeScoreResult]:
    """Calculate ranked ThemeScore results from market and optional news evidence."""
    stock_themes = _build_stock_theme_map(normalizations)
    if not stock_themes:
        return []

    limit_by_code = {stock.stock_code: stock for stock in limit_stocks}
    strong_by_code = {stock.stock_code: stock for stock in strong_stocks}
    theme_rows = _build_theme_rows(stock_themes, limit_by_code, strong_by_code)
    if not theme_rows:
        return []

    component_raw = _calculate_raw_components(theme_rows, news_events or [])
    component_scores = {
        component: _scale_to_100(values) for component, values in component_raw.items()
    }
    normalized_weights = _normalize_weights(weights or DEFAULT_THEME_SCORE_WEIGHTS)

    results = []
    for theme in sorted(theme_rows):
        components = {
            component: component_scores.get(component, {}).get(theme, 0.0)
            for component in normalized_weights
        }
        score = sum(components[component] * weight for component, weight in normalized_weights.items())
        results.append(
            ThemeScoreResult(
                trade_date=trade_date,
                theme=theme,
                score=round(min(100.0, max(0.0, score)), 6),
                components={key: round(value, 6) for key, value in components.items()},
                reasons=_build_reasons(theme_rows[theme]),
            )
        )

    results.sort(key=lambda item: (-item.score, item.theme))
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    if top_n is not None:
        return results[: max(0, int(top_n))]
    return results


def _build_stock_theme_map(normalizations: Iterable[ThemeNormalization]) -> dict[str, set[str]]:
    stock_themes: dict[str, set[str]] = defaultdict(set)
    for item in normalizations:
        themes = [item.primary_theme, *item.secondary_themes]
        for theme in themes:
            if str(theme).strip():
                stock_themes[item.stock_code].add(str(theme).strip())
    return dict(stock_themes)


def _build_theme_rows(
    stock_themes: dict[str, set[str]],
    limit_by_code: dict[str, LimitStock],
    strong_by_code: dict[str, StrongStock],
) -> dict[str, dict[str, list[LimitStock] | list[StrongStock]]]:
    theme_rows: dict[str, dict[str, list[LimitStock] | list[StrongStock]]] = {}
    for stock_code, themes in stock_themes.items():
        limit_stock = limit_by_code.get(stock_code)
        strong_stock = strong_by_code.get(stock_code)
        if limit_stock is None and strong_stock is None:
            continue
        for theme in themes:
            rows = theme_rows.setdefault(theme, {"limit_stocks": [], "strong_stocks": []})
            if limit_stock is not None:
                rows["limit_stocks"].append(limit_stock)
            if strong_stock is not None:
                rows["strong_stocks"].append(strong_stock)
    return theme_rows


def _calculate_raw_components(
    theme_rows: dict[str, dict[str, list[LimitStock] | list[StrongStock]]],
    news_events: Iterable[NewsEventAnalysis],
) -> dict[str, dict[str, float]]:
    raw = {
        "LimitUpStrength": {},
        "Breadth": {},
        "LeaderStrength": {},
        "NewsCatalyst": {},
        "VolumeExpansion": {},
        "Persistence": {},
        "Novelty": {},
    }
    for theme, rows in theme_rows.items():
        limit_stocks = list(rows["limit_stocks"])
        strong_stocks = list(rows["strong_stocks"])
        stock_codes = {stock.stock_code for stock in [*limit_stocks, *strong_stocks]}
        limit_up_count = sum(1 for stock in limit_stocks if stock.status == "limit_up")
        consecutive_counts = [max(1, int(stock.consecutive_limit_count or 1)) for stock in limit_stocks]
        amount_sum = sum(_non_negative(stock.amount) for stock in [*limit_stocks, *strong_stocks])
        max_pct_chg = max([float(stock.pct_chg or 0.0) for stock in [*limit_stocks, *strong_stocks]] or [0.0])

        raw["LimitUpStrength"][theme] = limit_up_count + 0.5 * sum(consecutive_counts)
        raw["Breadth"][theme] = len(stock_codes) + 0.5 * len(strong_stocks)
        raw["LeaderStrength"][theme] = 2.0 * max(consecutive_counts or [0]) + max_pct_chg / 10.0
        raw["VolumeExpansion"][theme] = amount_sum
        raw["Persistence"][theme] = 0.0

    news_by_theme = _group_news_events(news_events)
    for theme in theme_rows:
        events = news_by_theme.get(theme, [])
        raw["NewsCatalyst"][theme] = sum(event.importance * event.confidence for event in events) / max(1, len(events))
        raw["Novelty"][theme] = sum(event.novelty * event.confidence for event in events) / max(1, len(events))
    return raw


def _group_news_events(news_events: Iterable[NewsEventAnalysis]) -> dict[str, list[NewsEventAnalysis]]:
    grouped: dict[str, list[NewsEventAnalysis]] = defaultdict(list)
    for event in news_events:
        for theme in event.themes:
            grouped[theme].append(event)
    return dict(grouped)


def _scale_to_100(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0.0:
        return {theme: 0.0 for theme in values}
    return {theme: min(100.0, max(0.0, value / max_value * 100.0)) for theme, value in values.items()}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {str(key): float(value) for key, value in weights.items() if float(value) > 0.0}
    if not cleaned:
        raise ValueError("ThemeScore weights must contain at least one positive value.")
    total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items()}


def _non_negative(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, float(value))


def _build_reasons(rows: dict[str, list[LimitStock] | list[StrongStock]]) -> list[str]:
    limit_count = len(rows["limit_stocks"])
    strong_count = len(rows["strong_stocks"])
    reasons = []
    if limit_count:
        reasons.append(f"涨停/触板股票 {limit_count} 只")
    if strong_count:
        reasons.append(f"强势股票 {strong_count} 只")
    return reasons
