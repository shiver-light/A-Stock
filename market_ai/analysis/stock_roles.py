"""Rule-based stock role identification inside core themes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from market_ai.models import LimitStock, StockRole, StrongStock, ThemeNormalization, ThemeScoreResult


@dataclass(frozen=True)
class _StockEvidence:
    stock_code: str
    stock_name: str
    theme: str
    status: str
    pct_chg: float
    amount: float
    turnover_rate: float
    consecutive_limit_count: int
    sealed_amount: float


def identify_stock_roles(
    *,
    trade_date: str,
    core_themes: list[ThemeScoreResult],
    limit_stocks: list[LimitStock],
    strong_stocks: list[StrongStock],
    normalizations: list[ThemeNormalization],
    max_roles_per_theme: int = 8,
) -> list[StockRole]:
    """Identify explainable stock roles for each core theme using same-day market evidence."""
    if not core_themes or not normalizations:
        return []
    stock_themes = _stock_theme_map(normalizations)
    stock_rows = _stock_rows(limit_stocks, strong_stocks)
    theme_evidence = _theme_evidence(core_themes, stock_themes, stock_rows)
    roles = []
    for theme in sorted(core_themes, key=lambda item: (item.rank or 9999, -item.score, item.theme)):
        rows = sorted(
            theme_evidence.get(theme.theme, []),
            key=lambda item: (
                -item.consecutive_limit_count,
                -item.amount,
                -item.turnover_rate,
                -item.pct_chg,
                item.stock_code,
            ),
        )
        if not rows:
            continue
        context = _theme_context(rows)
        for row in rows[: max(1, int(max_roles_per_theme))]:
            role, confidence, reason = _classify_role(row, context)
            roles.append(
                StockRole(
                    trade_date=trade_date,
                    theme=theme.theme,
                    stock_code=row.stock_code,
                    stock_name=row.stock_name,
                    role=role,
                    confidence=confidence,
                    reason=reason,
                )
            )
    return roles


def _stock_theme_map(normalizations: list[ThemeNormalization]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in normalizations:
        for theme in [item.primary_theme, *item.secondary_themes]:
            text = str(theme or "").strip()
            if text:
                result[item.stock_code].add(text)
    return dict(result)


def _stock_rows(limit_stocks: list[LimitStock], strong_stocks: list[StrongStock]) -> dict[str, LimitStock | StrongStock]:
    rows: dict[str, LimitStock | StrongStock] = {stock.stock_code: stock for stock in strong_stocks}
    rows.update({stock.stock_code: stock for stock in limit_stocks})
    return rows


def _theme_evidence(
    core_themes: list[ThemeScoreResult],
    stock_themes: dict[str, set[str]],
    stock_rows: dict[str, LimitStock | StrongStock],
) -> dict[str, list[_StockEvidence]]:
    core_theme_set = {theme.theme for theme in core_themes}
    result: dict[str, list[_StockEvidence]] = defaultdict(list)
    for stock_code, themes in stock_themes.items():
        row = stock_rows.get(stock_code)
        if row is None:
            continue
        for theme in themes & core_theme_set:
            result[theme].append(_to_evidence(row, theme))
    return dict(result)


def _to_evidence(row: LimitStock | StrongStock, theme: str) -> _StockEvidence:
    return _StockEvidence(
        stock_code=row.stock_code,
        stock_name=row.stock_name,
        theme=theme,
        status=getattr(row, "status", "strong"),
        pct_chg=_number(row.pct_chg),
        amount=_number(row.amount),
        turnover_rate=_number(row.turnover_rate),
        consecutive_limit_count=max(0, int(getattr(row, "consecutive_limit_count", None) or 0)),
        sealed_amount=_number(getattr(row, "sealed_amount", None)),
    )


def _theme_context(rows: list[_StockEvidence]) -> dict[str, float]:
    amounts = [row.amount for row in rows if row.amount > 0.0]
    turnovers = [row.turnover_rate for row in rows if row.turnover_rate > 0.0]
    consecutive_counts = [row.consecutive_limit_count for row in rows]
    return {
        "max_board": float(max(consecutive_counts or [0])),
        "max_amount": float(max(amounts or [0.0])),
        "median_amount": float(median(amounts)) if amounts else 0.0,
        "max_turnover": float(max(turnovers or [0.0])),
    }


def _classify_role(row: _StockEvidence, context: dict[str, float]) -> tuple[str, float, list[str]]:
    max_board = int(context["max_board"])
    max_amount = context["max_amount"]
    median_amount = context["median_amount"]
    max_turnover = context["max_turnover"]
    is_limit_up = row.status == "limit_up"
    is_touched = row.status in {"touched_limit", "failed_limit"}

    if is_limit_up and row.consecutive_limit_count >= max(2, max_board):
        return (
            "space_leader",
            0.9,
            [f"题材内最高连板 {row.consecutive_limit_count} 板", f"涨幅 {row.pct_chg:.2f}%"],
        )
    if max_amount > 0.0 and row.amount == max_amount and (median_amount <= 0.0 or row.amount >= median_amount):
        return (
            "capacity_leader",
            0.84,
            [f"题材内成交额最高 {row.amount:.2f}", f"涨幅 {row.pct_chg:.2f}%"],
        )
    if is_limit_up and max_turnover > 0.0 and row.turnover_rate == max_turnover:
        return (
            "turnover_leader",
            0.8,
            [f"题材内换手率最高 {row.turnover_rate:.2f}%", f"涨幅 {row.pct_chg:.2f}%"],
        )
    if is_limit_up and row.consecutive_limit_count >= 2:
        return (
            "core_frontline",
            0.78,
            [f"连板 {row.consecutive_limit_count} 板", f"涨幅 {row.pct_chg:.2f}%"],
        )
    if is_limit_up and max_board >= 2 and row.consecutive_limit_count <= 1:
        return (
            "catch_up",
            0.72,
            ["题材已有连板高度，该股为首板补涨", f"涨幅 {row.pct_chg:.2f}%"],
        )
    if is_touched or row.pct_chg >= 7.0:
        return (
            "elasticity_stock",
            0.68,
            [f"强势涨幅 {row.pct_chg:.2f}%", f"状态 {row.status}"],
        )
    return ("follower", 0.6, [f"题材内跟随表现，涨幅 {row.pct_chg:.2f}%"])


def _number(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)
