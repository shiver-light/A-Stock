"""Markdown formatting for daily market radar reports."""

from __future__ import annotations

from market_ai.models import DailyRadarReport, NewsEventAnalysis, StockRole, ThemeCatalystSummary, ThemeScoreResult


LIFECYCLE_STAGE_NAMES = {
    0: "潜伏",
    1: "启动",
    2: "发酵",
    3: "主升",
    4: "高潮",
    5: "分歧",
    6: "回流",
    7: "退潮",
}


def render_daily_radar_report_markdown(report: DailyRadarReport) -> str:
    """Render a structured daily radar report as Markdown."""
    lines = [
        f"# 市场主线雷达 {report.trade_date}",
        "",
        "> 盘后结构化分析，仅用于研究复盘；不构成买入或卖出建议。",
        "",
        "## 核心板块",
    ]
    lines.extend(_render_core_themes(report.core_themes))
    lines.extend(["", "## 核心题材催化"])
    lines.extend(_render_theme_catalysts(report.theme_catalysts))
    lines.extend(["", "## 今日核心消息"])
    lines.extend(_render_news_events(report.news_events))
    lines.extend(["", "## 异常资金"])
    lines.extend(_render_unexplained_strength(report.unexplained_strength))
    lines.extend(["", "## 龙头结构"])
    lines.extend(_render_stock_roles(report.stock_roles))
    lines.extend(["", "## 明日观察"])
    lines.extend(_render_observations(report.next_day_observations))
    lines.extend(["", "## 风险提示"])
    lines.extend(
        [
            "- 基于历史行情、公开新闻和规则化标签，不保证未来表现。",
            "- 题材归因和新闻解释可能存在数据源缺失或语义误判。",
            "- 第一版 ThemeScore 仍是研究评分，不等同于可交易信号。",
        ]
    )
    if report.metadata:
        lines.extend(["", "## 元数据"])
        for key, value in sorted(report.metadata.items()):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def _render_core_themes(themes: list[ThemeScoreResult]) -> list[str]:
    if not themes:
        return ["暂无核心板块。"]
    lines = [
        "| 排名 | 题材 | 阶段 | ThemeScore | 涨停/强势证据 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for theme in sorted(themes, key=lambda item: (item.rank or 9999, -item.score, item.theme)):
        lines.append(
            "| {rank} | {theme} | {stage} | {score:.2f} | {reasons} |".format(
                rank=theme.rank or "",
                theme=theme.theme,
                stage=_stage_name(theme.lifecycle_stage),
                score=theme.score,
                reasons="<br>".join(theme.reasons) if theme.reasons else "N/A",
            )
        )
    return lines


def _render_news_events(events: list[NewsEventAnalysis]) -> list[str]:
    if not events:
        return ["暂无已结构化新闻事件。"]
    lines = []
    for event in events:
        themes = ", ".join(event.themes) if event.themes else "N/A"
        lines.extend(
            [
                f"- 事件：{event.event}",
                f"  - 对应题材：{themes}",
                f"  - 类型/时间：{event.event_type} / {event.event_time}",
                f"  - 重要性/新鲜度/置信度：{event.importance} / {event.novelty} / {event.confidence:.2f}",
                f"  - 资金验证：{event.validation_state or 'N/A'} / {event.market_confirm_score if event.market_confirm_score is not None else 'N/A'}",
                f"  - 验证原因：{'; '.join(event.validation_reason) if event.validation_reason else 'N/A'}",
                f"  - 影响范围：{event.scope or 'N/A'}",
                f"  - 预期持续性：{event.expected_duration or 'N/A'}",
            ]
        )
    return lines


def _render_theme_catalysts(catalysts: list[ThemeCatalystSummary]) -> list[str]:
    if not catalysts:
        return ["暂无核心题材催化聚合。"]
    lines = [
        "| 题材 | 主催化 | 已确认/未确认 | 证据事件 | 结论 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in catalysts:
        lines.append(
            "| {theme} | {primary_event} | {confirmed}/{unconfirmed} | {events} | {conclusion} |".format(
                theme=item.theme,
                primary_event=item.primary_event or "N/A",
                confirmed=item.confirmed_event_count,
                unconfirmed=item.unconfirmed_event_count,
                events="<br>".join(item.evidence_events) if item.evidence_events else "N/A",
                conclusion=item.conclusion or "N/A",
            )
        )
    return lines


def _render_unexplained_strength(themes: list[ThemeScoreResult]) -> list[str]:
    if not themes:
        return ["暂无资金强但消息解释不足的题材。"]
    lines = []
    for theme in sorted(themes, key=lambda item: (item.rank or 9999, -item.score, item.theme)):
        lines.append(f"- {theme.theme}: ThemeScore={theme.score:.2f}，原因：{'; '.join(theme.reasons) or 'N/A'}")
    return lines


def _render_stock_roles(roles: list[StockRole]) -> list[str]:
    if not roles:
        return ["暂无板块内部角色识别结果。"]
    lines = [
        "| 题材 | 股票 | 角色 | 置信度 | 原因 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for role in sorted(roles, key=lambda item: (item.theme, item.role, item.stock_code)):
        lines.append(
            "| {theme} | {stock_name}({stock_code}) | {role} | {confidence:.2f} | {reason} |".format(
                theme=role.theme,
                stock_name=role.stock_name,
                stock_code=role.stock_code,
                role=role.role,
                confidence=role.confidence,
                reason="<br>".join(role.reason) if role.reason else "N/A",
            )
        )
    return lines


def _render_observations(observations: list[str]) -> list[str]:
    if not observations:
        return ["- 暂无观察项。"]
    return [f"- {item}" for item in observations]


def _stage_name(stage: int | None) -> str:
    if stage is None:
        return "未判定"
    return LIFECYCLE_STAGE_NAMES.get(stage, f"未知({stage})")
