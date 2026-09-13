"""Rule-based aggregation of news catalysts by core theme."""

from __future__ import annotations

from market_ai.models import NewsEventAnalysis, ThemeCatalystSummary, ThemeScoreResult
from market_ai.scoring.news_confirmation import CONFIRMED_CATALYST
from market_ai.scoring.news_event_type import event_type_score


THEME_EVENT_KEYWORDS = {
    "AI算力": ("ai服务器", "aidc", "数据中心", "算力", "智算", "gpu", "液冷服务器"),
    "CPO光通信": ("cpo", "光模块", "光通信", "硅光", "lpo", "800g", "1.6t", "光芯片", "光器件"),
    "PCB": ("pcb", "覆铜板", "电子布", "高频高速板", "铜箔", "基板"),
    "PCB服务器液冷电源": ("pcb", "覆铜板", "电子布", "服务器", "液冷", "电源", "ups", "连接器", "铜缆"),
    "半导体国产替代": ("半导体", "国产替代", "芯片", "半导体设备", "光刻胶", "先进封装", "hbm", "存储"),
    "机器人具身智能": ("机器人", "具身智能", "人形机器人", "物理ai", "机器视觉", "伺服", "减速器", "丝杠"),
    "低空经济": ("低空经济", "无人机", "evtol", "飞控", "空管", "通航"),
    "端侧AI消费电子": ("端侧ai", "ai手机", "ai pc", "ai眼镜", "智能眼镜", "消费电子", "npu"),
    "卫星互联网商业航天": ("卫星互联网", "商业航天", "卫星通信", "北斗", "商业火箭", "6g", "相控阵雷达"),
    "脑机接口": ("脑机接口", "神经电极", "神经解码", "康复机器人"),
    "量子科技新材料": ("量子科技", "量子通信", "量子计算", "可控核聚变", "超导", "新材料"),
    "电力公用事业": ("电力", "电网", "储能", "火电", "水电", "风电", "光伏", "公用事业"),
}
PRIMARY_EVENT_RELEVANCE_THRESHOLD = 50.0


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
                -theme_event_relevance_score(theme.theme, event),
                -event_type_score(event.event_type),
                -(event.market_confirm_score or 0.0),
                event.event_time,
                event.event,
            ),
        )
        confirmed_count = sum(1 for event in related if event.validation_state == CONFIRMED_CATALYST)
        unconfirmed_count = len(related) - confirmed_count
        top_relevance = theme_event_relevance_score(theme.theme, related[0])
        primary_event = related[0].event if top_relevance >= PRIMARY_EVENT_RELEVANCE_THRESHOLD else ""
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
                conclusion=_conclusion(confirmed_count, unconfirmed_count, has_direct_primary=bool(primary_event)),
            )
        )
    return summaries


def theme_event_relevance_score(theme: str, event: NewsEventAnalysis) -> float:
    """Score whether an event title directly supports a specific theme."""
    theme_text = str(theme).strip()
    event_text = str(event.event or "").strip().lower().replace(" ", "")
    if not theme_text or not event_text:
        return 0.0
    if theme_text.lower().replace(" ", "") in event_text:
        return 100.0
    keywords = THEME_EVENT_KEYWORDS.get(theme_text, ())
    hits = [keyword for keyword in keywords if keyword.lower().replace(" ", "") in event_text]
    if hits:
        return min(95.0, 50.0 + 15.0 * len(hits))
    if theme_text in event.themes:
        return 10.0
    return 0.0


def _conclusion(confirmed_count: int, unconfirmed_count: int, *, has_direct_primary: bool = True) -> str:
    if not has_direct_primary and confirmed_count + unconfirmed_count > 0:
        return "有相关消息，但未直接命中该题材关键词，需人工复核催化归因。"
    if confirmed_count > 0:
        return f"消息与资金形成确认，已确认事件 {confirmed_count} 条。"
    if unconfirmed_count > 0:
        return "有相关消息，但尚未形成强资金确认。"
    return "暂无可验证消息。"
