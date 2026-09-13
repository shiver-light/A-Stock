from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.scoring import rank_core_news_events, select_core_news_events


class MarketAiCoreNewsEventsTestCase(unittest.TestCase):
    def test_select_core_news_events_prefers_authoritative_core_theme_news(self) -> None:
        news_items = [
            NewsItem(news_id="n1", source="新华社", title="算力政策发布", published_at="2026-09-11T15:00:00+08:00"),
            NewsItem(news_id="n2", source="未知来源", title="机器人消息", published_at="2026-09-11T15:01:00+08:00"),
        ]
        events = [
            NewsEventAnalysis(
                event="算力政策发布",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["AI算力"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="机器人消息",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["机器人具身智能"],
                source_news_ids=["n2"],
            ),
        ]

        selected = select_core_news_events(
            events,
            news_items=news_items,
            core_themes=["AI算力"],
            authority_scores={"新华社": 95.0},
            default_authority_score=40.0,
            min_score=60.0,
            top_n=10,
        )

        self.assertEqual([event.event for event in selected], ["算力政策发布"])

    def test_rank_core_news_events_can_keep_non_core_when_configured(self) -> None:
        news_items = [
            NewsItem(news_id="n1", source="财联社", title="非核心题材", published_at="2026-09-11T15:00:00+08:00")
        ]
        events = [
            NewsEventAnalysis(
                event="非核心题材",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["低空经济"],
                source_news_ids=["n1"],
            )
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["AI算力"],
            authority_scores={"财联社": 80.0},
            require_core_theme_match=False,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].core_theme_score, 0.0)

    def test_rank_core_news_events_uses_directness_and_impact(self) -> None:
        news_items = [
            NewsItem(
                news_id="n1",
                source="财联社",
                title="海外科技巨头发布会",
                published_at="2026-09-11T15:00:00+08:00",
                content="苹果发布会带动海外科技关注。",
            ),
            NewsItem(
                news_id="n2",
                source="财联社",
                title="A股PCB板块多股涨停",
                published_at="2026-09-11T15:01:00+08:00",
                content="AI服务器订单增长，产业链公司扩产。",
            ),
        ]
        events = [
            NewsEventAnalysis(
                event="海外科技巨头发布会",
                event_type="news",
                event_time="2026-09-11T15:00:00+08:00",
                themes=["PCB服务器液冷电源"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="A股PCB板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["PCB服务器液冷电源"],
                source_news_ids=["n2"],
            ),
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["PCB服务器液冷电源"],
            authority_scores={"财联社": 80.0},
        )

        self.assertEqual(ranked[0].event.event, "A股PCB板块多股涨停")
        self.assertGreater(ranked[0].directness_score, ranked[1].directness_score)
        self.assertGreater(ranked[0].impact_score, ranked[1].impact_score)
        self.assertEqual(ranked[0].event.event_type, "order")
        self.assertGreater(ranked[0].event_type_score, ranked[1].event_type_score)

    def test_rank_core_news_events_prefers_hard_catalyst_type(self) -> None:
        news_items = [
            NewsItem(
                news_id="n1",
                source="财联社",
                title="AI算力投资逻辑正被重新定义",
                published_at="2026-09-11T09:00:00+08:00",
                content="行业专题关注算力产业链。",
            ),
            NewsItem(
                news_id="n2",
                source="财联社",
                title="AI服务器订单增长",
                published_at="2026-09-11T10:00:00+08:00",
                content="上市公司签订采购合同。",
            ),
        ]
        events = [
            NewsEventAnalysis(
                event="AI算力投资逻辑正被重新定义",
                event_type="news",
                event_time="2026-09-11T09:00:00+08:00",
                themes=["AI算力"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="AI服务器订单增长",
                event_type="news",
                event_time="2026-09-11T10:00:00+08:00",
                themes=["AI算力"],
                source_news_ids=["n2"],
            ),
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["AI算力"],
            authority_scores={"财联社": 80.0},
            semantic_weights={
                "authority": 0.1,
                "core_theme": 0.1,
                "theme_match": 0.1,
                "directness": 0.1,
                "impact": 0.1,
                "event_type": 1.0,
            },
        )

        self.assertEqual(ranked[0].event.event, "AI服务器订单增长")
        self.assertEqual(ranked[0].event.event_type, "order")

    def test_rank_core_news_events_penalizes_summary_news(self) -> None:
        news_items = [
            NewsItem(
                news_id="n1",
                source="财联社",
                title="【早报】美股科技股上涨 A股PCB板块受关注",
                published_at="2026-09-11T08:00:00+08:00",
                content="AI服务器产业链受关注。",
            ),
            NewsItem(
                news_id="n2",
                source="财联社",
                title="A股PCB板块多股涨停",
                published_at="2026-09-11T15:01:00+08:00",
                content="AI服务器订单增长，产业链公司扩产。",
            ),
        ]
        events = [
            NewsEventAnalysis(
                event="【早报】美股科技股上涨 A股PCB板块受关注",
                event_type="news",
                event_time="2026-09-11T08:00:00+08:00",
                themes=["PCB服务器液冷电源"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="A股PCB板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["PCB服务器液冷电源"],
                source_news_ids=["n2"],
            ),
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["PCB服务器液冷电源"],
            authority_scores={"财联社": 80.0},
        )

        self.assertEqual(ranked[0].event.event, "A股PCB板块多股涨停")
        self.assertGreater(ranked[1].noise_penalty, 0.0)

    def test_rank_core_news_events_penalizes_single_stock_event(self) -> None:
        news_items = [
            NewsItem(
                news_id="n1",
                source="财联社",
                title="揭秘*ST高科控制权之争：前董事长被刑事立案",
                published_at="2026-09-11T10:00:00+08:00",
            ),
            NewsItem(
                news_id="n2",
                source="财联社",
                title="A股半导体板块多股涨停",
                published_at="2026-09-11T15:01:00+08:00",
                content="政策推动产业链订单增长。",
            ),
        ]
        events = [
            NewsEventAnalysis(
                event="揭秘*ST高科控制权之争：前董事长被刑事立案",
                event_type="news",
                event_time="2026-09-11T10:00:00+08:00",
                themes=["半导体国产替代"],
                source_news_ids=["n1"],
            ),
            NewsEventAnalysis(
                event="A股半导体板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["半导体国产替代"],
                source_news_ids=["n2"],
            ),
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["半导体国产替代"],
            authority_scores={"财联社": 80.0},
        )

        self.assertEqual(ranked[0].event.event, "A股半导体板块多股涨停")
        self.assertGreater(ranked[1].stock_event_penalty, 0.0)

    def test_penalty_weights_do_not_dilute_positive_score_components(self) -> None:
        news_items = [
            NewsItem(
                news_id="n1",
                source="财联社",
                title="A股PCB板块多股涨停",
                published_at="2026-09-11T15:01:00+08:00",
                content="AI服务器订单增长，产业链公司扩产。",
            )
        ]
        events = [
            NewsEventAnalysis(
                event="A股PCB板块多股涨停",
                event_type="news",
                event_time="2026-09-11T15:01:00+08:00",
                themes=["PCB服务器液冷电源"],
                source_news_ids=["n1"],
            )
        ]

        ranked = rank_core_news_events(
            events,
            news_items=news_items,
            core_themes=["PCB服务器液冷电源"],
            authority_scores={"财联社": 80.0},
            semantic_weights={
                "authority": 0.3,
                "core_theme": 0.3,
                "theme_match": 0.15,
                "directness": 0.15,
                "impact": 0.1,
                "noise_penalty": 0.9,
                "stock_event_penalty": 0.9,
            },
        )

        self.assertGreaterEqual(ranked[0].score, 60.0)


if __name__ == "__main__":
    unittest.main()
