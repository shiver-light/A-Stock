from __future__ import annotations

import unittest

from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.scoring import calculate_news_semantic_score


class MarketAiNewsSemanticTestCase(unittest.TestCase):
    def test_direct_a_share_policy_news_scores_high_directness(self) -> None:
        event = NewsEventAnalysis(
            event="工信部发布算力基础设施政策",
            event_type="news",
            event_time="2026-09-11T15:00:00+08:00",
            themes=["AI算力"],
            source_news_ids=["n1"],
        )
        item = NewsItem(
            news_id="n1",
            source="财联社",
            title="A股算力板块多股涨停",
            published_at="2026-09-11T15:00:00+08:00",
            content="政策推动数据中心产业链建设。",
        )

        score = calculate_news_semantic_score(event, item)

        self.assertGreaterEqual(score.directness_score, 80.0)
        self.assertGreaterEqual(score.impact_score, 60.0)

    def test_overseas_only_news_scores_low_directness(self) -> None:
        event = NewsEventAnalysis(
            event="特朗普称美股科技股上涨",
            event_type="news",
            event_time="2026-09-11T15:00:00+08:00",
            themes=["CPO光通信"],
            source_news_ids=["n1"],
        )

        score = calculate_news_semantic_score(event)

        self.assertEqual(score.directness_score, 20.0)
        self.assertGreater(score.noise_penalty, 0.0)

    def test_market_summary_news_has_noise_penalty(self) -> None:
        event = NewsEventAnalysis(
            event="【早报】美股科技股上涨 A股算力板块关注度提升",
            event_type="news",
            event_time="2026-09-11T08:00:00+08:00",
            themes=["AI算力"],
            source_news_ids=["n1"],
        )

        score = calculate_news_semantic_score(event)

        self.assertGreater(score.noise_penalty, 0.0)
        self.assertTrue(any(reason.startswith("summary_terms=") for reason in score.reason))


if __name__ == "__main__":
    unittest.main()
