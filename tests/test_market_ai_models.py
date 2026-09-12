from __future__ import annotations

import json
import unittest

from market_ai.models import (
    DailyRadarReport,
    LimitStock,
    NewsEventAnalysis,
    StockRole,
    ThemeNormalization,
    ThemeScoreResult,
)


class MarketAiModelsTestCase(unittest.TestCase):
    def test_theme_normalization_is_json_friendly(self) -> None:
        model = ThemeNormalization(
            stock_code="000001.SZ",
            primary_theme="AI算力",
            secondary_themes=["AIDC", "液冷"],
            related_entities=["NVIDIA", "华为"],
            event_driven=True,
            confidence=0.86,
            source="llm",
        )

        payload = model.to_dict()

        self.assertEqual(payload["primary_theme"], "AI算力")
        self.assertEqual(json.loads(json.dumps(payload, ensure_ascii=False))["confidence"], 0.86)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence must be between 0 and 1"):
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=1.2)

    def test_news_event_scores_are_bounded(self) -> None:
        event = NewsEventAnalysis(
            event="工信部发布算力基础设施政策",
            event_type="policy",
            event_time="2026-09-12T16:00:00+08:00",
            themes=["AI算力"],
            importance=5,
            novelty=4,
            confidence=0.75,
        )

        self.assertEqual(event.to_dict()["importance"], 5)
        with self.assertRaisesRegex(ValueError, "novelty must be between 1 and 5"):
            NewsEventAnalysis(
                event="old news",
                event_type="policy",
                event_time="2026-09-12",
                novelty=0,
            )

    def test_daily_radar_report_roundtrip(self) -> None:
        theme = ThemeScoreResult(
            trade_date="20260912",
            theme="机器人",
            score=82.5,
            components={"LimitUpStrength": 88.0, "NewsCatalyst": 75.0},
            rank=1,
            lifecycle_stage=2,
            validation_state="A",
            reasons=["涨停扩散", "消息得到资金确认"],
        )
        role = StockRole(
            trade_date="20260912",
            theme="机器人",
            stock_code="000001.SZ",
            stock_name="示例股份",
            role="capacity_leader",
            confidence=0.8,
            reason=["板块成交额第一"],
        )
        report = DailyRadarReport(
            trade_date="20260912",
            core_themes=[theme],
            unexplained_strength=[],
            stock_roles=[role],
            next_day_observations=["龙头是否晋级"],
            metadata={"prompt_version": "none"},
        )

        payload = report.to_dict()

        self.assertEqual(payload["core_themes"][0]["theme"], "机器人")
        self.assertEqual(payload["stock_roles"][0]["role"], "capacity_leader")
        self.assertEqual(json.loads(json.dumps(payload, ensure_ascii=False))["trade_date"], "20260912")

    def test_limit_stock_requires_identity_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "stock_code is required"):
            LimitStock(trade_date="20260912", stock_code="", stock_name="示例股份")


if __name__ == "__main__":
    unittest.main()
