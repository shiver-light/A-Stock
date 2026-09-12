from __future__ import annotations

import unittest

from market_ai.models import DailyRadarReport, NewsEventAnalysis, StockRole, ThemeScoreResult
from market_ai.reports import render_daily_radar_report_markdown


class MarketRadarReportFormatterTestCase(unittest.TestCase):
    def test_render_empty_report_contains_required_sections(self) -> None:
        report = DailyRadarReport(trade_date="20260912")

        text = render_daily_radar_report_markdown(report)

        self.assertIn("# 市场主线雷达 20260912", text)
        self.assertIn("## 核心板块", text)
        self.assertIn("暂无核心板块", text)
        self.assertIn("不构成买入或卖出建议", text)

    def test_render_report_with_core_theme_news_and_roles(self) -> None:
        report = DailyRadarReport(
            trade_date="20260912",
            core_themes=[
                ThemeScoreResult(
                    trade_date="20260912",
                    theme="AI算力",
                    score=86.5,
                    rank=1,
                    lifecycle_stage=2,
                    reasons=["涨停/触板股票 3 只", "强势股票 5 只"],
                )
            ],
            news_events=[
                NewsEventAnalysis(
                    event="算力基础设施政策发布",
                    event_type="policy",
                    event_time="2026-09-12T16:30:00+08:00",
                    themes=["AI算力"],
                    importance=5,
                    novelty=4,
                    scope="产业链",
                    expected_duration="中期",
                    confidence=0.8,
                )
            ],
            unexplained_strength=[
                ThemeScoreResult(
                    trade_date="20260912",
                    theme="机器人具身智能",
                    score=70.0,
                    rank=2,
                    reasons=["强势股票 4 只"],
                )
            ],
            stock_roles=[
                StockRole(
                    trade_date="20260912",
                    theme="AI算力",
                    stock_code="000001.SZ",
                    stock_name="样本股份",
                    role="capacity_leader",
                    confidence=0.86,
                    reason=["板块成交额第一"],
                )
            ],
            next_day_observations=["AI算力是否继续扩散到首板。"],
            metadata={"source": "fixture"},
        )

        text = render_daily_radar_report_markdown(report)

        self.assertIn("| 1 | AI算力 | 发酵 | 86.50 |", text)
        self.assertIn("算力基础设施政策发布", text)
        self.assertIn("机器人具身智能", text)
        self.assertIn("样本股份(000001.SZ)", text)
        self.assertIn("AI算力是否继续扩散到首板。", text)
        self.assertIn("- source: fixture", text)


if __name__ == "__main__":
    unittest.main()
