from __future__ import annotations

import unittest

from market_ai.models import LimitStock, NewsEventAnalysis, StrongStock, ThemeNormalization
from market_ai.scoring import calculate_theme_scores


class ThemeScoreTestCase(unittest.TestCase):
    def test_empty_normalizations_return_empty_scores(self) -> None:
        results = calculate_theme_scores(
            trade_date="20260912",
            limit_stocks=[],
            strong_stocks=[],
            normalizations=[],
        )

        self.assertEqual(results, [])

    def test_scores_rank_themes_from_market_evidence(self) -> None:
        limit_stocks = [
            LimitStock(
                trade_date="20260912",
                stock_code="000001.SZ",
                stock_name="样本一",
                pct_chg=10.0,
                amount=1000.0,
                consecutive_limit_count=2,
            ),
            LimitStock(
                trade_date="20260912",
                stock_code="000002.SZ",
                stock_name="样本二",
                pct_chg=10.0,
                amount=500.0,
                consecutive_limit_count=1,
            ),
        ]
        strong_stocks = [
            StrongStock(
                trade_date="20260912",
                stock_code="000003.SZ",
                stock_name="样本三",
                pct_chg=8.0,
                amount=300.0,
            )
        ]
        normalizations = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000002.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000003.SZ", primary_theme="CPO光通信", confidence=0.9),
        ]

        results = calculate_theme_scores(
            trade_date="20260912",
            limit_stocks=limit_stocks,
            strong_stocks=strong_stocks,
            normalizations=normalizations,
            weights={"LimitUpStrength": 1.0, "Breadth": 1.0, "VolumeExpansion": 1.0},
        )

        self.assertEqual(results[0].theme, "AI算力")
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, results[1].score)
        self.assertIn("LimitUpStrength", results[0].components)

    def test_news_catalyst_and_novelty_can_drive_score(self) -> None:
        strong_stocks = [
            StrongStock(
                trade_date="20260912",
                stock_code="000001.SZ",
                stock_name="样本一",
                pct_chg=8.0,
            ),
            StrongStock(
                trade_date="20260912",
                stock_code="000002.SZ",
                stock_name="样本二",
                pct_chg=8.0,
            ),
        ]
        normalizations = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000002.SZ", primary_theme="CPO光通信", confidence=0.9),
        ]
        news_events = [
            NewsEventAnalysis(
                event="算力政策",
                event_type="policy",
                event_time="2026-09-12T16:30:00+08:00",
                themes=["AI算力"],
                importance=5,
                novelty=5,
                confidence=1.0,
            )
        ]

        results = calculate_theme_scores(
            trade_date="20260912",
            limit_stocks=[],
            strong_stocks=strong_stocks,
            normalizations=normalizations,
            news_events=news_events,
            weights={"NewsCatalyst": 1.0, "Novelty": 1.0},
        )

        self.assertEqual(results[0].theme, "AI算力")
        self.assertEqual(results[0].components["NewsCatalyst"], 100.0)
        self.assertEqual(results[0].components["Novelty"], 100.0)

    def test_top_n_limits_ranked_results(self) -> None:
        strong_stocks = [
            StrongStock(trade_date="20260912", stock_code="000001.SZ", stock_name="样本一", pct_chg=8.0),
            StrongStock(trade_date="20260912", stock_code="000002.SZ", stock_name="样本二", pct_chg=8.0),
        ]
        normalizations = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000002.SZ", primary_theme="CPO光通信", confidence=0.9),
        ]

        results = calculate_theme_scores(
            trade_date="20260912",
            limit_stocks=[],
            strong_stocks=strong_stocks,
            normalizations=normalizations,
            weights={"Breadth": 1.0},
            top_n=1,
        )

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
