from __future__ import annotations

import unittest

from market_ai.analysis import identify_stock_roles
from market_ai.models import LimitStock, StrongStock, ThemeNormalization, ThemeScoreResult


class MarketAiStockRolesTestCase(unittest.TestCase):
    def test_identify_stock_roles_assigns_space_capacity_and_catch_up(self) -> None:
        limit_stocks = [
            LimitStock(
                trade_date="20260911",
                stock_code="000001.SZ",
                stock_name="空间龙",
                pct_chg=10.0,
                amount=1000.0,
                turnover_rate=5.0,
                consecutive_limit_count=3,
                status="limit_up",
            ),
            LimitStock(
                trade_date="20260911",
                stock_code="000002.SZ",
                stock_name="容量中军",
                pct_chg=10.0,
                amount=5000.0,
                turnover_rate=4.0,
                consecutive_limit_count=1,
                status="limit_up",
            ),
            LimitStock(
                trade_date="20260911",
                stock_code="000003.SZ",
                stock_name="补涨股",
                pct_chg=10.0,
                amount=800.0,
                turnover_rate=3.0,
                consecutive_limit_count=1,
                status="limit_up",
            ),
        ]
        normalizations = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000002.SZ", primary_theme="AI算力", confidence=0.9),
            ThemeNormalization(stock_code="000003.SZ", primary_theme="AI算力", confidence=0.9),
        ]

        roles = identify_stock_roles(
            trade_date="20260911",
            core_themes=[ThemeScoreResult(trade_date="20260911", theme="AI算力", score=80.0, rank=1)],
            limit_stocks=limit_stocks,
            strong_stocks=[],
            normalizations=normalizations,
        )

        role_by_code = {role.stock_code: role.role for role in roles}
        self.assertEqual(role_by_code["000001.SZ"], "space_leader")
        self.assertEqual(role_by_code["000002.SZ"], "capacity_leader")
        self.assertEqual(role_by_code["000003.SZ"], "catch_up")

    def test_identify_stock_roles_assigns_elasticity_for_strong_stock(self) -> None:
        strong_stocks = [
            StrongStock(
                trade_date="20260911",
                stock_code="000001.SZ",
                stock_name="强势股",
                pct_chg=8.0,
                amount=1000.0,
            )
        ]
        roles = identify_stock_roles(
            trade_date="20260911",
            core_themes=[ThemeScoreResult(trade_date="20260911", theme="AI算力", score=50.0, rank=1)],
            limit_stocks=[],
            strong_stocks=strong_stocks,
            normalizations=[ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9)],
        )

        self.assertEqual(roles[0].role, "capacity_leader")
        self.assertIn("成交额最高", roles[0].reason[0])

    def test_identify_stock_roles_returns_empty_without_theme_map(self) -> None:
        roles = identify_stock_roles(
            trade_date="20260911",
            core_themes=[ThemeScoreResult(trade_date="20260911", theme="AI算力", score=50.0, rank=1)],
            limit_stocks=[],
            strong_stocks=[],
            normalizations=[],
        )

        self.assertEqual(roles, [])


if __name__ == "__main__":
    unittest.main()
