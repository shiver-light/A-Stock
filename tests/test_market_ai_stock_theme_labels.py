from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_ai.models import ThemeNormalization
from market_ai.themes import load_stock_theme_labels, merge_theme_normalizations


class StockThemeLabelsTestCase(unittest.TestCase):
    def test_load_labels_filters_by_date_codes_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                {
                    "trade_date": ["20260911", "20260910", ""],
                    "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                    "primary_theme": ["AI算力", "CPO光通信", "PCB服务器液冷电源"],
                    "secondary_themes": ["AIDC|液冷", "", "PCB"],
                    "related_theme": ["数据中心", "", "AI算力"],
                    "source": ["manual", "manual", "llm"],
                    "confidence": [0.9, 0.95, 0.6],
                }
            ).to_csv(path, index=False)

            labels = load_stock_theme_labels(
                path,
                trade_date="20260911",
                stock_codes=["000001.SZ", "000002.SZ", "000003.SZ"],
                min_confidence=0.7,
            )

        self.assertEqual([label.stock_code for label in labels], ["000001.SZ"])
        self.assertEqual(labels[0].primary_theme, "AI算力")
        self.assertEqual(labels[0].secondary_themes, ["AIDC", "液冷"])
        self.assertEqual(labels[0].related_entities, ["数据中心"])

    def test_effective_window_supports_long_lived_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                {
                    "stock_code": ["000001.SZ", "000002.SZ"],
                    "primary_theme": ["AI算力", "CPO光通信"],
                    "effective_start": ["20260101", "20270101"],
                    "effective_end": ["20261231", ""],
                    "confidence": [0.9, 0.9],
                }
            ).to_csv(path, index=False)

            labels = load_stock_theme_labels(
                path,
                trade_date="20260911",
                stock_codes=["000001.SZ", "000002.SZ"],
            )

        self.assertEqual([label.stock_code for label in labels], ["000001.SZ"])

    def test_preferred_labels_override_rule_labels_by_stock_code(self) -> None:
        fallback = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="通信设备", confidence=0.8, source="rule")
        ]
        preferred = [
            ThemeNormalization(stock_code="000001.SZ", primary_theme="AI算力", confidence=0.9, source="manual")
        ]

        merged = merge_theme_normalizations(preferred, fallback)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].primary_theme, "AI算力")
        self.assertEqual(merged[0].source, "manual")

    def test_missing_required_columns_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame({"stock_code": ["000001.SZ"]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_stock_theme_labels(path, trade_date="20260911", stock_codes=["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
