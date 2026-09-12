from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_ai.__main__ import _append_theme_score_history, _discover_stock_theme_label_files, _get_a_share_trade_dates, main
from market_ai.models import ThemeScoreResult
from market_ai.themes import ThemeDefinition, ThemeTaxonomy
from market_ai.themes.enrichment import enrich_stock_theme_labels


class FakeTradeCalendarClient:
    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None, fields=None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cal_date": ["20260902", "20260901"],
                "is_open": [1, 1],
            }
        )


class MarketAiCliTestCase(unittest.TestCase):
    def test_get_a_share_trade_dates_returns_sorted_open_dates(self) -> None:
        with patch("market_ai.__main__.TushareClient", return_value=FakeTradeCalendarClient()):
            dates = _get_a_share_trade_dates("20260901", "20260902")

        self.assertEqual(dates, ["20260901", "20260902"])

    def test_append_theme_score_history_upserts_by_trade_date_and_theme(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theme_scores.csv"
            _append_theme_score_history(
                path,
                [ThemeScoreResult(trade_date="20260901", theme="AI算力", score=50.0, rank=1, lifecycle_stage=1)],
            )
            _append_theme_score_history(
                path,
                [
                    ThemeScoreResult(
                        trade_date="20260901",
                        theme="AI算力",
                        score=60.0,
                        rank=1,
                        lifecycle_stage=2,
                    )
                ],
            )

            data = pd.read_csv(path)

        self.assertEqual(len(data), 1)
        self.assertEqual(float(data.iloc[0]["score"]), 60.0)
        self.assertEqual(int(data.iloc[0]["lifecycle_stage"]), 2)

    def test_enrich_theme_labels_filters_and_prefers_reason_rows(self) -> None:
        taxonomy = ThemeTaxonomy(
            [
                ThemeDefinition(theme="机器人具身智能", aliases=("机器人", "减速器")),
                ThemeDefinition(theme="AI算力", aliases=("算力", "数据中心")),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.csv"
            reason_path = Path(directory) / "reason.csv"
            output_path = Path(directory) / "labels.csv"
            pd.DataFrame(
                {
                    "trade_date": ["20260910", "20260911"],
                    "stock_code": ["000001.SZ", "000002.SZ"],
                    "stock_name": ["样本一", "样本二"],
                    "primary_theme": ["传统制造", "AI算力"],
                    "source": ["industry_fallback", "public_industry_rule"],
                    "confidence": [0.35, 0.7],
                }
            ).to_csv(base_path, index=False)
            pd.DataFrame(
                {
                    "trade_date": ["20260910", "20260912"],
                    "stock_code": ["000001.SZ", "000003.SZ"],
                    "stock_name": ["样本一", "样本三"],
                    "theme_reason": ["公告披露跨界机器人减速器业务", "算力中心订单"],
                    "source": ["announcement", "news"],
                    "evidence_url": ["https://example.com/a", "https://example.com/b"],
                }
            ).to_csv(reason_path, index=False)

            result = enrich_stock_theme_labels(
                start_date="20260910",
                end_date="20260911",
                output_file=output_path,
                taxonomy=taxonomy,
                base_label_paths=[base_path],
                reason_csv_paths=[reason_path],
            )

        self.assertEqual(len(result), 2)
        first = result.loc[result["stock_code"].eq("000001.SZ")].iloc[0]
        self.assertEqual(first["primary_theme"], "机器人具身智能")
        self.assertEqual(first["source"], "announcement")
        self.assertGreater(float(first["confidence"]), 0.8)

    def test_enrich_themes_cli_writes_empty_file_without_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "labels.csv"
            code = main(
                [
                    "enrich-themes",
                    "--start-date",
                    "20260910",
                    "--end-date",
                    "20260911",
                    "--output-file",
                    str(output_path),
                ]
            )
            data = pd.read_csv(output_path)

        self.assertEqual(code, 0)
        self.assertEqual(list(data.columns)[:2], ["trade_date", "stock_code"])
        self.assertTrue(data.empty)

    def test_discover_stock_theme_label_files_under_input_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "market_radar"
            label_dir = input_dir / "theme_labels_2026"
            label_dir.mkdir(parents=True)
            expected = label_dir / "limit_up_stock_theme_labels_2026.csv"
            expected.write_text("stock_code,primary_theme\n000001.SZ,AI算力\n", encoding="utf-8")
            ignored = input_dir / "other.csv"
            ignored.write_text("stock_code,primary_theme\n000002.SZ,CPO光通信\n", encoding="utf-8")

            files = _discover_stock_theme_label_files(input_dir)

        self.assertEqual(files, [expected])


if __name__ == "__main__":
    unittest.main()
