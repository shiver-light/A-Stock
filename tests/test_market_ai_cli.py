from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_ai.__main__ import (
    _append_theme_daily_snapshot,
    _append_theme_score_history,
    _discover_stock_theme_label_files,
    _get_a_share_trade_dates,
    main,
    summarize_theme_daily_snapshot,
)
from market_ai.models import DailyRadarReport, ThemeCatalystSummary, ThemeScoreResult
from market_ai.providers.news.importer import import_news_csvs
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

    def test_append_theme_daily_snapshot_upserts_theme_catalysts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theme_daily_snapshot.csv"
            report = DailyRadarReport(
                trade_date="20260901",
                core_themes=[
                    ThemeScoreResult(
                        trade_date="20260901",
                        theme="AI算力",
                        score=50.0,
                        rank=1,
                        lifecycle_stage=1,
                        components={"LimitUpStrength": 80.0, "NewsCatalyst": 60.0},
                        reasons=["涨停/触板股票 3 只"],
                    )
                ],
                theme_catalysts=[
                    ThemeCatalystSummary(
                        trade_date="20260901",
                        theme="AI算力",
                        primary_event="算力政策发布",
                        confirmed_event_count=1,
                        related_event_count=2,
                        conclusion="消息与资金形成确认。",
                    )
                ],
            )
            _append_theme_daily_snapshot(path, report)
            report.core_themes[0].score = 65.0
            _append_theme_daily_snapshot(path, report)

            data = pd.read_csv(path)

        self.assertEqual(len(data), 1)
        row = data.iloc[0]
        self.assertEqual(row["theme"], "AI算力")
        self.assertEqual(float(row["score"]), 65.0)
        self.assertEqual(int(row["confirmed_event_count"]), 1)
        self.assertEqual(row["primary_event"], "算力政策发布")
        self.assertEqual(float(row["limit_up_strength"]), 80.0)

    def test_summarize_theme_daily_snapshot_uses_as_of_and_lookback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theme_daily_snapshot.csv"
            pd.DataFrame(
                {
                    "trade_date": ["20260908", "20260909", "20260910", "20260911", "20260912"],
                    "theme": ["AI算力", "AI算力", "机器人具身智能", "AI算力", "AI算力"],
                    "rank": [2, 1, 2, 1, 1],
                    "score": [50.0, 70.0, 45.0, 80.0, 10.0],
                    "lifecycle_stage": [1, 2, 1, 3, 7],
                    "confirmed_event_count": [0, 1, 0, 2, 0],
                    "unconfirmed_event_count": [1, 0, 0, 1, 0],
                    "related_event_count": [1, 1, 0, 3, 0],
                    "primary_event": ["", "算力政策", "", "算力订单", ""],
                    "catalyst_conclusion": ["", "确认", "", "确认", ""],
                }
            ).to_csv(path, index=False)

            summary = summarize_theme_daily_snapshot(path, as_of_date="20260911", lookback_days=2, top_n=10)

        self.assertEqual(summary.iloc[0]["theme"], "AI算力")
        self.assertEqual(summary.iloc[0]["latest_trade_date"], "20260911")
        self.assertEqual(int(summary.iloc[0]["latest_stage"]), 3)
        self.assertEqual(int(summary.iloc[0]["active_days"]), 1)
        self.assertEqual(float(summary.iloc[0]["latest_score"]), 80.0)
        self.assertNotIn("20260912", set(summary["latest_trade_date"]))

    def test_theme_summary_cli_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theme_daily_snapshot.csv"
            pd.DataFrame(
                {
                    "trade_date": ["20260911"],
                    "theme": ["AI算力"],
                    "rank": [1],
                    "score": [80.0],
                    "lifecycle_stage": [3],
                }
            ).to_csv(path, index=False)

            code = main(["theme-summary", "--snapshot", str(path), "--output", "json"])

        self.assertEqual(code, 0)

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

    def test_import_news_csvs_normalizes_aliases_and_filters_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw_news.csv"
            output_path = Path(directory) / "news.csv"
            pd.DataFrame(
                {
                    "来源": ["财联社", "东方财富", ""],
                    "标题": ["PCB板块多股涨停", "窗口外消息", "机器人公告催化"],
                    "发布时间": [
                        "2026-09-11 15:20:00",
                        "2026-09-05 09:00:00",
                        "2026-09-10T08:30:00+08:00",
                    ],
                    "链接": ["https://example.com/1", "https://example.com/2", ""],
                    "摘要": ["AI服务器、液冷方向活跃", "旧消息", "减速器订单增长"],
                }
            ).to_csv(input_path, index=False)

            result = import_news_csvs(
                [input_path],
                output_file=output_path,
                start_date="20260911",
                end_date="20260911",
                lookback_hours=36,
                default_source="manual",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["source"]), {"财联社", "manual"})
        self.assertTrue(result["news_id"].str.startswith("news_").all())
        self.assertTrue(result["published_at"].str.endswith("+08:00").all())

    def test_import_news_cli_writes_normalized_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw_news.csv"
            output_path = Path(directory) / "news.csv"
            pd.DataFrame(
                {
                    "source": ["cls"],
                    "title": ["液冷服务器需求提升"],
                    "published_at": ["2026-09-11T15:30:00+08:00"],
                }
            ).to_csv(input_path, index=False)

            code = main(
                [
                    "import-news",
                    "--start-date",
                    "20260911",
                    "--end-date",
                    "20260911",
                    "--input-csv",
                    str(input_path),
                    "--output-file",
                    str(output_path),
                ]
            )
            data = pd.read_csv(output_path)

        self.assertEqual(code, 0)
        self.assertEqual(len(data), 1)
        self.assertTrue({"news_id", "source", "title", "published_at", "url", "content"}.issubset(data.columns))
        self.assertTrue({"content_hash", "is_filtered", "filter_reason"}.issubset(data.columns))


if __name__ == "__main__":
    unittest.main()
