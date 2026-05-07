from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.daily_archive import archive_daily_consensus_recommendations, resolve_next_trading_day


class FakeCalendarService:
    def __init__(self, calendar: pd.DataFrame) -> None:
        self.calendar = calendar

    def get_trade_calendar(self, *, start_date: str, end_date: str, refresh: bool = False, exchange: str = "SSE") -> pd.DataFrame:
        mask = (self.calendar["cal_date"] >= start_date) & (self.calendar["cal_date"] <= end_date)
        return self.calendar.loc[mask].reset_index(drop=True)


class DailyArchiveTestCase(unittest.TestCase):
    def test_resolve_next_trading_day_returns_next_open_date(self) -> None:
        calendar = pd.DataFrame(
            {
                "exchange": ["SSE", "SSE", "SSE"],
                "cal_date": ["20260507", "20260508", "20260509"],
                "is_open": ["1", "1", "0"],
                "pretrade_date": ["20260506", "20260507", "20260508"],
            }
        )

        result = resolve_next_trading_day("20260507", calendar_service=FakeCalendarService(calendar))

        self.assertEqual(result, "20260508")

    def test_resolve_next_trading_day_returns_none_for_closed_signal_date(self) -> None:
        calendar = pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20260509", "20260511"],
                "is_open": ["0", "1"],
                "pretrade_date": ["20260508", "20260508"],
            }
        )

        result = resolve_next_trading_day("20260509", calendar_service=FakeCalendarService(calendar))

        self.assertIsNone(result)

    @patch("research.daily_archive.generate_daily_consensus_recommendations")
    def test_archive_daily_consensus_recommendations_writes_date_folder(self, mock_generate) -> None:
        mock_generate.return_value = {
            "run_dir": "demo_run",
            "as_of_date": "20260507",
            "model_roles": {
                "core_models": ["core"],
                "confirm_models": ["confirm"],
                "watch_models": [],
            },
            "selected_models": [],
            "trade_consensus": [
                {
                    "ts_code": "000001.SZ",
                    "recommend_level": "A",
                    "consensus_count": 2,
                    "source_models": ["core", "confirm"],
                    "best_rank": 1,
                    "avg_rank": 1.5,
                    "avg_score": 0.9,
                    "in_core_model": True,
                    "in_confirm_model": True,
                    "in_watch_model": False,
                }
            ],
            "trade_core": [],
            "watch_list": [],
            "all_recommendations": [],
        }
        calendar = pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20260507", "20260508"],
                "is_open": ["1", "1"],
                "pretrade_date": ["20260506", "20260507"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = archive_daily_consensus_recommendations(
                signal_date="20260507",
                archive_dir=tmp_dir,
                profiles={
                    "hs300": {
                        "run_dir": "demo_run",
                        "core_models": ["core"],
                        "confirm_models": ["confirm"],
                        "watch_models": [],
                        "mode": "trade",
                    }
                },
                calendar_service=FakeCalendarService(calendar),
            )

            target_dir = Path(tmp_dir) / "20260508"
            self.assertEqual(result["status"], "completed")
            self.assertTrue((target_dir / "hs300.json").exists())
            self.assertTrue((target_dir / "hs300.txt").exists())
            self.assertTrue((target_dir / "manifest.json").exists())
            self.assertTrue((target_dir / "summary.txt").exists())
            manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["signal_date"], "20260507")
            self.assertEqual(manifest["target_trade_date"], "20260508")
            self.assertEqual(manifest["universes"]["hs300"]["trade_consensus_count"], 1)

    def test_archive_daily_consensus_recommendations_skips_closed_signal_date(self) -> None:
        calendar = pd.DataFrame(
            {
                "exchange": ["SSE", "SSE"],
                "cal_date": ["20260509", "20260511"],
                "is_open": ["0", "1"],
                "pretrade_date": ["20260508", "20260508"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = archive_daily_consensus_recommendations(
                signal_date="20260509",
                archive_dir=tmp_dir,
                profiles={},
                calendar_service=FakeCalendarService(calendar),
            )

            self.assertEqual(result["status"], "skipped")
            self.assertFalse(any(Path(tmp_dir).iterdir()))


if __name__ == "__main__":
    unittest.main()
