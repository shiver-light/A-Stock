from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_ai.__main__ import _append_theme_score_history, _get_a_share_trade_dates
from market_ai.models import ThemeScoreResult


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


if __name__ == "__main__":
    unittest.main()
