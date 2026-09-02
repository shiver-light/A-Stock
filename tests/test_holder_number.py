from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.holder_number import AShareHolderNumberService


class FakeHolderNumberClient:
    def __init__(self) -> None:
        self.holder_calls = 0
        self.calendar_calls = 0

    def stk_holdernumber(self, **kwargs) -> pd.DataFrame:
        self.holder_calls += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20250110", "20250410"],
                "end_date": ["20241231", "20250331"],
                "holder_num": [1000, 800],
            }
        )

    def trade_cal(self, **kwargs) -> pd.DataFrame:
        self.calendar_calls += 1
        dates = pd.date_range(kwargs["start_date"], kwargs["end_date"], freq="D").strftime("%Y%m%d").tolist()
        return pd.DataFrame(
            {
                "exchange": ["SSE"] * len(dates),
                "cal_date": dates,
                "is_open": ["1"] * len(dates),
                "pretrade_date": [""] * len(dates),
            }
        )


class HolderNumberTestCase(unittest.TestCase):
    def test_align_holder_number_to_daily_uses_announcement_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AShareHolderNumberService(client=FakeHolderNumberClient(), cache_dir=Path(directory))

            result = service.align_holder_number_to_daily(
                ts_code="000001.SZ",
                start_date="20250408",
                end_date="20250412",
            )

        before_announcement = result.loc[result["trade_date"] == "20250409"].iloc[0]
        on_announcement = result.loc[result["trade_date"] == "20250410"].iloc[0]
        self.assertTrue(pd.isna(before_announcement["holder_num_change_ratio_negative"]))
        self.assertEqual(on_announcement["holder_num"], 800)
        self.assertAlmostEqual(on_announcement["holder_num_change_ratio"], -0.2)
        self.assertAlmostEqual(on_announcement["holder_num_change_ratio_negative"], 0.2)
        self.assertEqual(on_announcement["holder_report_lag_trading_days"], 10)

    def test_report_lag_trading_days_counts_end_to_announcement_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AShareHolderNumberService(client=FakeHolderNumberClient(), cache_dir=Path(directory))
            calendar = pd.DataFrame(
                {
                    "exchange": ["SSE"] * 5,
                    "cal_date": ["20250331", "20250401", "20250402", "20250403", "20250404"],
                    "is_open": ["1"] * 5,
                    "pretrade_date": [""] * 5,
                }
            )
            disclosures = pd.DataFrame(
                {
                    "ann_date": ["20250402", "20250404"],
                    "end_date": ["20250331", "20250331"],
                }
            )

            result = service._report_lag_trading_days(disclosures, calendar)

        self.assertEqual(result.tolist(), [2, 4])

    def test_announcement_age_trading_days_counts_from_announcement_to_signal_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AShareHolderNumberService(client=FakeHolderNumberClient(), cache_dir=Path(directory))
            calendar = pd.DataFrame(
                {
                    "exchange": ["SSE"] * 4,
                    "cal_date": ["20250410", "20250411", "20250414", "20250415"],
                    "is_open": ["1"] * 4,
                    "pretrade_date": [""] * 4,
                }
            )
            aligned = pd.DataFrame(
                {
                    "trade_date": ["20250410", "20250411", "20250415"],
                    "ann_date": ["20250410", "20250410", "20250410"],
                }
            )

            result = service._announcement_age_trading_days(aligned, calendar)

        self.assertEqual(result.tolist(), [0, 1, 3])

    def test_holder_number_cache_metadata_prevents_repeat_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeHolderNumberClient()
            service = AShareHolderNumberService(client=client, cache_dir=Path(directory))

            service.get_holder_number_history(ts_code="000001.SZ", start_date="20250101", end_date="20250630")
            service.get_holder_number_history(ts_code="000001.SZ", start_date="20250201", end_date="20250531")

        self.assertEqual(client.holder_calls, 1)


if __name__ == "__main__":
    unittest.main()
