from __future__ import annotations

import unittest

import pandas as pd

from analysis.data_checks import (
    check_duplicate_keys,
    check_factor_signal_alignment,
    check_missing_ratio_by_date,
    check_universe_stability,
)


class DataChecksTestCase(unittest.TestCase):
    def test_check_duplicate_keys_reports_duplicates(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240102"],
                "ts_code": ["A", "A", "B"],
                "factor_value": [1.0, 2.0, 3.0],
            }
        )

        result = check_duplicate_keys(data, ["trade_date", "ts_code"])

        self.assertTrue(result["has_duplicates"])
        self.assertEqual(result["duplicate_row_count"], 2)
        self.assertEqual(result["duplicate_group_count"], 1)

    def test_check_missing_ratio_by_date_tracks_sample_change_and_missing_ratio(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103"],
                "ts_code": ["A", "B", "A"],
                "factor_value": [1.0, None, None],
            }
        )

        result = check_missing_ratio_by_date(data, ["factor_value"])

        self.assertEqual(result["date_count"], 2)
        self.assertEqual(result["per_date"][0]["sample_count"], 2)
        self.assertEqual(result["per_date"][1]["sample_count_change"], -1)
        self.assertEqual(result["column_summary"][0]["anomaly_dates"], ["20240103"])

    def test_check_factor_signal_alignment_flags_overlap_and_missing_signal_dates(self) -> None:
        factor_df = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103"],
                "ts_code": ["A", "A"],
                "factor_value": [1.0, 2.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "signal_date": ["20240101", "20240102"],
                "execution_date": ["20240102", "20240103"],
            }
        )

        result = check_factor_signal_alignment(factor_df, schedule)

        self.assertTrue(result["has_alignment_risk"])
        self.assertEqual(result["overlap_with_execution_dates"], ["20240102", "20240103"])
        self.assertEqual(result["missing_signal_dates"], ["20240101"])

    def test_check_universe_stability_supports_as_of_date_history(self) -> None:
        universe_history = pd.DataFrame(
            {
                "as_of_date": ["20240131", "20240131", "20240229", "20240229"],
                "ts_code": ["A", "B", "B", "C"],
            }
        )

        result = check_universe_stability(universe_history)

        self.assertEqual(result["date_col"], "as_of_date")
        self.assertEqual(result["date_count"], 2)
        self.assertAlmostEqual(result["per_date"][1]["turnover_ratio"], 1.0)
        self.assertEqual(result["per_date"][1]["additions"], ["C"])
        self.assertEqual(result["per_date"][1]["removals"], ["A"])


if __name__ == "__main__":
    unittest.main()
