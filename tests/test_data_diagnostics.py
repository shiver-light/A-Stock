from __future__ import annotations

import unittest

import pandas as pd

from analysis.data_diagnostics import (
    summarize_complete_case_count_by_date,
    summarize_factor_coverage,
    summarize_universe_count_by_date,
)


class DataDiagnosticsTestCase(unittest.TestCase):
    def test_summarize_functions_handle_empty_input(self) -> None:
        empty = pd.DataFrame(columns=["trade_date", "ts_code", "factor_a", "factor_b"])

        universe_summary = summarize_universe_count_by_date(empty)
        coverage_summary = summarize_factor_coverage(empty, factor_cols=["factor_a", "factor_b"])
        complete_case_summary = summarize_complete_case_count_by_date(empty, factor_cols=["factor_a", "factor_b"])

        self.assertEqual(universe_summary["date_count"], 0)
        self.assertEqual(coverage_summary["per_date"], [])
        self.assertEqual(complete_case_summary["per_date"], [])

    def test_summaries_capture_partial_missingness_and_count_changes(self) -> None:
        factor_data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102", "20240103", "20240104"],
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000001.SZ"],
                "factor_a": [1.0, None, 2.0, 3.0],
                "factor_b": [4.0, 5.0, None, 6.0],
            }
        )

        coverage_summary = summarize_factor_coverage(factor_data, factor_cols=["factor_a", "factor_b"])
        complete_case_summary = summarize_complete_case_count_by_date(
            factor_data,
            factor_cols=["factor_a", "factor_b"],
        )
        universe_summary = summarize_universe_count_by_date(factor_data)

        first_day_coverage = coverage_summary["per_date"][0]
        self.assertEqual(first_day_coverage["trade_date"], "20240102")
        self.assertEqual(first_day_coverage["universe_count"], 2)
        self.assertEqual(first_day_coverage["coverage_by_factor"]["factor_a"]["available_count"], 1)
        self.assertEqual(first_day_coverage["coverage_by_factor"]["factor_a"]["missing_count"], 1)
        self.assertAlmostEqual(first_day_coverage["coverage_by_factor"]["factor_a"]["coverage_ratio"], 0.5)

        first_day_complete = complete_case_summary["per_date"][0]
        self.assertEqual(first_day_complete["complete_case_count"], 1)
        self.assertEqual(first_day_complete["incomplete_case_count"], 1)
        self.assertAlmostEqual(first_day_complete["complete_case_ratio"], 0.5)

        self.assertEqual(universe_summary["per_date"][0]["universe_count"], 2)
        self.assertEqual(universe_summary["per_date"][1]["count_change"], -1)
        self.assertEqual(universe_summary["per_date"][-1]["universe_count"], 1)


if __name__ == "__main__":
    unittest.main()
