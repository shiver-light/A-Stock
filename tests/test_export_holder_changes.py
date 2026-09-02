from __future__ import annotations

import unittest

import pandas as pd

from scripts.export_holder_changes import normalize_holdernumber_data, summarize_latest_holder_changes


class ExportHolderChangesTestCase(unittest.TestCase):
    def test_summarize_latest_holder_changes_uses_previous_disclosure(self) -> None:
        holder_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
                "ann_date": ["20260601", "20260301", "20260501", "20250201"],
                "end_date": ["20260531", "20260228", "20250430", "20250131"],
                "holder_num": [1200, 1000, 1800, 2000],
            }
        )
        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "name": ["平安银行", "浦发银行"],
                "industry": ["银行", "银行"],
                "market": ["主板", "主板"],
            }
        )

        result = summarize_latest_holder_changes(holder_data, stock_basic=stock_basic)

        first = result.loc[result["ts_code"] == "000001.SZ"].iloc[0]
        self.assertEqual(first["holder_num"], 1200)
        self.assertEqual(first["previous_holder_num"], 1000)
        self.assertEqual(first["holder_num_change"], 200)
        self.assertAlmostEqual(first["holder_num_change_ratio"], 0.2)

        second = result.loc[result["ts_code"] == "600000.SH"].iloc[0]
        self.assertEqual(second["holder_num_change"], -200)
        self.assertAlmostEqual(second["holder_num_change_ratio"], -0.1)

    def test_normalize_holdernumber_data_handles_empty_input(self) -> None:
        result = normalize_holdernumber_data(pd.DataFrame())

        self.assertEqual(result.columns.tolist(), ["ts_code", "ann_date", "end_date", "holder_num"])
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
