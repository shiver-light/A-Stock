from __future__ import annotations

import unittest

import pandas as pd

from analysis.efficiency_report import build_efficiency_report, render_efficiency_report_text


class EfficiencyReportTestCase(unittest.TestCase):
    def test_build_efficiency_report_identifies_improving_structure(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y%m%d"),
                "ts_code": ["000001.SZ"] * 10,
                "pre_close": [10.0] * 10,
                "close": [9.8, 10.2, 9.9, 10.3, 9.95, 10.4, 9.98, 10.8, 9.995, 11.0],
                "high": [10.1, 10.4, 10.0, 10.5, 10.0, 10.6, 10.0, 11.0, 10.0, 11.2],
                "low": [9.6, 9.9, 9.7, 10.0, 9.8, 10.1, 9.9, 10.4, 9.9, 10.6],
                "vol": [100, 110, 90, 120, 80, 130, 70, 150, 65, 160],
                "turnover_rate_f": [10.0, 10.0, 8.0, 10.0, 6.0, 8.0, 4.0, 8.0, 3.0, 7.0],
            }
        )

        report = build_efficiency_report(data, stock_label="平安银行/000001.SZ")

        self.assertEqual(report["stock"], "平安银行/000001.SZ")
        self.assertEqual(report["up_efficiency_trend"], "提高")
        self.assertEqual(report["down_efficiency_trend"], "下降")
        self.assertEqual(report["long_short_structure"], "改善")
        self.assertIn("上涨效率提高", report["final_conclusion"])

    def test_render_efficiency_report_text_uses_expected_sections(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y%m%d"),
                "ts_code": ["000001.SZ"] * 10,
                "pre_close": [10.0] * 10,
                "close": [10.1] * 10,
                "high": [10.2] * 10,
                "low": [9.8] * 10,
                "vol": [100.0] * 10,
                "turnover_rate_f": [5.0] * 10,
            }
        )

        report = build_efficiency_report(data, stock_label="000001.SZ")
        text = render_efficiency_report_text(report)

        self.assertIn("【最近10日上涨/下跌效率报告】", text)
        self.assertIn("股票：000001.SZ", text)
        self.assertIn("最终结论：", text)


if __name__ == "__main__":
    unittest.main()
