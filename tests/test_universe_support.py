from __future__ import annotations

import unittest
from unittest.mock import patch

from data import INDEX_CODE_MAP
from universe.provider import SUPPORTED_UNIVERSES, get_universe


class UniverseSupportTestCase(unittest.TestCase):
    def test_index_code_map_includes_zz2000(self) -> None:
        self.assertEqual(INDEX_CODE_MAP["zz2000"], "932000.CSI")

    def test_supported_universes_include_zz2000(self) -> None:
        self.assertIn("zz2000", SUPPORTED_UNIVERSES)

    def test_supported_universes_include_zz2000_ex_bj(self) -> None:
        self.assertIn("zz2000_ex_bj", SUPPORTED_UNIVERSES)

    def test_supported_universes_include_zz2000_ex_bj_ex_chinext_star(self) -> None:
        self.assertIn("zz2000_ex_bj_ex_chinext_star", SUPPORTED_UNIVERSES)

    def test_supported_universes_include_all_a_ex_chinext_st(self) -> None:
        self.assertIn("all_a_ex_chinext_st", SUPPORTED_UNIVERSES)

    def test_supported_universes_include_all_a_ex_chinext_star_st(self) -> None:
        self.assertIn("all_a_ex_chinext_star_st", SUPPORTED_UNIVERSES)

    @patch("universe.provider.get_stock_basic_history")
    def test_get_universe_all_a_ex_chinext_st_filters_chinext_and_st(self, mock_get_stock_basic_history) -> None:
        import pandas as pd

        mock_get_stock_basic_history.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600001.SH", "600002.SH"],
                "name": ["平安银行", "创业科技", "科创科技", "ST风险", "*ST退市"],
                "market": ["主板", "创业板", "科创板", "主板", "主板"],
                "exchange": ["SZSE", "SZSE", "SSE", "SSE", "SSE"],
                "list_date": ["20000101", "20000101", "20000101", "20000101", "20000101"],
                "delist_date": ["", "", "", "", ""],
            }
        )

        result = get_universe("all_a_ex_chinext_st", "20260501")

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "688001.SH"])
        self.assertTrue((result["universe_name"] == "all_a_ex_chinext_st").all())

    @patch("universe.provider.get_stock_basic_history")
    def test_get_universe_all_a_ex_chinext_star_st_filters_chinext_star_and_st(
        self,
        mock_get_stock_basic_history,
    ) -> None:
        import pandas as pd

        mock_get_stock_basic_history.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "300001.SZ", "301001.SZ", "688001.SH", "600001.SH", "002001.SZ"],
                "name": ["平安银行", "创业科技", "创业新股", "科创科技", "ST风险", "普通股票"],
                "market": ["主板", "创业板", "创业板", "科创板", "主板", "主板"],
                "exchange": ["SZSE", "SZSE", "SZSE", "SSE", "SSE", "SZSE"],
                "list_date": ["20000101", "20000101", "20000101", "20000101", "20000101", "20000101"],
                "delist_date": ["", "", "", "", "", ""],
            }
        )

        result = get_universe("all_a_ex_chinext_star_st", "20260501")

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "002001.SZ"])
        self.assertTrue((result["universe_name"] == "all_a_ex_chinext_star_st").all())

    @patch("universe.provider.get_index_constituents")
    def test_get_universe_zz2000_ex_bj_filters_bj_constituents(self, mock_get_index_constituents) -> None:
        import pandas as pd

        mock_get_index_constituents.return_value = pd.DataFrame(
            {
                "as_of_date": ["20260501", "20260501", "20260501"],
                "ts_code": ["000001.SZ", "600000.SH", "920001.BJ"],
                "weight": [0.4, 0.4, 0.2],
                "in_universe": [True, True, True],
            }
        )

        result = get_universe("zz2000_ex_bj", "20260501", include_weights=True)

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertTrue((result["universe_name"] == "zz2000_ex_bj").all())

    @patch("universe.provider.get_stock_basic_history")
    @patch("universe.provider.get_index_constituents")
    def test_get_universe_zz2000_ex_bj_ex_chinext_star_filters_growth_boards(
        self,
        mock_get_index_constituents,
        mock_get_stock_basic_history,
    ) -> None:
        import pandas as pd

        mock_get_index_constituents.return_value = pd.DataFrame(
            {
                "as_of_date": ["20260501"] * 5,
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600000.SH", "920001.BJ"],
                "weight": [0.2] * 5,
                "in_universe": [True] * 5,
            }
        )
        mock_get_stock_basic_history.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH", "600000.SH"],
                "market": ["主板", "创业板", "科创板", "主板"],
            }
        )

        result = get_universe("zz2000_ex_bj_ex_chinext_star", "20260501", include_weights=True)

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertTrue((result["universe_name"] == "zz2000_ex_bj_ex_chinext_star").all())


if __name__ == "__main__":
    unittest.main()
