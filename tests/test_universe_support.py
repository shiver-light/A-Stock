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


if __name__ == "__main__":
    unittest.main()
