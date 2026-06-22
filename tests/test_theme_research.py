from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from research.theme_research import (
    build_and_write_technology_theme_research,
    build_technology_theme_research_config,
    build_theme_stock_pool,
    intersect_theme_with_universe,
)


class ThemeResearchTestCase(unittest.TestCase):
    def test_build_theme_stock_pool_matches_industry_and_name_keywords(self) -> None:
        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["银行A", "光模块B", "机械C"],
                "industry": ["银行", "通信设备", "专用机械"],
                "market": ["主板", "主板", "主板"],
                "exchange": ["SZSE", "SZSE", "SZSE"],
                "list_date": ["20200101", "20200101", "20270101"],
                "delist_date": ["", "", ""],
            }
        )

        result = build_theme_stock_pool(
            stock_basic,
            as_of_date="20250101",
            industry_keywords=["通信设备"],
            name_keywords=["光模块"],
        )

        self.assertEqual(result["ts_code"].tolist(), ["000002.SZ"])
        self.assertIn("industry:通信设备", result.iloc[0]["match_reason"])
        self.assertIn("name:光模块", result.iloc[0]["match_reason"])

    def test_intersect_theme_with_universe_keeps_only_shared_codes(self) -> None:
        theme_pool = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["A", "B"],
                "industry": ["半导体", "通信设备"],
                "market": ["主板", "主板"],
                "exchange": ["SZSE", "SZSE"],
                "match_reason": ["industry:半导体", "industry:通信设备"],
            }
        )
        universe = pd.DataFrame({"ts_code": ["000002.SZ", "000003.SZ"]})

        result = intersect_theme_with_universe(theme_pool, universe, "zz500")

        self.assertEqual(result["universe_name"].tolist(), ["zz500"])
        self.assertEqual(result["ts_code"].tolist(), ["000002.SZ"])

    def test_build_technology_theme_research_config_uses_static_ts_codes(self) -> None:
        config = build_technology_theme_research_config(
            pools_by_universe={
                "hs300": ["000001.SZ", "000002.SZ"],
                "zz500": ["000003.SZ"],
            },
            start_date="20250101",
            end_date="20260608",
            top_ns=(10,),
        )

        self.assertEqual(len(config["experiments"]), 6)
        first = config["experiments"][0]
        self.assertEqual(first["benchmark_code"], "000300.SH")
        self.assertEqual(first["ts_codes"], ["000001.SZ", "000002.SZ"])
        self.assertIn("signal_filters", first)

    @patch("research.theme_research.get_universe")
    @patch("research.theme_research.get_stock_basic_history")
    def test_build_and_write_technology_theme_research_writes_outputs(self, mock_stock_basic, mock_get_universe) -> None:
        mock_stock_basic.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["芯片A", "银行B", "通信C"],
                "industry": ["半导体", "银行", "通信设备"],
                "market": ["主板", "主板", "主板"],
                "exchange": ["SZSE", "SZSE", "SZSE"],
                "list_date": ["20200101", "20200101", "20200101"],
                "delist_date": ["", "", ""],
            }
        )
        mock_get_universe.side_effect = [
            pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}),
            pd.DataFrame({"ts_code": ["000003.SZ"]}),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "theme.yaml"
            pool_path = Path(tmp_dir) / "pool.csv"
            result = build_and_write_technology_theme_research(
                start_date="20250101",
                end_date="20260608",
                output_config_path=config_path,
                output_pool_path=pool_path,
                industry_keywords=["半导体", "通信设备"],
                name_keywords=["芯片"],
            )

            self.assertEqual(result["pool_counts"], {"hs300": 1, "zz500": 1})
            self.assertTrue(config_path.exists())
            self.assertTrue(pool_path.exists())

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["global"]["pool_as_of_date"], "20250101")
            self.assertEqual(len(config["experiments"]), 18)
            pool = pd.read_csv(pool_path)
            self.assertEqual(set(pool["ts_code"].astype(str)), {"000001.SZ", "000003.SZ"})


if __name__ == "__main__":
    unittest.main()
