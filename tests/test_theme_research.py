from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from research.theme_research import (
    build_and_write_technology_theme_research,
    build_and_write_taxonomy_theme_research,
    build_and_write_theme_validation_config,
    build_technology_theme_research_config,
    build_theme_validation_config,
    build_theme_pool_from_tags,
    build_theme_stock_pool,
    build_theme_tags,
    filter_theme_tags,
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

    def test_build_theme_tags_outputs_json_friendly_tag_rows(self) -> None:
        stock_basic = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "name": ["光模块A", "银行B"],
                "industry": ["通信设备", "银行"],
                "market": ["主板", "主板"],
                "exchange": ["SZSE", "SZSE"],
                "list_date": ["20200101", "20200101"],
                "delist_date": ["", ""],
            }
        )
        taxonomy = {
            "themes": [
                {
                    "name": "AI算力",
                    "sub_themes": [
                        {
                            "name": "CPO/光模块",
                            "confidence": 0.8,
                            "industry_keywords": ["通信设备"],
                            "name_keywords": ["光模块"],
                        }
                    ],
                }
            ]
        }

        result = build_theme_tags(stock_basic, taxonomy, as_of_date="20250101")

        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(result.iloc[0]["theme"], "AI算力")
        self.assertEqual(result.iloc[0]["sub_theme"], "CPO/光模块")
        self.assertEqual(result.iloc[0]["source"], "stock_basic_keyword")
        self.assertIn("industry:通信设备", result.iloc[0]["evidence"])
        self.assertIn("name:光模块", result.iloc[0]["evidence"])

    def test_filter_theme_tags_and_build_pool(self) -> None:
        tags = pd.DataFrame(
            {
                "as_of_date": ["20250101", "20250101", "20250101"],
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "name": ["A", "A", "B"],
                "industry": ["通信设备", "通信设备", "银行"],
                "market": ["主板", "主板", "主板"],
                "exchange": ["SZSE", "SZSE", "SZSE"],
                "theme": ["AI算力", "AI算力", "金融"],
                "sub_theme": ["CPO/光模块", "PCB/服务器/高速连接", "银行"],
                "source": ["stock_basic_keyword"] * 3,
                "confidence": [0.8, 0.7, 0.9],
                "evidence": ["industry:通信设备", "industry:通信设备", "industry:银行"],
                "valid_from": ["20250101"] * 3,
                "valid_to": [""] * 3,
            }
        )

        filtered = filter_theme_tags(tags, themes=["AI算力"], min_confidence=0.75)
        pool = build_theme_pool_from_tags(filtered)

        self.assertEqual(filtered["sub_theme"].tolist(), ["CPO/光模块"])
        self.assertEqual(pool["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(pool.iloc[0]["match_reason"], "CPO/光模块")

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

    @patch("research.theme_research.get_universe")
    @patch("research.theme_research.get_stock_basic_history")
    def test_build_and_write_taxonomy_theme_research_writes_tags_pool_and_config(
        self,
        mock_stock_basic,
        mock_get_universe,
    ) -> None:
        mock_stock_basic.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["光模块A", "银行B", "芯片C"],
                "industry": ["通信设备", "银行", "半导体"],
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
            taxonomy_path = Path(tmp_dir) / "taxonomy.yaml"
            taxonomy_path.write_text(
                yaml.safe_dump(
                    {
                        "themes": [
                            {
                                "name": "AI算力",
                                "sub_themes": [
                                    {
                                        "name": "CPO/光模块",
                                        "confidence": 0.8,
                                        "industry_keywords": ["通信设备"],
                                        "name_keywords": ["光模块"],
                                    }
                                ],
                            },
                            {
                                "name": "半导体国产替代",
                                "sub_themes": [
                                    {
                                        "name": "芯片设计/功率半导体",
                                        "confidence": 0.8,
                                        "industry_keywords": ["半导体"],
                                        "name_keywords": ["芯片"],
                                    }
                                ],
                            },
                        ]
                    },
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config_path = Path(tmp_dir) / "theme.yaml"
            pool_path = Path(tmp_dir) / "pool.csv"
            tags_path = Path(tmp_dir) / "tags.csv"
            result = build_and_write_taxonomy_theme_research(
                taxonomy_path=taxonomy_path,
                start_date="20250101",
                end_date="20260608",
                output_config_path=config_path,
                output_pool_path=pool_path,
                output_tags_path=tags_path,
                themes=["AI算力"],
                min_confidence=0.7,
            )

            self.assertEqual(result["tag_count"], 1)
            self.assertEqual(result["pool_counts"], {"hs300": 1, "zz500": 0})
            tags = pd.read_csv(tags_path)
            self.assertEqual(tags["theme"].tolist(), ["AI算力"])
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["global"]["selected_themes"], ["AI算力"])
            self.assertEqual(config["global"]["theme_tags_csv"], str(tags_path))

    def test_build_theme_validation_config_clones_selected_models_across_windows(self) -> None:
        base_config = {
            "global": {
                "start_date": "20250101",
                "end_date": "20260608",
                "backtest_config": {"slippage_bps": 15.0},
            },
            "experiments": [
                {
                    "name": "model_a",
                    "universe_name": "hs300",
                    "ts_codes": ["000001.SZ"],
                    "top_n": 10,
                    "factor_config": {"return_60d": 1.0},
                },
                {
                    "name": "model_b",
                    "universe_name": "zz500",
                    "ts_codes": ["000002.SZ"],
                    "top_n": 20,
                    "factor_config": {"amount_mean_20d": 1.0},
                },
            ],
        }

        result = build_theme_validation_config(
            base_config,
            selected_model_names=["model_a", "model_b"],
            windows=[
                {"name": "w1", "start_date": "20250101", "end_date": "20250630"},
                {"name": "w2", "start_date": "20250701", "end_date": "20251231"},
            ],
        )

        self.assertEqual(len(result["experiments"]), 4)
        self.assertEqual(result["global"]["selected_validation_models"], ["model_a", "model_b"])
        self.assertEqual(result["experiments"][0]["name"], "model_a_w1")
        self.assertEqual(result["experiments"][0]["start_date"], "20250101")
        self.assertEqual(result["experiments"][0]["end_date"], "20250630")
        self.assertEqual(result["experiments"][0]["validation_base_model"], "model_a")
        self.assertEqual(result["experiments"][0]["factor_config"], {"return_60d": 1.0})

    def test_build_theme_validation_config_rejects_missing_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            build_theme_validation_config(
                {"global": {}, "experiments": [{"name": "model_a"}]},
                selected_model_names=["missing"],
            )

    def test_build_and_write_theme_validation_config_writes_yaml(self) -> None:
        base_config = {
            "global": {"start_date": "20250101", "end_date": "20260608"},
            "experiments": [
                {
                    "name": "model_a",
                    "universe_name": "hs300",
                    "ts_codes": ["000001.SZ"],
                    "top_n": 10,
                    "factor_config": {"return_60d": 1.0},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.yaml"
            output_path = Path(tmp_dir) / "validation.yaml"
            base_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
            result = build_and_write_theme_validation_config(
                base_config_path=base_path,
                output_config_path=output_path,
                selected_model_names=["model_a"],
                windows=[{"name": "w1", "start_date": "20250101", "end_date": "20250630"}],
            )

            self.assertEqual(result["experiment_count"], 1)
            config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(config["experiments"][0]["name"], "model_a_w1")


if __name__ == "__main__":
    unittest.main()
