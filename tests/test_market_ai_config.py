from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from market_ai.config import (
    DEFAULT_NEWS_SEMANTIC_WEIGHTS,
    DEFAULT_THEME_SCORE_WEIGHTS,
    LLMConfig,
    NewsScoringConfig,
    ThemeScoreConfig,
    load_market_radar_config,
)


class MarketAiConfigTestCase(unittest.TestCase):
    def test_load_default_market_radar_config(self) -> None:
        config = load_market_radar_config()

        self.assertEqual(config.providers.market, "tushare")
        self.assertEqual(config.providers.news, ["local_csv"])
        self.assertEqual(config.news.lookback_hours, 36)
        self.assertEqual(config.news_scoring.default_authority_score, 50.0)
        self.assertEqual(config.news_scoring.semantic_weights, DEFAULT_NEWS_SEMANTIC_WEIGHTS)
        self.assertIn("财联社", config.news_scoring.authority_scores)
        self.assertFalse(config.llm.enabled)
        self.assertEqual(config.llm.provider, "none")
        self.assertEqual(config.top_theme_count, 5)
        self.assertEqual(config.theme_score.weights, DEFAULT_THEME_SCORE_WEIGHTS)

    def test_load_custom_market_radar_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radar.yaml"
            path.write_text(
                """
providers:
  market: fake
  news: [local_csv, announcements]
  llm: fake_llm
news:
  lookback_hours: 24
news_scoring:
  default_authority_score: 45
  authority_scores:
    财联社: 80
  semantic_weights:
    authority: 1
    directness: 1
theme_score:
  weights:
    LimitUpStrength: 2
    NewsCatalyst: 1
output:
  base_dir: tmp_runs
  report_dir: tmp_reports
  cache_dir: tmp_cache
strong_stock_min_pct_chg: 8
top_theme_count: 3
""",
                encoding="utf-8",
            )

            config = load_market_radar_config(path)

        self.assertEqual(config.providers.market, "fake")
        self.assertEqual(config.providers.news, ["local_csv", "announcements"])
        self.assertEqual(config.providers.llm, "fake_llm")
        self.assertFalse(config.llm.enabled)
        self.assertEqual(config.news.lookback_hours, 24)
        self.assertEqual(config.news_scoring.default_authority_score, 45.0)
        self.assertEqual(config.news_scoring.authority_scores["财联社"], 80.0)
        self.assertEqual(config.news_scoring.semantic_weights, {"authority": 1.0, "directness": 1.0})
        self.assertEqual(config.theme_score.normalized_weights(), {"LimitUpStrength": 2 / 3, "NewsCatalyst": 1 / 3})
        self.assertEqual(config.output.base_dir, "tmp_runs")
        self.assertEqual(config.strong_stock_min_pct_chg, 8.0)
        self.assertEqual(config.top_theme_count, 3)

    def test_theme_score_weights_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            ThemeScoreConfig(weights={"LimitUpStrength": -1.0})
        with self.assertRaisesRegex(ValueError, "sum must be positive"):
            ThemeScoreConfig(weights={"LimitUpStrength": 0.0})

    def test_news_scoring_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "default_authority_score"):
            NewsScoringConfig(default_authority_score=120)
        with self.assertRaisesRegex(ValueError, "authority_scores"):
            NewsScoringConfig(authority_scores={"财联社": -1})
        with self.assertRaisesRegex(ValueError, "semantic_weights"):
            NewsScoringConfig(semantic_weights={"authority": 0})

    def test_llm_config_accepts_ollama_openai(self) -> None:
        config = LLMConfig(
            enabled=True,
            provider="ollama_openai",
            base_url="http://192.168.85.248:11434/v1/",
            model="qwen3:8b",
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, "http://192.168.85.248:11434/v1")
        self.assertEqual(config.model, "qwen3:8b")

    def test_llm_config_rejects_unknown_enabled_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "ollama_openai"):
            LLMConfig(enabled=True, provider="unknown")

    def test_missing_config_file_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Market radar config file not found"):
            load_market_radar_config("missing_market_radar_config.yaml")


if __name__ == "__main__":
    unittest.main()
