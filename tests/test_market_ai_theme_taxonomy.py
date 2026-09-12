from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from market_ai.themes import ThemeDefinition, ThemeTaxonomy, load_theme_taxonomy


class ThemeTaxonomyTestCase(unittest.TestCase):
    def test_load_default_taxonomy_and_match_aliases(self) -> None:
        taxonomy = load_theme_taxonomy()

        matches = taxonomy.match_text("AI服务器订单增长，CPO光模块需求提升")

        self.assertEqual(matches[0].theme, "AI算力")
        self.assertIn("CPO光通信", {match.theme for match in matches})

    def test_best_match_returns_none_for_unknown_text(self) -> None:
        taxonomy = ThemeTaxonomy([ThemeDefinition(theme="AI算力", aliases=("AIDC",))])

        self.assertIsNone(taxonomy.best_match("白酒消费复苏"))

    def test_duplicate_alias_across_themes_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicated"):
            ThemeTaxonomy(
                [
                    ThemeDefinition(theme="AI算力", aliases=("数据中心",)),
                    ThemeDefinition(theme="IDC电源", aliases=("数据中心",)),
                ]
            )

    def test_load_custom_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theme_taxonomy.yaml"
            path.write_text(
                """
themes:
  - theme: AI算力
    parent: 科技
    aliases:
      - AIDC
      - GPU服务器
""",
                encoding="utf-8",
            )

            taxonomy = load_theme_taxonomy(path)

        match = taxonomy.best_match("AIDC 建设提速")
        self.assertIsNotNone(match)
        self.assertEqual(match.theme, "AI算力")
        self.assertEqual(match.parent, "科技")


if __name__ == "__main__":
    unittest.main()
