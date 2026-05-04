from __future__ import annotations

import unittest

from data import INDEX_CODE_MAP
from universe.provider import SUPPORTED_UNIVERSES


class UniverseSupportTestCase(unittest.TestCase):
    def test_index_code_map_includes_zz2000(self) -> None:
        self.assertEqual(INDEX_CODE_MAP["zz2000"], "932000.CSI")

    def test_supported_universes_include_zz2000(self) -> None:
        self.assertIn("zz2000", SUPPORTED_UNIVERSES)


if __name__ == "__main__":
    unittest.main()
