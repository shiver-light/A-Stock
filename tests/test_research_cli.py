from __future__ import annotations

import unittest

from research.__main__ import build_parser


class ResearchCliTestCase(unittest.TestCase):
    def test_summary_accepts_forward_max_gain_sort_column(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "summary",
                "--run-dir",
                "research_runs/demo",
                "--sort-by",
                "mean_forward_3d_max_gain",
            ]
        )

        self.assertEqual(args.command, "summary")
        self.assertEqual(args.sort_by, "mean_forward_3d_max_gain")


if __name__ == "__main__":
    unittest.main()
