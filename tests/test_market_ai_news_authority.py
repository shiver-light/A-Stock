from __future__ import annotations

import unittest

from market_ai.scoring import calculate_authority_score


class MarketAiNewsAuthorityTestCase(unittest.TestCase):
    def test_configured_source_score(self) -> None:
        result = calculate_authority_score(
            "财联社",
            authority_scores={"财联社": 80.0},
            default_score=50.0,
        )

        self.assertEqual(result.score, 80.0)
        self.assertEqual(result.reason, "configured_source_score")

    def test_unknown_source_uses_default_score(self) -> None:
        result = calculate_authority_score("未知来源", authority_scores={"财联社": 80.0}, default_score=45.0)

        self.assertEqual(result.score, 45.0)
        self.assertEqual(result.reason, "unknown_source_default")

    def test_empty_source_uses_default_score(self) -> None:
        result = calculate_authority_score("", authority_scores={"财联社": 80.0}, default_score=45.0)

        self.assertEqual(result.score, 45.0)
        self.assertEqual(result.reason, "empty_source_default")


if __name__ == "__main__":
    unittest.main()
