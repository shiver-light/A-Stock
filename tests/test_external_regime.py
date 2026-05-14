from __future__ import annotations

import unittest

import pandas as pd

from analysis.external_regime import build_external_regime_flags


class ExternalRegimeTestCase(unittest.TestCase):
    def test_build_external_regime_flags_returns_empty_for_empty_input(self) -> None:
        result = build_external_regime_flags(
            pd.DataFrame(columns=["trade_date", "us_risk_on"]),
            {"allowed_col": "us_risk_on"},
        )

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["trade_date", "external_regime_allowed"])

    def test_build_external_regime_flags_uses_allowed_column(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103", "20240104"],
                "us_risk_on": [1, 0, 1],
            }
        )

        result = build_external_regime_flags(data, {"allowed_col": "us_risk_on"})

        self.assertEqual(result["external_regime_allowed"].tolist(), [True, False, True])

    def test_build_external_regime_flags_combines_numeric_rules(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240103", "20240104"],
                "qqq_return_20d": [0.02, 0.03, -0.01],
                "vix_return_20d": [0.05, 0.2, 0.01],
            }
        )

        result = build_external_regime_flags(
            data,
            {
                "rules": [
                    {"column": "qqq_return_20d", "op": "gte", "value": 0.0},
                    {"column": "vix_return_20d", "op": "lte", "value": 0.1},
                ]
            },
        )

        self.assertEqual(result["external_regime_allowed"].tolist(), [True, False, False])

    def test_build_external_regime_flags_rejects_missing_rule_column(self) -> None:
        data = pd.DataFrame({"trade_date": ["20240102"], "qqq_return_20d": [0.02]})

        with self.assertRaisesRegex(ValueError, "column not found"):
            build_external_regime_flags(
                data,
                {"rules": [{"column": "vix_return_20d", "op": "lte", "value": 0.1}]},
            )


if __name__ == "__main__":
    unittest.main()
