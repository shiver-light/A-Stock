from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from data.valuation_daily import AShareDailyValuationService, DailyValuationRequest


class DailyValuationServiceTestCase(unittest.TestCase):
    def test_get_daily_valuation_returns_requested_schema_for_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = Mock()
            client.daily_basic.return_value = pd.DataFrame()
            service = AShareDailyValuationService(client=client, cache_dir=tmp_dir)

            result = service.get_daily_valuation(
                DailyValuationRequest(
                    ts_code="002820.SH",
                    start_date="20160101",
                    end_date="20260608",
                )
            )

            self.assertTrue(result.empty)
            self.assertEqual(
                result.columns.tolist(),
                ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "pe_ttm", "pb"],
            )


if __name__ == "__main__":
    unittest.main()
