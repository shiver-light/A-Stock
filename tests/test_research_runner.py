from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.runner import run_experiments


class ResearchRunnerTestCase(unittest.TestCase):
    @patch("research.runner.run_minimal_pipeline")
    def test_run_experiments_persists_factor_diagnostics_when_enabled(self, mock_pipeline) -> None:
        mock_pipeline.return_value = {
            "performance": {"cumulative_return": 0.1},
            "latest_selection": {"top_stocks": []},
            "report": {"backtest_summary": {"cumulative_return": 0.1}},
            "report_text": "strategy report",
            "factor_diagnostics": {
                "return_20d": [
                    {
                        "factor_name": "return_20d",
                        "horizon": 5,
                        "ic_mean": 0.02,
                        "ic_std": 0.01,
                        "icir": 2.0,
                        "rank_ic_mean": 0.03,
                        "coverage": 0.8,
                        "quantile_return_summary": [],
                    }
                ]
            },
            "factor_report_text": {"return_20d": "factor diagnostics text"},
        }

        config = {
            "global": {
                "start_date": "20240101",
                "end_date": "20240131",
            },
            "experiments": [
                {
                    "name": "exp_with_factor_diag",
                    "ts_codes": ["000001.SZ"],
                    "enable_factor_diagnostics": True,
                    "analysis_horizons": [5, 10],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_experiments(config, output_dir=tmp_dir, run_name="demo_run", resume=False)
            experiment_dir = Path(tmp_dir) / "demo_run" / "experiments" / "exp_with_factor_diag"

            self.assertEqual(len(results), 1)
            self.assertTrue((experiment_dir / "factor_diagnostics.json").exists())
            self.assertTrue((experiment_dir / "factor_report.txt").exists())

            diagnostics = json.loads((experiment_dir / "factor_diagnostics.json").read_text(encoding="utf-8"))
            self.assertIn("return_20d", diagnostics)
            report_text = (experiment_dir / "factor_report.txt").read_text(encoding="utf-8")
            self.assertIn("[return_20d]", report_text)
            self.assertIn("factor diagnostics text", report_text)

            _, kwargs = mock_pipeline.call_args
            self.assertTrue(kwargs["enable_factor_diagnostics"])
            self.assertEqual(kwargs["analysis_horizons"], (5, 10))

    @patch("research.runner.run_minimal_pipeline")
    def test_run_experiments_remains_compatible_without_factor_diagnostics(self, mock_pipeline) -> None:
        mock_pipeline.return_value = {
            "performance": {"cumulative_return": 0.1},
            "latest_selection": {"top_stocks": []},
            "report": {"backtest_summary": {"cumulative_return": 0.1}},
            "report_text": "strategy report",
        }

        config = {
            "global": {
                "start_date": "20240101",
                "end_date": "20240131",
            },
            "experiments": [
                {
                    "name": "exp_without_factor_diag",
                    "ts_codes": ["000001.SZ"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_experiments(config, output_dir=tmp_dir, run_name="demo_run", resume=False)
            experiment_dir = Path(tmp_dir) / "demo_run" / "experiments" / "exp_without_factor_diag"

            self.assertEqual(len(results), 1)
            self.assertFalse((experiment_dir / "factor_diagnostics.json").exists())
            self.assertFalse((experiment_dir / "factor_report.txt").exists())

            _, kwargs = mock_pipeline.call_args
            self.assertFalse(kwargs["enable_factor_diagnostics"])
            self.assertEqual(kwargs["analysis_horizons"], (5, 10, 20))

    @patch("research.runner.run_minimal_pipeline")
    def test_run_experiments_passes_backtest_config(self, mock_pipeline) -> None:
        mock_pipeline.return_value = {
            "performance": {"cumulative_return": 0.1},
            "latest_selection": {"top_stocks": []},
            "report": {"backtest_summary": {"cumulative_return": 0.1}},
            "report_text": "strategy report",
        }

        config = {
            "global": {
                "start_date": "20240101",
                "end_date": "20240131",
                "backtest_config": {"slippage_bps": 10.0, "min_amount": 5000000.0},
            },
            "experiments": [
                {
                    "name": "exp_with_backtest_config",
                    "ts_codes": ["000001.SZ"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_experiments(config, output_dir=tmp_dir, run_name="demo_run", resume=False)

            self.assertEqual(len(results), 1)
            _, kwargs = mock_pipeline.call_args
            self.assertEqual(kwargs["backtest_config"], {"slippage_bps": 10.0, "min_amount": 5000000.0})


if __name__ == "__main__":
    unittest.main()
