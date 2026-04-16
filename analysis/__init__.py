"""Factor analysis helpers."""

from .factor_metrics import (
    calc_factor_coverage,
    calc_forward_returns,
    calc_ic,
    calc_quantile_groups,
    calc_quantile_returns,
    calc_rank_ic,
)
from .factor_report import build_factor_diagnostics_report, render_factor_report_text

__all__ = [
    "calc_factor_coverage",
    "calc_forward_returns",
    "calc_ic",
    "calc_quantile_groups",
    "calc_quantile_returns",
    "calc_rank_ic",
    "build_factor_diagnostics_report",
    "render_factor_report_text",
]
