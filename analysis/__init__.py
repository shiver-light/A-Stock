"""Factor analysis helpers."""

from .data_diagnostics import (
    summarize_complete_case_count_by_date,
    summarize_factor_coverage,
    summarize_universe_count_by_date,
)
from .data_checks import (
    check_duplicate_keys,
    check_factor_signal_alignment,
    check_missing_ratio_by_date,
    check_universe_stability,
)
from .external_regime import build_external_regime_flags, load_external_regime_data
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
    "summarize_complete_case_count_by_date",
    "summarize_factor_coverage",
    "summarize_universe_count_by_date",
    "check_duplicate_keys",
    "check_factor_signal_alignment",
    "check_missing_ratio_by_date",
    "check_universe_stability",
    "calc_factor_coverage",
    "calc_forward_returns",
    "calc_ic",
    "calc_quantile_groups",
    "calc_quantile_returns",
    "calc_rank_ic",
    "build_factor_diagnostics_report",
    "render_factor_report_text",
    "build_external_regime_flags",
    "load_external_regime_data",
]
