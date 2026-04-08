"""Minimal daily backtest module."""

from .engine import attach_benchmark, calc_benchmark_returns, generate_weights, run_backtest
from .metrics import calc_performance, calc_relative_performance

__all__ = [
    "generate_weights",
    "run_backtest",
    "calc_benchmark_returns",
    "attach_benchmark",
    "calc_performance",
    "calc_relative_performance",
]
