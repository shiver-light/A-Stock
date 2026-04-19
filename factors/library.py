"""Factor registry."""

from __future__ import annotations

from factors.fundamental import pb_factor, pe_ttm_factor, revenue_growth_factor, roe_factor, turnover_mean_20d_factor
from factors.technical import (
    amplitude_20d_factor,
    close_to_high_20d_factor,
    momentum_60d_factor,
    price_rank_60d_factor,
    return_20d_factor,
    return_5d_factor,
    reversal_5d_factor,
    turnover_volatility_20d_factor,
    volatility_20d_factor,
)

FACTOR_REGISTRY = {
    "return_5d": return_5d_factor,
    "return_20d": return_20d_factor,
    "momentum_60d": momentum_60d_factor,
    "reversal_5d": reversal_5d_factor,
    "volatility_20d": volatility_20d_factor,
    "turnover_mean_20d": turnover_mean_20d_factor,
    "turnover_volatility_20d": turnover_volatility_20d_factor,
    "price_rank_60d": price_rank_60d_factor,
    "amplitude_20d": amplitude_20d_factor,
    "close_to_high_20d": close_to_high_20d_factor,
    "pe_ttm": pe_ttm_factor,
    "pb": pb_factor,
    "roe": roe_factor,
    "revenue_growth": revenue_growth_factor,
}


def list_factors() -> list[str]:
    return sorted(FACTOR_REGISTRY.keys())


def get_factor_function(name: str):
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"Unknown factor: {name}")
    return FACTOR_REGISTRY[name]
