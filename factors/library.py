"""Factor registry."""

from __future__ import annotations

from factors.fundamental import (
    bp_factor,
    ep_ttm_factor,
    pb_factor,
    pe_ttm_factor,
    revenue_growth_factor,
    roe_factor,
    roe_ttm_factor,
    turnover_mean_20d_factor,
)
from factors.technical import (
    amount_mean_20d_factor,
    amplitude_20d_factor,
    close_to_high_20d_factor,
    close_near_high_on_high_amount_20d_factor,
    distribution_risk_20d_negative_factor,
    down_day_absorption_20d_factor,
    down_day_support_20d_factor,
    gap_risk_20d_negative_factor,
    high_turnover_low_range_20d_factor,
    illiq_negative_factor,
    liquidity_improvement_20d_factor,
    low_range_high_amount_days_ratio_20d_factor,
    max_drawdown_60d_negative_factor,
    momentum_60d_factor,
    money_flow_strength_20d_factor,
    price_rank_60d_factor,
    price_suppression_20d_factor,
    position_safety_60d_factor,
    pullback_after_trend_60d_factor,
    return_120d_factor,
    return_20d_factor,
    small_body_high_turnover_20d_factor,
    return_5d_negative_factor,
    return_60d_factor,
    return_5d_factor,
    reversal_5d_factor,
    turnover_stability_20d_factor,
    turnover_volatility_20d_factor,
    volatility_contraction_20d_factor,
    volatility_20d_factor,
    volatility_20d_negative_factor,
    volatility_60d_factor,
    volatility_60d_negative_factor,
)

FACTOR_REGISTRY = {
    "return_5d": return_5d_factor,
    "return_20d": return_20d_factor,
    "return_60d": return_60d_factor,
    "return_120d": return_120d_factor,
    "return_5d_negative": return_5d_negative_factor,
    "momentum_60d": momentum_60d_factor,
    "reversal_5d": reversal_5d_factor,
    "volatility_20d": volatility_20d_factor,
    "volatility_20d_negative": volatility_20d_negative_factor,
    "volatility_60d": volatility_60d_factor,
    "volatility_60d_negative": volatility_60d_negative_factor,
    "max_drawdown_60d_negative": max_drawdown_60d_negative_factor,
    "turnover_mean_20d": turnover_mean_20d_factor,
    "turnover_volatility_20d": turnover_volatility_20d_factor,
    "turnover_stability_20d": turnover_stability_20d_factor,
    "amount_mean_20d": amount_mean_20d_factor,
    "gap_risk_20d_negative": gap_risk_20d_negative_factor,
    "distribution_risk_20d_negative": distribution_risk_20d_negative_factor,
    "illiq_negative": illiq_negative_factor,
    "liquidity_improvement_20d": liquidity_improvement_20d_factor,
    "low_range_high_amount_days_ratio_20d": low_range_high_amount_days_ratio_20d_factor,
    "money_flow_strength_20d": money_flow_strength_20d_factor,
    "high_turnover_low_range_20d": high_turnover_low_range_20d_factor,
    "price_suppression_20d": price_suppression_20d_factor,
    "down_day_support_20d": down_day_support_20d_factor,
    "small_body_high_turnover_20d": small_body_high_turnover_20d_factor,
    "close_near_high_on_high_amount_20d": close_near_high_on_high_amount_20d_factor,
    "down_day_absorption_20d": down_day_absorption_20d_factor,
    "position_safety_60d": position_safety_60d_factor,
    "pullback_after_trend_60d": pullback_after_trend_60d_factor,
    "volatility_contraction_20d": volatility_contraction_20d_factor,
    "price_rank_60d": price_rank_60d_factor,
    "amplitude_20d": amplitude_20d_factor,
    "close_to_high_20d": close_to_high_20d_factor,
    "ep_ttm": ep_ttm_factor,
    "pe_ttm": pe_ttm_factor,
    "bp": bp_factor,
    "pb": pb_factor,
    "roe": roe_factor,
    "roe_ttm": roe_ttm_factor,
    "revenue_growth": revenue_growth_factor,
}


def list_factors() -> list[str]:
    return sorted(FACTOR_REGISTRY.keys())


def get_factor_function(name: str):
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"Unknown factor: {name}")
    return FACTOR_REGISTRY[name]
