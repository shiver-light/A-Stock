"""Formatting helpers for single-factor diagnostic reports."""

from __future__ import annotations

import pandas as pd


def _safe_mean(data: pd.Series) -> float | None:
    valid = data.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _safe_std(data: pd.Series) -> float | None:
    valid = data.dropna()
    if valid.empty:
        return None
    return float(valid.std(ddof=1))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return float(numerator / denominator)


def _prepare_metric_data(data: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data.loc[:, required_columns].copy()


def _build_quantile_summary(quantile_returns: pd.DataFrame, horizon: int) -> list[dict[str, object]]:
    if quantile_returns.empty:
        return []

    prepared = _prepare_metric_data(
        quantile_returns,
        ["trade_date", "quantile", "horizon", "forward_return", "n_obs"],
    )
    filtered = prepared.loc[prepared["horizon"] == horizon].dropna(subset=["quantile", "forward_return"])
    if filtered.empty:
        return []

    grouped = (
        filtered.groupby("quantile", sort=True)
        .agg(
            mean_return=("forward_return", "mean"),
            mean_count=("n_obs", "mean"),
            date_count=("trade_date", "nunique"),
        )
        .reset_index()
    )
    grouped["mean_return"] = grouped["mean_return"].astype(float)
    grouped["mean_count"] = grouped["mean_count"].astype(float)
    grouped["date_count"] = grouped["date_count"].astype(int)

    return grouped.to_dict(orient="records")


def build_factor_diagnostics_report(
    *,
    factor_name: str,
    horizon: int,
    ic_data: pd.DataFrame,
    rank_ic_data: pd.DataFrame,
    coverage_data: pd.DataFrame,
    quantile_returns: pd.DataFrame,
) -> dict[str, object]:
    """Build a JSON-friendly single-factor diagnostics summary for one horizon."""

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    ic_prepared = _prepare_metric_data(ic_data, ["trade_date", "horizon", "ic", "n_obs", "method"])
    rank_ic_prepared = _prepare_metric_data(
        rank_ic_data,
        ["trade_date", "horizon", "ic", "n_obs", "method"],
    )
    coverage_prepared = _prepare_metric_data(
        coverage_data,
        ["trade_date", "coverage", "non_null_count", "total_count"],
    )

    ic_filtered = ic_prepared.loc[ic_prepared["horizon"] == horizon]
    rank_ic_filtered = rank_ic_prepared.loc[rank_ic_prepared["horizon"] == horizon]

    ic_mean = _safe_mean(ic_filtered["ic"]) if not ic_filtered.empty else None
    ic_std = _safe_std(ic_filtered["ic"]) if not ic_filtered.empty else None
    rank_ic_mean = _safe_mean(rank_ic_filtered["ic"]) if not rank_ic_filtered.empty else None
    coverage_mean = _safe_mean(coverage_prepared["coverage"]) if not coverage_prepared.empty else None

    return {
        "factor_name": factor_name,
        "horizon": horizon,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": _safe_ratio(ic_mean, ic_std),
        "rank_ic_mean": rank_ic_mean,
        "coverage": coverage_mean,
        "observation_summary": {
            "ic_dates": int(ic_filtered["trade_date"].nunique()) if not ic_filtered.empty else 0,
            "rank_ic_dates": int(rank_ic_filtered["trade_date"].nunique()) if not rank_ic_filtered.empty else 0,
            "coverage_dates": int(coverage_prepared["trade_date"].nunique()) if not coverage_prepared.empty else 0,
        },
        "quantile_return_summary": _build_quantile_summary(quantile_returns, horizon),
        "risk": [
            "基于历史样本统计，不保证未来有效。",
            "IC、分组收益和覆盖率可能随市场风格变化而失效。",
            "当前报告仅做单因子诊断，不包含交易成本和执行约束。",
        ],
    }


def render_factor_report_text(report: dict[str, object]) -> str:
    """Render a single-factor diagnostics report into a readable text block."""

    quantile_summary = report.get("quantile_return_summary", [])
    observation_summary = report.get("observation_summary", {})
    risk = report.get("risk", [])

    quantile_lines = [
        (
            f"q{item['quantile']}: mean_return={float(item['mean_return']):.6f}, "
            f"mean_count={float(item['mean_count']):.2f}, dates={int(item['date_count'])}"
        )
        for item in quantile_summary
    ]

    def _fmt(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "factor diagnostics:",
        f"factor name: {report.get('factor_name')}",
        f"horizon: {report.get('horizon')}d",
        "",
        "summary:",
        f"ic mean: {_fmt(report.get('ic_mean'))}",
        f"ic std: {_fmt(report.get('ic_std'))}",
        f"icir: {_fmt(report.get('icir'))}",
        f"rank ic mean: {_fmt(report.get('rank_ic_mean'))}",
        f"coverage: {_fmt(report.get('coverage'))}",
        "",
        "observation summary:",
        f"ic dates: {observation_summary.get('ic_dates', 0)}",
        f"rank ic dates: {observation_summary.get('rank_ic_dates', 0)}",
        f"coverage dates: {observation_summary.get('coverage_dates', 0)}",
        "",
        "quantile return summary:",
    ]
    lines.extend(quantile_lines or ["N/A"])
    lines.extend(["", "risk:"])
    lines.extend(risk or ["N/A"])
    return "\n".join(lines)
