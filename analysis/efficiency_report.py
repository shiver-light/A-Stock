"""Recent up/down efficiency diagnostics for single-stock daily bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["trade_date", "ts_code", "close", "pre_close", "high", "low", "vol", "turnover_rate_f"]


def build_efficiency_report(
    market_data: pd.DataFrame,
    *,
    stock_label: str | None = None,
    benchmark_data: pd.DataFrame | None = None,
    benchmark_label: str | None = None,
    window: int = 10,
    recent_days: int = 3,
) -> dict[str, object]:
    """Build a JSON-friendly recent up/down efficiency report.

    Up efficiency is positive daily return divided by turnover rate. Down
    efficiency is absolute negative daily return divided by turnover rate.
    Both use same-day daily bars and turnover, with no future information.
    """

    if window <= 0:
        raise ValueError("window must be positive.")
    if recent_days <= 0 or recent_days >= window:
        raise ValueError("recent_days must be positive and smaller than window.")

    if market_data.empty:
        return _empty_report(stock_label, window, "input data is empty")
    _validate_columns(market_data, REQUIRED_COLUMNS)

    data = market_data.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    for column in ["close", "pre_close", "high", "low", "vol", "turnover_rate_f"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["trade_date", "ts_code"]).drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    data = data.tail(window).reset_index(drop=True)
    if data.empty:
        return _empty_report(stock_label, window, "no rows after cleaning")

    data["daily_return"] = np.where(data["pre_close"] == 0, np.nan, (data["close"] / data["pre_close"]) - 1.0)
    data["amplitude"] = np.where(data["pre_close"] == 0, np.nan, (data["high"] - data["low"]) / data["pre_close"])
    intraday_range = data["high"] - data["low"]
    data["close_position"] = np.where(intraday_range == 0, np.nan, (data["close"] - data["low"]) / intraday_range)
    turnover = data["turnover_rate_f"].replace(0, np.nan)
    data["up_efficiency"] = np.where(data["daily_return"] > 0, data["daily_return"] / turnover, np.nan)
    data["down_efficiency"] = np.where(data["daily_return"] < 0, data["daily_return"].abs() / turnover, np.nan)
    data["volume_ratio"] = data["vol"] / data["vol"].rolling(5, min_periods=1).mean().shift(1)

    recent = data.tail(recent_days)
    previous = data.iloc[: max(len(data) - recent_days, 0)]

    up_trend = _trend_label(data["up_efficiency"])
    down_trend = _trend_label(data["down_efficiency"])
    recent_change = _describe_recent_change(recent, previous)
    selling_pressure = _selling_pressure_label(recent, previous)
    absorption = _absorption_label(recent, previous)
    long_short = _long_short_label(up_trend, down_trend, selling_pressure, absorption)
    stage = _stage_label(data, up_trend, down_trend, selling_pressure, absorption, long_short)
    benchmark_context = build_benchmark_context(
        benchmark_data,
        stock_returns=data.loc[:, ["trade_date", "daily_return"]],
        label=benchmark_label,
    )

    rows = []
    for _, row in data.iterrows():
        rows.append(
            {
                "trade_date": row["trade_date"],
                "daily_return": _to_float(row["daily_return"]),
                "turnover_rate_f": _to_float(row["turnover_rate_f"]),
                "volume": _to_float(row["vol"]),
                "volume_ratio": _to_float(row["volume_ratio"]),
                "amplitude": _to_float(row["amplitude"]),
                "close_position": _to_float(row["close_position"]),
                "up_efficiency": _to_float(row["up_efficiency"]),
                "down_efficiency": _to_float(row["down_efficiency"]),
                "up_efficiency_x10000": _scale_efficiency(row["up_efficiency"]),
                "down_efficiency_x10000": _scale_efficiency(row["down_efficiency"]),
            }
        )

    return {
        "stock": stock_label or _infer_stock_label(data),
        "window": int(window),
        "recent_days": int(recent_days),
        "start_date": str(data["trade_date"].min()),
        "end_date": str(data["trade_date"].max()),
        "up_efficiency_trend": up_trend,
        "down_efficiency_trend": down_trend,
        "recent_change": recent_change,
        "up_state": _up_state(up_trend),
        "down_state": _down_state(down_trend),
        "selling_pressure": selling_pressure,
        "absorption": absorption,
        "long_short_structure": long_short,
        "current_stage": stage,
        "benchmark_context": benchmark_context,
        "sector_context": _empty_sector_context(),
        "final_conclusion": _final_conclusion(
            up_trend,
            down_trend,
            selling_pressure,
            absorption,
            benchmark_context,
        ),
        "summary": _summary_stats(data, recent, previous),
        "efficiency_display_scale": 10000,
        "rows": rows,
        "warnings": [],
    }


def build_benchmark_context(
    benchmark_data: pd.DataFrame | None,
    *,
    stock_returns: pd.DataFrame | None = None,
    label: str | None = None,
) -> dict[str, object]:
    """Build market environment context from historical index daily closes."""

    if benchmark_data is None or benchmark_data.empty:
        return {
            "label": label or "",
            "status": "暂无数据",
            "strength": "暂无数据",
            "relative_strength": "暂无数据",
            "description": "未提供大盘指数数据，暂不判断市场环境。",
        }
    _validate_columns(benchmark_data, ["trade_date", "close"])
    data = benchmark_data.copy()
    data = data.loc[:, [column for column in ["trade_date", "ts_code", "close"] if column in data.columns]]
    data["trade_date"] = data["trade_date"].astype(str)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    close = data["close"]
    return_5d = _period_return(close, 5)
    return_10d = _period_return(close, 10)
    return_20d = _period_return(close, 20)
    ma20 = close.rolling(20).mean()
    above_ma20 = bool(close.iloc[-1] >= ma20.iloc[-1]) if len(close) >= 20 and pd.notna(ma20.iloc[-1]) else False
    recent_3d = _period_return(close, 3)
    strength = _benchmark_strength(return_20d, above_ma20, recent_3d)
    relative_strength = _relative_strength_label(stock_returns, data)
    return {
        "label": label or _infer_benchmark_label(data),
        "status": "available",
        "strength": strength,
        "relative_strength": relative_strength,
        "return_5d": return_5d,
        "return_10d": return_10d,
        "return_20d": return_20d,
        "return_3d": recent_3d,
        "close_above_ma20": above_ma20,
        "description": (
            f"大盘近5/10/20日收益分别为{_format_pct(return_5d)}、"
            f"{_format_pct(return_10d)}、{_format_pct(return_20d)}，"
            f"{'站上' if above_ma20 else '未站上'}MA20。"
        ),
    }


def render_efficiency_report_text(report: dict[str, object]) -> str:
    """Render an efficiency report in a human-readable Chinese text format."""

    summary = report.get("summary", {})
    benchmark_context = report.get("benchmark_context", {})
    sector_context = report.get("sector_context", {})
    rows = report.get("rows", [])
    lines = [
        "【最近10日上涨/下跌效率报告】",
        "",
        f"股票：{report.get('stock', '')}",
        "",
        f"上涨效率趋势：{report.get('up_efficiency_trend', '其他')}",
        f"下跌效率趋势：{report.get('down_efficiency_trend', '其他')}",
        "",
        "最近3日变化：",
        str(report.get("recent_change", "")),
        "",
        "上涨状态：",
        str(report.get("up_state", "")),
        "",
        "下跌状态：",
        str(report.get("down_state", "")),
        "",
        f"卖压：{report.get('selling_pressure', '正常')}",
        "",
        f"承接：{report.get('absorption', '一般')}",
        "",
        f"多空结构：{report.get('long_short_structure', '平衡')}",
        "",
        f"当前阶段：{report.get('current_stage', '其他')}",
        "",
        f"大盘环境：{benchmark_context.get('strength', '暂无数据')}",
        f"说明：{benchmark_context.get('description', '未提供大盘指数数据。')}",
        "",
        f"板块环境：{sector_context.get('strength', '暂无数据')}",
        f"说明：{sector_context.get('description', '暂未接入板块数据。')}",
        "",
        f"个股相对强弱：{benchmark_context.get('relative_strength', '暂无数据')}",
        "",
        "最终结论：",
        str(report.get("final_conclusion", "")),
        "",
        "关键统计：",
        (
            f"最近3日均涨跌幅 { _format_pct(summary.get('recent_mean_return')) }，"
            f"前7日均涨跌幅 { _format_pct(summary.get('previous_mean_return')) }；"
            f"最近3日均换手 { _format_num(summary.get('recent_mean_turnover')) }%，"
            f"前7日均换手 { _format_num(summary.get('previous_mean_turnover')) }%。"
        ),
        "",
        "最近10日明细：",
        "trade_date  return  turnover%  vol_ratio  amplitude  close_pos  up_eff_x10000  down_eff_x10000",
    ]
    for row in rows:
        lines.append(
            " ".join(
                [
                    str(row["trade_date"]),
                    _format_pct(row["daily_return"]).rjust(8),
                    _format_num(row["turnover_rate_f"]).rjust(9),
                    _format_num(row["volume_ratio"]).rjust(9),
                    _format_pct(row["amplitude"]).rjust(9),
                    _format_num(row["close_position"]).rjust(9),
                    _format_num(row["up_efficiency_x10000"]).rjust(13),
                    _format_num(row["down_efficiency_x10000"]).rjust(15),
                ]
            )
        )
    return "\n".join(lines)


def _validate_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _empty_report(stock_label: str | None, window: int, warning: str) -> dict[str, object]:
    return {
        "stock": stock_label or "",
        "window": window,
        "recent_days": 3,
        "up_efficiency_trend": "其他",
        "down_efficiency_trend": "其他",
        "recent_change": "数据不足，无法判断最近3日相对前7日变化。",
        "up_state": "数据不足。",
        "down_state": "数据不足。",
        "selling_pressure": "正常",
        "absorption": "一般",
        "long_short_structure": "平衡",
        "current_stage": "其他",
        "benchmark_context": build_benchmark_context(None),
        "sector_context": _empty_sector_context(),
        "final_conclusion": "数据不足，暂不形成短线效率判断。",
        "summary": {},
        "efficiency_display_scale": 10000,
        "rows": [],
        "warnings": [warning],
    }


def _infer_stock_label(data: pd.DataFrame) -> str:
    ts_code = data["ts_code"].dropna().astype(str)
    return ts_code.iloc[0] if not ts_code.empty else ""


def _infer_benchmark_label(data: pd.DataFrame) -> str:
    if "ts_code" in data.columns:
        ts_code = data["ts_code"].dropna().astype(str)
        if not ts_code.empty:
            return ts_code.iloc[0]
    return "benchmark"


def _empty_sector_context() -> dict[str, object]:
    return {
        "status": "not_available",
        "strength": "暂无数据",
        "relative_strength": "暂无数据",
        "description": "暂未接入板块/行业成分数据，当前报告只判断个股和大盘。",
    }


def _period_return(close: pd.Series, periods: int) -> float | None:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if len(values) <= periods:
        return None
    previous = values.iloc[-periods - 1]
    latest = values.iloc[-1]
    if previous == 0:
        return None
    return float((latest / previous) - 1.0)


def _benchmark_strength(return_20d: float | None, above_ma20: bool, return_3d: float | None) -> str:
    if return_20d is None:
        return "暂无数据"
    if return_20d > 0.03 and above_ma20:
        return "强"
    if return_20d < -0.03 and not above_ma20:
        return "弱"
    if return_3d is not None and return_3d < -0.03 and not above_ma20:
        return "弱"
    return "中性"


def _relative_strength_label(stock_returns: pd.DataFrame | None, benchmark_data: pd.DataFrame) -> str:
    if stock_returns is None or stock_returns.empty or "daily_return" not in stock_returns.columns:
        return "暂无数据"
    stock = stock_returns.copy()
    stock["trade_date"] = stock["trade_date"].astype(str)
    stock = stock.sort_values("trade_date").tail(10)
    benchmark = benchmark_data.loc[:, ["trade_date", "close"]].copy()
    benchmark["trade_date"] = benchmark["trade_date"].astype(str)
    benchmark["benchmark_return"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change()
    aligned = stock.loc[:, ["trade_date", "daily_return"]].merge(
        benchmark.loc[:, ["trade_date", "benchmark_return"]],
        on="trade_date",
        how="inner",
    )
    if aligned.empty:
        return "暂无数据"
    stock_total = _compound_return(aligned["daily_return"])
    benchmark_total = _compound_return(aligned["benchmark_return"])
    if stock_total is None or benchmark_total is None:
        return "暂无数据"
    diff = stock_total - benchmark_total
    if diff > 0.03:
        return "强于大盘"
    if diff < -0.03:
        return "弱于大盘"
    return "跟随大盘"


def _compound_return(returns: pd.Series) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return None
    return float((1.0 + values).prod() - 1.0)


def _trend_label(series: pd.Series) -> str:
    values = series.dropna()
    if len(values) < 2:
        return "持平"
    x = np.arange(len(values), dtype="float64")
    slope = float(np.polyfit(x, values.to_numpy(dtype="float64"), 1)[0])
    mean_abs = float(values.abs().mean())
    threshold = max(mean_abs * 0.08, 0.0001)
    if slope > threshold:
        return "提高"
    if slope < -threshold:
        return "下降"
    return "持平"


def _mean(data: pd.DataFrame, column: str) -> float | None:
    if data.empty or column not in data.columns:
        return None
    value = data[column].dropna().mean()
    return _to_float(value)


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _scale_efficiency(value) -> float | None:
    raw = _to_float(value)
    if raw is None:
        return None
    return raw * 10000.0


def _summary_stats(data: pd.DataFrame, recent: pd.DataFrame, previous: pd.DataFrame) -> dict[str, float | None]:
    return {
        "mean_return": _mean(data, "daily_return"),
        "recent_mean_return": _mean(recent, "daily_return"),
        "previous_mean_return": _mean(previous, "daily_return"),
        "mean_turnover": _mean(data, "turnover_rate_f"),
        "recent_mean_turnover": _mean(recent, "turnover_rate_f"),
        "previous_mean_turnover": _mean(previous, "turnover_rate_f"),
        "recent_mean_volume_ratio": _mean(recent, "volume_ratio"),
        "previous_mean_volume_ratio": _mean(previous, "volume_ratio"),
        "recent_mean_amplitude": _mean(recent, "amplitude"),
        "previous_mean_amplitude": _mean(previous, "amplitude"),
        "recent_mean_close_position": _mean(recent, "close_position"),
        "previous_mean_close_position": _mean(previous, "close_position"),
        "recent_mean_up_efficiency": _mean(recent, "up_efficiency"),
        "previous_mean_up_efficiency": _mean(previous, "up_efficiency"),
        "recent_mean_down_efficiency": _mean(recent, "down_efficiency"),
        "previous_mean_down_efficiency": _mean(previous, "down_efficiency"),
    }


def _describe_recent_change(recent: pd.DataFrame, previous: pd.DataFrame) -> str:
    stats = _summary_stats(pd.concat([previous, recent], ignore_index=True), recent, previous)
    return (
        f"最近3日均换手为{_format_num(stats['recent_mean_turnover'])}%，"
        f"前7日为{_format_num(stats['previous_mean_turnover'])}%；"
        f"最近3日均振幅为{_format_pct(stats['recent_mean_amplitude'])}，"
        f"前7日为{_format_pct(stats['previous_mean_amplitude'])}；"
        f"最近3日收盘位置均值为{_format_num(stats['recent_mean_close_position'])}，"
        f"前7日为{_format_num(stats['previous_mean_close_position'])}。"
    )


def _up_state(up_trend: str) -> str:
    if up_trend == "提高":
        return "上涨效率提高，同等换手下价格上行更有效。"
    if up_trend == "下降":
        return "上涨效率下降，上涨需要更多换手配合，短线更费力。"
    return "上涨效率大体持平，暂未体现明显边际变化。"


def _down_state(down_trend: str) -> str:
    if down_trend == "提高":
        return "下跌效率提高，同等换手下跌幅扩大，下行更容易。"
    if down_trend == "下降":
        return "下跌效率下降，同等换手下跌幅收敛，下跌更困难。"
    return "下跌效率大体持平，卖压变化不明显。"


def _selling_pressure_label(recent: pd.DataFrame, previous: pd.DataFrame) -> str:
    recent_down = recent.loc[recent["daily_return"] < 0]
    previous_down = previous.loc[previous["daily_return"] < 0]
    recent_abs_down = _mean(recent_down.assign(abs_return=recent_down["daily_return"].abs()), "abs_return")
    previous_abs_down = _mean(previous_down.assign(abs_return=previous_down["daily_return"].abs()), "abs_return")
    recent_turnover = _mean(recent, "turnover_rate_f")
    previous_turnover = _mean(previous, "turnover_rate_f")
    if recent_abs_down is None:
        return "减弱"
    if previous_abs_down is None:
        return "正常"
    shrinking_volume = recent_turnover is not None and previous_turnover is not None and recent_turnover < previous_turnover * 0.9
    if shrinking_volume and recent_abs_down < previous_abs_down * 0.8:
        return "衰竭"
    if recent_abs_down < previous_abs_down:
        return "减弱"
    if recent_abs_down > previous_abs_down * 1.2:
        return "增强"
    return "正常"


def _absorption_label(recent: pd.DataFrame, previous: pd.DataFrame) -> str:
    recent_close_position = _mean(recent, "close_position")
    previous_close_position = _mean(previous, "close_position")
    recent_down_eff = _mean(recent, "down_efficiency")
    previous_down_eff = _mean(previous, "down_efficiency")
    if recent_close_position is not None and previous_close_position is not None:
        if recent_close_position > previous_close_position + 0.1 and (
            previous_down_eff is None or recent_down_eff is None or recent_down_eff <= previous_down_eff
        ):
            return "增强"
        if recent_close_position < previous_close_position - 0.1:
            return "减弱"
    return "一般"


def _long_short_label(up_trend: str, down_trend: str, selling_pressure: str, absorption: str) -> str:
    if up_trend == "提高" and down_trend == "下降":
        return "改善"
    if up_trend == "下降" and down_trend == "提高":
        return "恶化"
    if selling_pressure in {"衰竭", "减弱"} and absorption == "增强":
        return "改善"
    if selling_pressure == "增强" or absorption == "减弱":
        return "恶化"
    return "平衡"


def _stage_label(
    data: pd.DataFrame,
    up_trend: str,
    down_trend: str,
    selling_pressure: str,
    absorption: str,
    long_short: str,
) -> str:
    close_position = _mean(data.tail(3), "close_position")
    amplitude = _mean(data.tail(3), "amplitude")
    if long_short == "改善" and up_trend == "提高" and down_trend == "下降":
        return "底部转强"
    if selling_pressure == "衰竭" and absorption in {"增强", "一般"}:
        return "缩量整理"
    if long_short == "恶化" and down_trend == "提高":
        return "弱势下跌"
    if close_position is not None and close_position < 0.35 and selling_pressure in {"正常", "增强"}:
        return "二次探底"
    if amplitude is not None and amplitude > 0.06 and long_short != "改善":
        return "高位分歧"
    return "其他"


def _final_conclusion(
    up_trend: str,
    down_trend: str,
    selling_pressure: str,
    absorption: str,
    benchmark_context: dict[str, object] | None = None,
) -> str:
    market_note = _market_context_note(benchmark_context)
    if up_trend == "提高" and down_trend == "下降":
        return (
            "最近10日出现上涨效率提高、下跌效率降低的组合，说明上涨相对越来越容易、下跌相对越来越困难。"
            f"若同时伴随承接增强或卖压衰竭，属于短线转强信号。{market_note}"
        )
    if up_trend == "下降" and down_trend == "提高":
        return (
            "最近10日表现为上涨效率下降、下跌效率提高，说明上涨越来越费力、下跌越来越容易。"
            f"短线结构偏弱，不宜仅凭缩量判断卖压衰竭。{market_note}"
        )
    if selling_pressure == "衰竭" and absorption == "增强":
        return (
            "最近下跌幅度和换手同步收敛，同时收盘位置改善，卖压有衰竭迹象且承接增强。"
            f"结构有改善，但仍需观察后续上涨效率能否继续提高。{market_note}"
        )
    return f"最近10日上涨/下跌效率未形成单边强信号，当前更适合观察量能、换手和收盘位置是否继续改善。{market_note}"


def _market_context_note(benchmark_context: dict[str, object] | None) -> str:
    if not benchmark_context or benchmark_context.get("status") != "available":
        return ""
    strength = benchmark_context.get("strength")
    relative = benchmark_context.get("relative_strength")
    if strength == "强" and relative != "强于大盘":
        return " 但当前大盘偏强，需警惕个股只是跟随市场反弹。"
    if strength == "弱" and relative == "强于大盘":
        return " 当前大盘偏弱但个股强于大盘，若量价继续改善，信号质量更高。"
    if strength == "弱":
        return " 当前大盘偏弱，短线信号需要降低仓位或等待确认。"
    return ""


def _format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _format_num(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
