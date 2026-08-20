"""Recent up/down efficiency diagnostics for single-stock daily bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["trade_date", "ts_code", "close", "pre_close", "high", "low", "vol", "turnover_rate_f"]


def build_efficiency_report(
    market_data: pd.DataFrame,
    *,
    stock_label: str | None = None,
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
        "final_conclusion": _final_conclusion(up_trend, down_trend, selling_pressure, absorption),
        "summary": _summary_stats(data, recent, previous),
        "efficiency_display_scale": 10000,
        "rows": rows,
        "warnings": [],
    }


def render_efficiency_report_text(report: dict[str, object]) -> str:
    """Render an efficiency report in a human-readable Chinese text format."""

    summary = report.get("summary", {})
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
        "final_conclusion": "数据不足，暂不形成短线效率判断。",
        "summary": {},
        "efficiency_display_scale": 10000,
        "rows": [],
        "warnings": [warning],
    }


def _infer_stock_label(data: pd.DataFrame) -> str:
    ts_code = data["ts_code"].dropna().astype(str)
    return ts_code.iloc[0] if not ts_code.empty else ""


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


def _final_conclusion(up_trend: str, down_trend: str, selling_pressure: str, absorption: str) -> str:
    if up_trend == "提高" and down_trend == "下降":
        return "最近10日出现上涨效率提高、下跌效率降低的组合，说明上涨相对越来越容易、下跌相对越来越困难。若同时伴随承接增强或卖压衰竭，属于短线转强信号。"
    if up_trend == "下降" and down_trend == "提高":
        return "最近10日表现为上涨效率下降、下跌效率提高，说明上涨越来越费力、下跌越来越容易。短线结构偏弱，不宜仅凭缩量判断卖压衰竭。"
    if selling_pressure == "衰竭" and absorption == "增强":
        return "最近下跌幅度和换手同步收敛，同时收盘位置改善，卖压有衰竭迹象且承接增强。结构有改善，但仍需观察后续上涨效率能否继续提高。"
    return "最近10日上涨/下跌效率未形成单边强信号，当前更适合观察量能、换手和收盘位置是否继续改善。"


def _format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _format_num(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
