"""Archive daily consensus recommendations for the next trading day."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from data.fundamental_daily import AShareFundamentalService
from research.recommend import generate_daily_consensus_recommendations, render_consensus_recommendation_text


DEFAULT_DAILY_CONSENSUS_PROFILES = {
    "hs300": {
        "run_dir": "research_runs/hs300_turnover_momentum_stage2",
        "core_models": ["hstm2_04_hs300_turnover50_ret60_40_rev5_10_top10"],
        "confirm_models": ["hstm2_02_hs300_turnover60_ret60_30_rev5_10_top15"],
        "watch_models": ["hstm2_07_hs300_turnover40_ret60_40_rev5_20_top10"],
        "mode": "trade",
    },
    "zz500": {
        "run_dir": "research_runs/zz500_value_stage3",
        "core_models": ["zz5v3_01_ep_ttm_top10"],
        "confirm_models": ["zz5v3_11_ep50_bp20_ret60_30_top20"],
        "watch_models": ["zz5v3_02_ep_ttm_top20"],
        "mode": "trade",
    },
    "zz1000": {
        "run_dir": "research_runs/zz1000_structure_stage4",
        "core_models": ["st4_01_zz1000_position_safety_top50"],
        "confirm_models": ["st4_07_zz1000_turnover_stability_close_high_safety_top50"],
        "watch_models": [],
        "mode": "watch-only",
    },
}


RECOMMENDATION_CSV_COLUMNS = [
    "signal_date",
    "target_trade_date",
    "universe",
    "archive_mode",
    "bucket",
    "recommend_level",
    "ts_code",
    "consensus_count",
    "source_model_count",
    "source_models",
    "best_rank",
    "avg_rank",
    "avg_score",
    "in_core_model",
    "in_confirm_model",
    "in_watch_model",
]


def default_archive_dir() -> Path:
    """Return the default macOS Documents archive directory."""

    return Path.home() / "Documents" / "A-Stock" / "daily_recommendations"


def today_yyyymmdd() -> str:
    """Return today's date in Asia/Shanghai as YYYYMMDD."""

    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")


def resolve_next_trading_day(
    signal_date: str,
    *,
    calendar_service: AShareFundamentalService | None = None,
) -> str | None:
    """Return the next open trading day after signal_date, or None if signal_date is closed."""

    service = calendar_service or AShareFundamentalService()
    start_date = signal_date
    end_date = (datetime.strptime(signal_date, "%Y%m%d") + timedelta(days=14)).strftime("%Y%m%d")
    calendar = service.get_trade_calendar(start_date=start_date, end_date=end_date)
    if calendar.empty:
        raise ValueError(f"No trade calendar rows available for signal_date={signal_date}")

    open_dates = calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].astype(str).sort_values().tolist()
    if signal_date not in open_dates:
        return None

    later = [date for date in open_dates if date > signal_date]
    if not later:
        raise ValueError(f"No next trading day found after signal_date={signal_date}")
    return later[0]


def _bucket_from_level(recommend_level: object) -> str:
    if recommend_level == "A":
        return "trade_consensus"
    if recommend_level == "B":
        return "trade_core"
    if recommend_level == "C":
        return "watch_list"
    return ""


def _csv_value(value: object) -> object:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def _fallback_recommendations_from_buckets(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, object]] = set()
    for bucket_name in ["trade_consensus", "trade_core", "watch_list"]:
        for item in report.get(bucket_name, []):
            if not isinstance(item, dict):
                continue
            key = (item.get("ts_code"), item.get("recommend_level"))
            if key in seen:
                continue
            seen.add(key)
            row = dict(item)
            row["bucket"] = bucket_name
            rows.append(row)
    return rows


def build_recommendation_csv_rows(universe_name: str, report: dict[str, object]) -> list[dict[str, object]]:
    """Build flat CSV rows from a consensus recommendation report."""

    recommendations = report.get("all_recommendations", [])
    if not recommendations:
        recommendations = _fallback_recommendations_from_buckets(report)

    rows: list[dict[str, object]] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        source_models = item.get("source_models", [])
        source_model_count = item.get("source_model_count")
        if source_model_count is None and isinstance(source_models, list):
            source_model_count = len(source_models)

        row = {
            "signal_date": report.get("signal_date"),
            "target_trade_date": report.get("target_trade_date"),
            "universe": universe_name,
            "archive_mode": report.get("archive_mode"),
            "bucket": item.get("bucket") or _bucket_from_level(item.get("recommend_level")),
            "recommend_level": item.get("recommend_level"),
            "ts_code": item.get("ts_code"),
            "consensus_count": item.get("consensus_count"),
            "source_model_count": source_model_count,
            "source_models": source_models,
            "best_rank": item.get("best_rank"),
            "avg_rank": item.get("avg_rank"),
            "avg_score": item.get("avg_score"),
            "in_core_model": item.get("in_core_model"),
            "in_confirm_model": item.get("in_confirm_model"),
            "in_watch_model": item.get("in_watch_model"),
        }
        rows.append({column: _csv_value(row.get(column)) for column in RECOMMENDATION_CSV_COLUMNS})
    return rows


def write_recommendation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write recommendation rows with a stable header, even when empty."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RECOMMENDATION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def infer_consensus_universe_name(report: dict[str, object], fallback: str = "consensus") -> str:
    """Infer a single universe name from a recommendation report."""

    universe_names = {
        str(item.get("universe_name"))
        for item in report.get("selected_models", [])
        if isinstance(item, dict) and item.get("universe_name")
    }
    if len(universe_names) == 1:
        return next(iter(universe_names))
    return fallback


def archive_consensus_recommendation_csv(
    report: dict[str, object],
    *,
    universe_name: str | None = None,
    archive_dir: str | Path | None = None,
    archive_date: str | None = None,
    filename: str | None = None,
) -> dict[str, object]:
    """Archive one consensus recommendation report as a CSV file."""

    resolved_archive_date = archive_date or str(report.get("target_trade_date") or report.get("as_of_date") or "")
    if not resolved_archive_date:
        raise ValueError("archive_date is required when report has no as_of_date or target_trade_date")

    resolved_universe = universe_name or infer_consensus_universe_name(report)
    resolved_archive_dir = Path(archive_dir) if archive_dir is not None else default_archive_dir()
    target_dir = resolved_archive_dir / resolved_archive_date
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = filename or f"{resolved_universe}.csv"
    csv_path = target_dir / csv_filename
    csv_rows = build_recommendation_csv_rows(resolved_universe, report)
    write_recommendation_csv(csv_path, csv_rows)
    return {
        "status": "completed",
        "archive_date": resolved_archive_date,
        "archive_path": str(target_dir),
        "csv_path": str(csv_path),
        "csv_row_count": len(csv_rows),
        "universe": resolved_universe,
    }


def archive_daily_consensus_recommendations(
    *,
    signal_date: str | None = None,
    archive_dir: str | Path | None = None,
    profiles: dict[str, dict[str, object]] | None = None,
    calendar_service: AShareFundamentalService | None = None,
) -> dict[str, object]:
    """Generate and archive HS300/ZZ500/ZZ1000 consensus recommendations.

    The recommendation signal uses signal_date close data. The archive folder is
    named by the next trading day because that is the intended execution date.
    """

    resolved_signal_date = signal_date or today_yyyymmdd()
    target_trade_date = resolve_next_trading_day(
        resolved_signal_date,
        calendar_service=calendar_service,
    )
    if target_trade_date is None:
        return {
            "status": "skipped",
            "reason": "signal_date_is_not_open",
            "signal_date": resolved_signal_date,
            "target_trade_date": None,
            "archive_path": None,
            "universes": {},
        }

    resolved_archive_dir = Path(archive_dir) if archive_dir is not None else default_archive_dir()
    target_dir = resolved_archive_dir / target_trade_date
    target_dir.mkdir(parents=True, exist_ok=True)

    active_profiles = profiles or DEFAULT_DAILY_CONSENSUS_PROFILES
    universe_reports: dict[str, dict[str, object]] = {}
    all_csv_rows: list[dict[str, object]] = []
    for universe_name, profile in active_profiles.items():
        report = generate_daily_consensus_recommendations(
            profile["run_dir"],
            as_of_date=resolved_signal_date,
            core_model_names=list(profile.get("core_models", [])),
            confirm_model_names=list(profile.get("confirm_models", [])),
            watch_model_names=list(profile.get("watch_models", [])),
        )
        report["signal_date"] = resolved_signal_date
        report["target_trade_date"] = target_trade_date
        report["archive_mode"] = profile.get("mode", "trade")
        universe_reports[universe_name] = report

        (target_dir / f"{universe_name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (target_dir / f"{universe_name}.txt").write_text(
            render_consensus_recommendation_text(report),
            encoding="utf-8",
        )
        csv_rows = build_recommendation_csv_rows(universe_name, report)
        write_recommendation_csv(target_dir / f"{universe_name}.csv", csv_rows)
        all_csv_rows.extend(csv_rows)

    write_recommendation_csv(target_dir / "all_consensus.csv", all_csv_rows)

    manifest = {
        "status": "completed",
        "signal_date": resolved_signal_date,
        "target_trade_date": target_trade_date,
        "archive_path": str(target_dir),
        "combined_csv_path": "all_consensus.csv",
        "universes": {
            name: {
                "mode": report.get("archive_mode"),
                "trade_consensus_count": len(report.get("trade_consensus", [])),
                "trade_core_count": len(report.get("trade_core", [])),
                "watch_list_count": len(report.get("watch_list", [])),
                "csv_path": f"{name}.csv",
            }
            for name, report in universe_reports.items()
        },
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary_lines = [
        f"signal_date: {resolved_signal_date}",
        f"target_trade_date: {target_trade_date}",
        f"archive_path: {target_dir}",
        "",
        "universes:",
    ]
    for universe_name, summary in manifest["universes"].items():
        summary_lines.append(
            f"{universe_name}: mode={summary['mode']} "
            f"trade_consensus={summary['trade_consensus_count']} "
            f"trade_core={summary['trade_core_count']} "
            f"watch_list={summary['watch_list_count']}"
        )
    (target_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return {**manifest, "reports": universe_reports}
