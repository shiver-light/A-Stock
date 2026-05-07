"""Archive daily consensus recommendations for the next trading day."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from data.fundamental_daily import AShareFundamentalService
from research.recommend import generate_daily_consensus_recommendations, render_consensus_recommendation_text


DEFAULT_DAILY_CONSENSUS_PROFILES = {
    "hs300": {
        "run_dir": "research_runs/stage2_candidates",
        "core_models": ["c01_hs300_turnover_top10"],
        "confirm_models": ["c03_hs300_turnover_ret60_70_30_top20"],
        "watch_models": [],
        "mode": "trade",
    },
    "zz500": {
        "run_dir": "research_runs/zz500_stage2_min300k",
        "core_models": ["s2_m01_zz500_turnover_top10"],
        "confirm_models": ["s2_m04_zz500_ep_ttm_top10"],
        "watch_models": [],
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

    manifest = {
        "status": "completed",
        "signal_date": resolved_signal_date,
        "target_trade_date": target_trade_date,
        "archive_path": str(target_dir),
        "universes": {
            name: {
                "mode": report.get("archive_mode"),
                "trade_consensus_count": len(report.get("trade_consensus", [])),
                "trade_core_count": len(report.get("trade_core", [])),
                "watch_list_count": len(report.get("watch_list", [])),
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
