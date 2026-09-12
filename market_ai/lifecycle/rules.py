"""Rule-based theme lifecycle classification from historical ThemeScore."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_ai.models import ThemeScoreResult


STAGE_NAMES = {
    0: "潜伏",
    1: "启动",
    2: "发酵",
    3: "主升",
    4: "高潮",
    5: "分歧",
    6: "回流",
    7: "退潮",
}


def load_theme_score_history(path: str | Path | None) -> pd.DataFrame:
    """Load historical theme score rows from CSV."""
    if path is None:
        return _empty_history()
    history_path = Path(path)
    if not history_path.exists():
        return _empty_history()
    return _normalize_history(pd.read_csv(history_path))


def classify_theme_lifecycle(
    current_themes: list[ThemeScoreResult],
    history: pd.DataFrame,
    *,
    trade_date: str,
    lookback_days: int = 5,
) -> list[ThemeScoreResult]:
    """Assign lifecycle stages using only trade_date and earlier history."""
    if not current_themes:
        return current_themes
    normalized_history = _normalize_history(history)
    current_rows = _current_theme_rows(current_themes)
    combined = _normalize_history(pd.concat([normalized_history, current_rows], ignore_index=True))
    combined = combined.loc[combined["trade_date"].astype(str).le(str(trade_date))].copy()

    for theme in current_themes:
        theme_history = combined.loc[combined["theme"].eq(theme.theme)].sort_values("trade_date")
        recent = theme_history.tail(max(1, int(lookback_days)))
        previous = theme_history.iloc[:-1]
        stage, reasons = _classify_one(theme, recent, previous)
        theme.lifecycle_stage = stage
        theme.reasons = [*theme.reasons, *reasons]
    return current_themes


def _classify_one(
    theme: ThemeScoreResult,
    recent: pd.DataFrame,
    previous: pd.DataFrame,
) -> tuple[int, list[str]]:
    today_score = float(theme.score)
    active_days = int(recent["score"].ge(30.0).sum()) if not recent.empty else 1
    latest_rank = int(theme.rank or 9999)
    score_trend = _score_trend(recent)
    previous_active = not previous.loc[previous["score"].ge(30.0)].empty
    reasons = [
        f"近{len(recent)}日活跃天数 {active_days}",
        f"ThemeScore趋势 {score_trend:.2f}",
    ]

    if active_days >= 4 and today_score >= 85.0 and latest_rank <= 2:
        return 4, [*reasons, "高分且连续活跃，判断为高潮"]
    if active_days >= 3 and today_score >= 65.0 and latest_rank <= 3:
        return 3, [*reasons, "连续活跃且排名靠前，判断为主升"]
    if previous_active and active_days <= 1 and today_score < 30.0:
        return 7, [*reasons, "前期活跃后明显降温，判断为退潮"]
    if previous_active and score_trend >= 15.0 and today_score >= 50.0:
        return 6, [*reasons, "前期活跃后重新回升，判断为回流"]
    if previous_active and score_trend <= -15.0 and today_score >= 40.0:
        return 5, [*reasons, "仍有强度但趋势转弱，判断为分歧"]
    if active_days >= 2 and today_score >= 40.0:
        return 2, [*reasons, "连续进入活跃状态，判断为发酵"]
    if not previous_active and today_score >= 30.0:
        return 1, [*reasons, "近期首次进入活跃状态，判断为启动"]
    return 0, [*reasons, "强度较低或持续性不足，判断为潜伏"]


def _score_trend(recent: pd.DataFrame) -> float:
    if len(recent) < 2:
        return 0.0
    scores = recent["score"].astype(float).to_list()
    return scores[-1] - scores[0]


def _current_theme_rows(current_themes: list[ThemeScoreResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": theme.trade_date,
                "theme": theme.theme,
                "score": theme.score,
                "rank": theme.rank,
            }
            for theme in current_themes
        ]
    )


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return _empty_history()
    missing = [column for column in ["trade_date", "theme", "score"] if column not in history.columns]
    if missing:
        raise ValueError(f"Theme score history missing required columns: {missing}")
    result = history.copy()
    result["trade_date"] = result["trade_date"].fillna("").astype(str).str.strip().map(_clean_date_value)
    result["theme"] = result["theme"].fillna("").astype(str).str.strip()
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    if "rank" not in result.columns:
        result["rank"] = pd.NA
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    valid = result["trade_date"].ne("") & result["theme"].ne("") & result["score"].notna()
    return (
        result.loc[valid, ["trade_date", "theme", "score", "rank"]]
        .sort_values(["trade_date", "theme"])
        .drop_duplicates(subset=["trade_date", "theme"], keep="last")
        .reset_index(drop=True)
    )


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["trade_date", "theme", "score", "rank"])


def _clean_date_value(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text
