"""Configuration loading for daily market radar workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency is declared in requirements.txt
    yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "radar.yaml"
DEFAULT_THEME_SCORE_WEIGHTS = {
    "LimitUpStrength": 0.25,
    "Breadth": 0.20,
    "LeaderStrength": 0.15,
    "NewsCatalyst": 0.15,
    "VolumeExpansion": 0.10,
    "Persistence": 0.10,
    "Novelty": 0.05,
}
DEFAULT_NEWS_SEMANTIC_WEIGHTS = {
    "authority": 0.30,
    "core_theme": 0.30,
    "theme_match": 0.15,
    "directness": 0.15,
    "impact": 0.10,
    "noise_penalty": 0.20,
}


@dataclass(frozen=True)
class ProviderConfig:
    """Provider names selected by the radar config."""

    market: str = "tushare"
    news: list[str] = field(default_factory=lambda: ["local_csv"])
    llm: str = "none"


@dataclass(frozen=True)
class NewsWindowConfig:
    """News collection lookback window in hours."""

    lookback_hours: int = 36

    def __post_init__(self) -> None:
        if self.lookback_hours <= 0:
            raise ValueError("news.lookback_hours must be positive.")


@dataclass(frozen=True)
class NewsScoringConfig:
    """Configurable deterministic news scoring parameters."""

    default_authority_score: float = 50.0
    authority_scores: dict[str, float] = field(default_factory=dict)
    core_event_top_n: int = 10
    min_core_event_score: float = 60.0
    require_core_theme_match: bool = True
    semantic_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NEWS_SEMANTIC_WEIGHTS))

    def __post_init__(self) -> None:
        default_score = float(self.default_authority_score)
        if not 0.0 <= default_score <= 100.0:
            raise ValueError("news_scoring.default_authority_score must be between 0 and 100.")
        min_score = float(self.min_core_event_score)
        if not 0.0 <= min_score <= 100.0:
            raise ValueError("news_scoring.min_core_event_score must be between 0 and 100.")
        top_n = int(self.core_event_top_n)
        if top_n <= 0:
            raise ValueError("news_scoring.core_event_top_n must be positive.")
        scores = {str(source).strip(): float(score) for source, score in self.authority_scores.items()}
        if any(not source for source in scores):
            raise ValueError("news_scoring.authority_scores source must not be empty.")
        if any(score < 0.0 or score > 100.0 for score in scores.values()):
            raise ValueError("news_scoring.authority_scores must be between 0 and 100.")
        semantic_weights = {str(key).strip(): float(value) for key, value in self.semantic_weights.items()}
        if any(not key for key in semantic_weights):
            raise ValueError("news_scoring.semantic_weights key must not be empty.")
        if any(value < 0.0 for value in semantic_weights.values()):
            raise ValueError("news_scoring.semantic_weights must be non-negative.")
        if sum(semantic_weights.values()) <= 0.0:
            raise ValueError("news_scoring.semantic_weights sum must be positive.")
        object.__setattr__(self, "default_authority_score", default_score)
        object.__setattr__(self, "authority_scores", scores)
        object.__setattr__(self, "core_event_top_n", top_n)
        object.__setattr__(self, "min_core_event_score", min_score)
        object.__setattr__(self, "require_core_theme_match", bool(self.require_core_theme_match))
        object.__setattr__(self, "semantic_weights", semantic_weights)


@dataclass(frozen=True)
class ThemeScoreConfig:
    """Configurable ThemeScore weights."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THEME_SCORE_WEIGHTS))

    def __post_init__(self) -> None:
        normalized = {str(key): float(value) for key, value in self.weights.items()}
        if not normalized:
            raise ValueError("theme_score.weights must not be empty.")
        if any(value < 0.0 for value in normalized.values()):
            raise ValueError("theme_score.weights must be non-negative.")
        if sum(normalized.values()) <= 0.0:
            raise ValueError("theme_score.weights sum must be positive.")
        object.__setattr__(self, "weights", normalized)

    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {key: value / total for key, value in self.weights.items()}


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem output paths for radar artifacts."""

    base_dir: str = "market_ai_runs"
    report_dir: str = "reports/market_radar"
    cache_dir: str = "data/cache/market_ai"


@dataclass(frozen=True)
class RadarConfig:
    """Top-level daily market radar configuration."""

    providers: ProviderConfig = field(default_factory=ProviderConfig)
    news: NewsWindowConfig = field(default_factory=NewsWindowConfig)
    news_scoring: NewsScoringConfig = field(default_factory=NewsScoringConfig)
    theme_score: ThemeScoreConfig = field(default_factory=ThemeScoreConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    strong_stock_min_pct_chg: float = 7.0
    top_theme_count: int = 5

    def __post_init__(self) -> None:
        if self.strong_stock_min_pct_chg <= 0.0:
            raise ValueError("strong_stock_min_pct_chg must be positive.")
        if self.top_theme_count <= 0:
            raise ValueError("top_theme_count must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_market_radar_config(path: str | Path | None = None) -> RadarConfig:
    """Load a market radar YAML config into typed dataclasses."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Market radar config file not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load market radar configs.")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Market radar config must be a mapping.")
    return _build_radar_config(payload)


def _build_radar_config(payload: dict[str, Any]) -> RadarConfig:
    providers = payload.get("providers", {}) or {}
    news = payload.get("news", {}) or {}
    news_scoring = payload.get("news_scoring", {}) or {}
    theme_score = payload.get("theme_score", {}) or {}
    output = payload.get("output", {}) or {}
    if not isinstance(providers, dict):
        raise ValueError("providers must be a mapping.")
    if not isinstance(news, dict):
        raise ValueError("news must be a mapping.")
    if not isinstance(news_scoring, dict):
        raise ValueError("news_scoring must be a mapping.")
    if not isinstance(theme_score, dict):
        raise ValueError("theme_score must be a mapping.")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping.")

    return RadarConfig(
        providers=ProviderConfig(
            market=str(providers.get("market", "tushare")),
            news=list(providers.get("news", ["local_csv"]) or []),
            llm=str(providers.get("llm", "none")),
        ),
        news=NewsWindowConfig(lookback_hours=int(news.get("lookback_hours", 36))),
        news_scoring=NewsScoringConfig(
            default_authority_score=float(news_scoring.get("default_authority_score", 50.0)),
            authority_scores=dict(news_scoring.get("authority_scores", {}) or {}),
            core_event_top_n=int(news_scoring.get("core_event_top_n", 10)),
            min_core_event_score=float(news_scoring.get("min_core_event_score", 60.0)),
            require_core_theme_match=bool(news_scoring.get("require_core_theme_match", True)),
            semantic_weights=dict(news_scoring.get("semantic_weights", DEFAULT_NEWS_SEMANTIC_WEIGHTS) or DEFAULT_NEWS_SEMANTIC_WEIGHTS),
        ),
        theme_score=ThemeScoreConfig(
            weights=dict(theme_score.get("weights", DEFAULT_THEME_SCORE_WEIGHTS) or DEFAULT_THEME_SCORE_WEIGHTS)
        ),
        output=OutputConfig(
            base_dir=str(output.get("base_dir", "market_ai_runs")),
            report_dir=str(output.get("report_dir", "reports/market_radar")),
            cache_dir=str(output.get("cache_dir", "data/cache/market_ai")),
        ),
        strong_stock_min_pct_chg=float(payload.get("strong_stock_min_pct_chg", 7.0)),
        top_theme_count=int(payload.get("top_theme_count", 5)),
    )
