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
    theme_score = payload.get("theme_score", {}) or {}
    output = payload.get("output", {}) or {}
    if not isinstance(providers, dict):
        raise ValueError("providers must be a mapping.")
    if not isinstance(news, dict):
        raise ValueError("news must be a mapping.")
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
