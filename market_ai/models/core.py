"""JSON-friendly core models for daily market theme radar outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _require_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(value: float, field_name: str = "confidence") -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return number


def _score(value: float, field_name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100.")
    return number


def _int_score(value: int, field_name: str) -> int:
    number = int(value)
    if not 1 <= number <= 5:
        raise ValueError(f"{field_name} must be between 1 and 5.")
    return number


def _text_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


@dataclass
class JsonModel:
    """Mixin for deterministic JSON-compatible dict conversion."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LimitStock(JsonModel):
    """Unified row for limit-up, touched-limit, and failed-limit stocks."""

    trade_date: str
    stock_code: str
    stock_name: str
    close: float | None = None
    pct_chg: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None
    first_limit_time: str | None = None
    last_limit_time: str | None = None
    open_count: int | None = None
    sealed_amount: float | None = None
    consecutive_limit_count: int | None = None
    limit_reason: str | None = None
    industry: str | None = None
    concepts: list[str] = field(default_factory=list)
    status: str = "limit_up"
    source: str = ""

    def __post_init__(self) -> None:
        self.trade_date = _require_text(self.trade_date, "trade_date")
        self.stock_code = _require_text(self.stock_code, "stock_code")
        self.stock_name = _require_text(self.stock_name, "stock_name")
        self.status = _require_text(self.status, "status")
        self.source = _optional_text(self.source) or ""
        self.concepts = _text_list(self.concepts)


@dataclass
class StrongStock(JsonModel):
    """Unified row for strong non-limit stocks included in the daily radar."""

    trade_date: str
    stock_code: str
    stock_name: str
    pct_chg: float
    close: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None
    industry: str | None = None
    concepts: list[str] = field(default_factory=list)
    source: str = ""

    def __post_init__(self) -> None:
        self.trade_date = _require_text(self.trade_date, "trade_date")
        self.stock_code = _require_text(self.stock_code, "stock_code")
        self.stock_name = _require_text(self.stock_name, "stock_name")
        self.pct_chg = float(self.pct_chg)
        self.source = _optional_text(self.source) or ""
        self.concepts = _text_list(self.concepts)


@dataclass
class ThemeNormalization(JsonModel):
    """Schema-checked theme normalization for one stock or event reason."""

    stock_code: str
    primary_theme: str
    secondary_themes: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    event_driven: bool = False
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        self.stock_code = _require_text(self.stock_code, "stock_code")
        self.primary_theme = _require_text(self.primary_theme, "primary_theme")
        self.secondary_themes = _text_list(self.secondary_themes)
        self.related_entities = _text_list(self.related_entities)
        self.event_driven = bool(self.event_driven)
        self.confidence = _confidence(self.confidence)
        self.source = _optional_text(self.source) or ""


@dataclass
class NewsItem(JsonModel):
    """Raw news item from one provider before AI event analysis."""

    news_id: str
    source: str
    title: str
    published_at: str
    url: str | None = None
    content: str | None = None

    def __post_init__(self) -> None:
        self.news_id = _require_text(self.news_id, "news_id")
        self.source = _require_text(self.source, "source")
        self.title = _require_text(self.title, "title")
        self.published_at = _require_text(self.published_at, "published_at")
        self.url = _optional_text(self.url)
        self.content = _optional_text(self.content)


@dataclass
class NewsEventAnalysis(JsonModel):
    """Structured AI output for a news or policy event."""

    event: str
    event_type: str
    event_time: str
    themes: list[str] = field(default_factory=list)
    related_stocks: list[str] = field(default_factory=list)
    importance: int = 1
    novelty: int = 1
    scope: str = ""
    expected_duration: str = ""
    confidence: float = 0.0
    source_news_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.event = _require_text(self.event, "event")
        self.event_type = _require_text(self.event_type, "event_type")
        self.event_time = _require_text(self.event_time, "event_time")
        self.themes = _text_list(self.themes)
        self.related_stocks = _text_list(self.related_stocks)
        self.importance = _int_score(self.importance, "importance")
        self.novelty = _int_score(self.novelty, "novelty")
        self.scope = _optional_text(self.scope) or ""
        self.expected_duration = _optional_text(self.expected_duration) or ""
        self.confidence = _confidence(self.confidence)
        self.source_news_ids = _text_list(self.source_news_ids)


@dataclass
class ThemeScoreResult(JsonModel):
    """Daily configurable ThemeScore output for one normalized theme."""

    trade_date: str
    theme: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    rank: int | None = None
    lifecycle_stage: int | None = None
    validation_state: str | None = None
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trade_date = _require_text(self.trade_date, "trade_date")
        self.theme = _require_text(self.theme, "theme")
        self.score = _score(self.score, "score")
        self.components = {str(key): float(value) for key, value in self.components.items()}
        if self.rank is not None:
            self.rank = int(self.rank)
        if self.lifecycle_stage is not None:
            self.lifecycle_stage = int(self.lifecycle_stage)
            if not 0 <= self.lifecycle_stage <= 7:
                raise ValueError("lifecycle_stage must be between 0 and 7.")
        self.validation_state = _optional_text(self.validation_state)
        self.reasons = _text_list(self.reasons)


@dataclass
class StockRole(JsonModel):
    """Role classification for one stock inside a core theme."""

    trade_date: str
    theme: str
    stock_code: str
    stock_name: str
    role: str
    confidence: float
    reason: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trade_date = _require_text(self.trade_date, "trade_date")
        self.theme = _require_text(self.theme, "theme")
        self.stock_code = _require_text(self.stock_code, "stock_code")
        self.stock_name = _require_text(self.stock_name, "stock_name")
        self.role = _require_text(self.role, "role")
        self.confidence = _confidence(self.confidence)
        self.reason = _text_list(self.reason)


@dataclass
class DailyRadarReport(JsonModel):
    """Complete structured daily radar report before Markdown rendering."""

    trade_date: str
    core_themes: list[ThemeScoreResult] = field(default_factory=list)
    news_events: list[NewsEventAnalysis] = field(default_factory=list)
    unexplained_strength: list[ThemeScoreResult] = field(default_factory=list)
    stock_roles: list[StockRole] = field(default_factory=list)
    next_day_observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trade_date = _require_text(self.trade_date, "trade_date")
        self.next_day_observations = _text_list(self.next_day_observations)
