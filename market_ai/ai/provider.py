"""OpenAI-compatible LLM provider for structured news event analysis."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from market_ai.config import LLMConfig
from market_ai.models import NewsEventAnalysis, NewsItem
from market_ai.themes import RuleBasedThemeNormalizer, ThemeTaxonomy


SYSTEM_PROMPT = """你是A股市场主线雷达的新闻事件结构化分析器。
只输出一个JSON对象，不要输出解释、Markdown或代码块。
所有themes必须从allowed_themes中选择；无法判断时返回空themes。
不要预测涨跌，只判断新闻事件对应题材、重要性、新鲜度、影响范围和置信度。
importance和novelty取1到5的整数，confidence取0到1。
"""


@dataclass(frozen=True)
class LLMNewsAnalysisWarning:
    """Auditable LLM fallback warning for one news item."""

    news_id: str
    title: str
    reason: str
    fallback_used: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "news_id": self.news_id,
            "title": self.title,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class OpenAICompatibleLLMProvider:
    """Minimal OpenAI-compatible chat completions client."""

    config: LLMConfig

    def analyze_news_item(self, news: NewsItem, taxonomy: ThemeTaxonomy) -> NewsEventAnalysis | None:
        """Analyze one news item and return schema-checked event data."""
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_news_prompt(news, taxonomy)},
            ],
        }
        response = self._post_json("/chat/completions", payload)
        content = _extract_message_content(response)
        parsed = _extract_json_object(content)
        return _coerce_news_event(parsed, news, taxonomy)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response is not valid JSON.") from exc


def analyze_news_items_with_llm(
    news_items: list[NewsItem],
    taxonomy: ThemeTaxonomy,
    *,
    provider: OpenAICompatibleLLMProvider | None,
) -> list[NewsEventAnalysis]:
    """Analyze news with an optional LLM and fall back to deterministic taxonomy rules."""
    events, _warnings = analyze_news_items_with_llm_with_warnings(news_items, taxonomy, provider=provider)
    return events


def analyze_news_items_with_llm_with_warnings(
    news_items: list[NewsItem],
    taxonomy: ThemeTaxonomy,
    *,
    provider: OpenAICompatibleLLMProvider | None,
) -> tuple[list[NewsEventAnalysis], list[LLMNewsAnalysisWarning]]:
    """Analyze news with LLM and return explicit fallback warnings."""
    rule_normalizer = RuleBasedThemeNormalizer(taxonomy)
    events = []
    warnings = []
    for news in news_items:
        event = None
        if provider is not None:
            try:
                event = provider.analyze_news_item(news, taxonomy)
                if event is None:
                    fallback = rule_normalizer.normalize_news_item(news)
                    warnings.append(
                        LLMNewsAnalysisWarning(
                            news_id=news.news_id,
                            title=news.title,
                            reason="LLM output did not produce a valid known-theme event.",
                            fallback_used=fallback is not None,
                        )
                    )
                    event = fallback
            except RuntimeError as exc:
                fallback = rule_normalizer.normalize_news_item(news)
                warnings.append(
                    LLMNewsAnalysisWarning(
                        news_id=news.news_id,
                        title=news.title,
                        reason=str(exc),
                        fallback_used=fallback is not None,
                    )
                )
                event = fallback
        else:
            event = rule_normalizer.normalize_news_item(news)
        if event is None:
            continue
        events.append(event)
    return events, warnings


def _build_news_prompt(news: NewsItem, taxonomy: ThemeTaxonomy) -> str:
    allowed_themes = sorted(taxonomy.definitions)
    return json.dumps(
        {
            "task": "classify_a_share_market_news",
            "schema": {
                "event": "string",
                "event_type": "policy|price_hike|order|capacity_expansion|industry_research|earnings|overseas|market_summary|single_stock|other",
                "event_time": "string",
                "themes": ["allowed theme names only"],
                "related_stocks": ["ts_code or stock name"],
                "importance": "integer 1-5",
                "novelty": "integer 1-5",
                "scope": "industry|theme|single_stock|macro|overseas|unknown",
                "expected_duration": "short|medium|long|unknown",
                "confidence": "float 0-1",
            },
            "allowed_themes": allowed_themes,
            "news": news.to_dict(),
        },
        ensure_ascii=False,
    )


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response missing choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("LLM response missing message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response content is empty.")
    return content


def _extract_json_object(content: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("LLM response does not contain a JSON object.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response JSON object is invalid.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM response JSON must be an object.")
    return payload


def _coerce_news_event(payload: dict[str, Any], news: NewsItem, taxonomy: ThemeTaxonomy) -> NewsEventAnalysis | None:
    allowed = set(taxonomy.definitions)
    raw_themes = payload.get("themes", [])
    if not isinstance(raw_themes, list):
        raw_themes = []
    themes = [str(theme).strip() for theme in raw_themes if str(theme).strip() in allowed]
    if not themes:
        return None
    try:
        return NewsEventAnalysis(
            event=str(payload.get("event") or news.title),
            event_type=str(payload.get("event_type") or "news"),
            event_time=str(payload.get("event_time") or news.published_at),
            themes=list(dict.fromkeys(themes)),
            related_stocks=_text_list(payload.get("related_stocks")),
            importance=_bounded_int(payload.get("importance"), minimum=1, maximum=5, default=1),
            novelty=_bounded_int(payload.get("novelty"), minimum=1, maximum=5, default=1),
            scope=str(payload.get("scope") or ""),
            expected_duration=str(payload.get("expected_duration") or ""),
            confidence=_bounded_float(payload.get("confidence"), minimum=0.0, maximum=1.0, default=0.5),
            source_news_ids=[news.news_id],
            theme_evidence=[
                {
                    "theme": theme,
                    "matched_keyword": "llm",
                    "source": "llm",
                    "source_text": news.title,
                    "confidence": _bounded_float(payload.get("confidence"), minimum=0.0, maximum=1.0, default=0.5),
                }
                for theme in list(dict.fromkeys(themes))
            ],
        )
    except ValueError:
        return None


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_int(value: object, *, minimum: int, maximum: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(maximum, max(minimum, number))


def _bounded_float(value: object, *, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(maximum, max(minimum, number))
