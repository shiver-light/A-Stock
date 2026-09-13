"""Config-driven theme taxonomy and synonym matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency is declared in requirements.txt
    yaml = None


DEFAULT_THEME_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "configs" / "theme_taxonomy.yaml"


@dataclass(frozen=True)
class ThemeDefinition:
    """One standard theme and its matching aliases."""

    theme: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    parent: str = ""
    description: str = ""
    include_keywords: tuple[str, ...] = field(default_factory=tuple)
    exclude_keywords: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        theme = str(self.theme).strip()
        if not theme:
            raise ValueError("theme is required.")
        aliases = tuple(dict.fromkeys(_clean_text(alias) for alias in self.aliases if _clean_text(alias)))
        include_keywords = tuple(
            dict.fromkeys(_clean_text(keyword) for keyword in self.include_keywords if _clean_text(keyword))
        )
        exclude_keywords = tuple(
            dict.fromkeys(_clean_text(keyword) for keyword in self.exclude_keywords if _clean_text(keyword))
        )
        object.__setattr__(self, "theme", theme)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "parent", str(self.parent).strip())
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "include_keywords", include_keywords)
        object.__setattr__(self, "exclude_keywords", exclude_keywords)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThemeMatch:
    """One deterministic taxonomy match from raw text to a standard theme."""

    theme: str
    matched_alias: str
    source_text: str
    parent: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThemeTaxonomy:
    """In-memory taxonomy that maps aliases to standard theme names."""

    def __init__(self, definitions: list[ThemeDefinition]) -> None:
        if not definitions:
            raise ValueError("Theme taxonomy definitions must not be empty.")
        self.definitions = {definition.theme: definition for definition in definitions}
        if len(self.definitions) != len(definitions):
            raise ValueError("Theme taxonomy contains duplicated theme names.")
        self.alias_to_theme = _build_alias_index(definitions)

    def match_text(self, text: str) -> list[ThemeMatch]:
        """Return all theme matches found in text, sorted by alias length."""
        source_text = str(text or "")
        normalized_text = _clean_text(source_text)
        if not normalized_text:
            return []

        matches = []
        for alias, theme in self.alias_to_theme.items():
            if alias in normalized_text:
                definition = self.definitions[theme]
                if not _passes_theme_filters(definition, normalized_text):
                    continue
                matches.append(
                    ThemeMatch(
                        theme=theme,
                        matched_alias=alias,
                        source_text=source_text,
                        parent=definition.parent,
                        confidence=1.0,
                    )
                )
        return sorted(matches, key=lambda item: (-len(item.matched_alias), item.theme))

    def best_match(self, text: str) -> ThemeMatch | None:
        """Return the strongest deterministic theme match for text."""
        matches = self.match_text(text)
        if not matches:
            return None
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {"themes": [definition.to_dict() for definition in self.definitions.values()]}


def load_theme_taxonomy(path: str | Path | None = None) -> ThemeTaxonomy:
    """Load a YAML theme taxonomy into a deterministic matcher."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load theme taxonomy configs.")

    taxonomy_path = Path(path) if path is not None else DEFAULT_THEME_TAXONOMY_PATH
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Theme taxonomy config file not found: {taxonomy_path}")

    with taxonomy_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Theme taxonomy config must be a mapping.")

    themes = payload.get("themes", [])
    if not isinstance(themes, list):
        raise ValueError("themes must be a list.")

    definitions = []
    for item in themes:
        if not isinstance(item, dict):
            raise ValueError("Each theme definition must be a mapping.")
        definitions.append(
            ThemeDefinition(
                theme=str(item.get("theme", "")),
                aliases=tuple(item.get("aliases", []) or []),
                parent=str(item.get("parent", "") or ""),
                description=str(item.get("description", "") or ""),
                include_keywords=tuple(item.get("include_keywords", []) or []),
                exclude_keywords=tuple(item.get("exclude_keywords", []) or []),
            )
        )
    return ThemeTaxonomy(definitions)


def _build_alias_index(definitions: list[ThemeDefinition]) -> dict[str, str]:
    alias_to_theme: dict[str, str] = {}
    for definition in definitions:
        aliases = (definition.theme, *definition.aliases)
        for alias in aliases:
            normalized = _clean_text(alias)
            if not normalized:
                continue
            existing = alias_to_theme.get(normalized)
            if existing is not None and existing != definition.theme:
                raise ValueError(f"Theme alias duplicated across themes: {alias}")
            alias_to_theme[normalized] = definition.theme
    return alias_to_theme


def _clean_text(value: object) -> str:
    return str(value).strip().lower().replace(" ", "")


def _passes_theme_filters(definition: ThemeDefinition, normalized_text: str) -> bool:
    if definition.include_keywords and not any(keyword in normalized_text for keyword in definition.include_keywords):
        return False
    if definition.exclude_keywords and any(keyword in normalized_text for keyword in definition.exclude_keywords):
        return False
    return True
