"""Deterministic news authority scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorityScoreResult:
    """Authority score with an auditable reason."""

    source: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "score": self.score, "reason": self.reason}


def calculate_authority_score(
    source: str,
    *,
    authority_scores: dict[str, float],
    default_score: float = 50.0,
) -> AuthorityScoreResult:
    """Calculate a 0-100 authority score from source configuration."""
    normalized_source = str(source or "").strip()
    if not normalized_source:
        return AuthorityScoreResult(source="", score=float(default_score), reason="empty_source_default")
    if normalized_source in authority_scores:
        return AuthorityScoreResult(
            source=normalized_source,
            score=float(authority_scores[normalized_source]),
            reason="configured_source_score",
        )
    return AuthorityScoreResult(source=normalized_source, score=float(default_score), reason="unknown_source_default")
