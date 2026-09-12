"""Abstract news provider interface for daily radar workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from market_ai.models import NewsItem


class NewsProvider(ABC):
    """Source-agnostic interface for recent market news and announcements.

    Implementations may read local CSV files or call remote APIs. Business logic
    should depend only on this interface, and provider implementations must not
    return provider-specific raw payloads directly to scoring or AI modules.
    """

    @abstractmethod
    def fetch_news(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        sources: Iterable[str] | None = None,
    ) -> list[NewsItem]:
        """Return normalized news items published in [start_time, end_time]."""
