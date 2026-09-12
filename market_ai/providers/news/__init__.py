"""News provider interfaces."""

from market_ai.providers.news.base import NewsProvider
from market_ai.providers.news.local_csv import LocalCsvNewsProvider

__all__ = ["LocalCsvNewsProvider", "NewsProvider"]
