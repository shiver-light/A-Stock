"""News provider interfaces."""

from market_ai.providers.news.base import NewsProvider
from market_ai.providers.news.local_csv import LocalCsvNewsProvider
from market_ai.providers.news.tushare_major import TushareMajorNewsProvider

__all__ = ["LocalCsvNewsProvider", "NewsProvider", "TushareMajorNewsProvider"]
