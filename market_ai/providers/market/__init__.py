"""Market data provider interfaces."""

from market_ai.providers.market.base import MarketProvider
from market_ai.providers.market.local_csv import LocalCsvMarketProvider

__all__ = ["LocalCsvMarketProvider", "MarketProvider"]
