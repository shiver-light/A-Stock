"""Data services for research and backtesting."""

from .daily_market import AShareDailyMarketService, DailyMarketRequest, get_a_share_daily_prices
from .fundamental_daily import AShareFundamentalService, get_a_share_fundamental_daily
from .index_daily import AShareIndexDailyService, IndexDailyRequest, get_a_share_index_daily
from .stock_universe import (
    INDEX_CODE_MAP,
    IndexConstituentRequest,
    StockUniverseDataService,
    get_index_constituents,
    get_stock_basic_history,
)
from .tushare_client import TushareClient, TushareClientConfig
from .tushare_specs import TUSHARE_A_STOCK_DAILY_SPECS
from .us_market import USDailyRequest, USMarketDailyService, build_us_market_regime, get_us_daily_prices
from .valuation_daily import AShareDailyValuationService, DailyValuationRequest, get_a_share_daily_valuation

__all__ = [
    "AShareDailyMarketService",
    "AShareDailyValuationService",
    "AShareFundamentalService",
    "AShareIndexDailyService",
    "StockUniverseDataService",
    "USMarketDailyService",
    "DailyMarketRequest",
    "USDailyRequest",
    "DailyValuationRequest",
    "IndexDailyRequest",
    "IndexConstituentRequest",
    "INDEX_CODE_MAP",
    "get_a_share_daily_prices",
    "get_a_share_daily_valuation",
    "get_a_share_fundamental_daily",
    "get_a_share_index_daily",
    "get_stock_basic_history",
    "get_index_constituents",
    "get_us_daily_prices",
    "build_us_market_regime",
    "TushareClient",
    "TushareClientConfig",
    "TUSHARE_A_STOCK_DAILY_SPECS",
]
