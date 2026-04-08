"""Data services for research and backtesting."""

from .daily_market import AShareDailyMarketService, DailyMarketRequest, get_a_share_daily_prices
from .fundamental_daily import AShareFundamentalService, get_a_share_fundamental_daily
from .index_daily import AShareIndexDailyService, IndexDailyRequest, get_a_share_index_daily
from .tushare_client import TushareClient, TushareClientConfig
from .tushare_specs import TUSHARE_A_STOCK_DAILY_SPECS
from .valuation_daily import AShareDailyValuationService, DailyValuationRequest, get_a_share_daily_valuation

__all__ = [
    "AShareDailyMarketService",
    "AShareDailyValuationService",
    "AShareFundamentalService",
    "AShareIndexDailyService",
    "DailyMarketRequest",
    "DailyValuationRequest",
    "IndexDailyRequest",
    "get_a_share_daily_prices",
    "get_a_share_daily_valuation",
    "get_a_share_fundamental_daily",
    "get_a_share_index_daily",
    "TushareClient",
    "TushareClientConfig",
    "TUSHARE_A_STOCK_DAILY_SPECS",
]
