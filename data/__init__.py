"""Data services for research and backtesting."""

from .daily_market import AShareDailyMarketService, DailyMarketRequest
from .tushare_client import TushareClient, TushareClientConfig
from .tushare_specs import TUSHARE_A_STOCK_DAILY_SPECS

__all__ = [
    "AShareDailyMarketService",
    "DailyMarketRequest",
    "TushareClient",
    "TushareClientConfig",
    "TUSHARE_A_STOCK_DAILY_SPECS",
]
