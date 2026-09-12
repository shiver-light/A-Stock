"""Abstract market data provider interface for daily radar workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from market_ai.models import LimitStock, StrongStock


class MarketProvider(ABC):
    """Source-agnostic interface for daily A-share market radar inputs.

    Implementations may use Tushare, local CSV, or another source, but business
    logic should depend only on this interface. All methods must use data known
    by the requested trade_date close unless a concrete implementation documents
    a stricter availability rule.
    """

    @abstractmethod
    def get_limit_stocks(self, *, trade_date: str) -> list[LimitStock]:
        """Return final limit-up, touched-limit, and failed-limit stocks for one trade date."""

    @abstractmethod
    def get_strong_stocks(self, *, trade_date: str, min_pct_chg: float = 7.0) -> list[StrongStock]:
        """Return strong non-limit stocks, typically pct_chg >= min_pct_chg."""

    @abstractmethod
    def get_daily_quotes(self, *, trade_date: str, stock_codes: Iterable[str] | None = None) -> pd.DataFrame:
        """Return same-day quote and liquidity fields for stocks.

        Expected columns are source dependent, but providers should prefer:
        trade_date, stock_code, stock_name, close, pct_chg, amount,
        turnover_rate, volume_ratio, industry, concepts.
        """
