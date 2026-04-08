"""Thin Tushare client wrapper with token, retry, and rate limiting."""

from __future__ import annotations

import importlib
import os
import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _load_tushare_sdk():
    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    try:
        site_packages = [path for path in site.getsitepackages() if path not in sys.path]
        sys.path = site_packages + [path for path in sys.path if Path(path or ".").resolve() != repo_root]
        return importlib.import_module("tushare")
    finally:
        sys.path = original_path


ts = _load_tushare_sdk()


class TushareClientError(RuntimeError):
    """Raised when Tushare API calls fail after retries."""


@dataclass(frozen=True)
class TushareClientConfig:
    token: str | None = None
    min_interval_seconds: float = 0.35
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


class TushareClient:
    """Centralized Tushare Pro client."""

    def __init__(self, config: TushareClientConfig | None = None) -> None:
        self.config = config or TushareClientConfig()
        self.token = self._resolve_token(self.config.token)
        self._pro = ts.pro_api(self.token)
        self._last_call_at = 0.0

    @staticmethod
    def _resolve_token(explicit_token: str | None) -> str:
        token = explicit_token or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
        if not token:
            raise ValueError(
                "Missing Tushare token. Set TUSHARE_TOKEN or TS_TOKEN in the environment."
            )
        return token

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call_at
        sleep_seconds = self.config.min_interval_seconds - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _call_with_retry(self, func: Callable[..., pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._throttle()
                result = func(**kwargs)
                self._last_call_at = time.monotonic()
                if result is None:
                    return pd.DataFrame()
                return result
            except Exception as exc:  # pragma: no cover - depends on remote API
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
        raise TushareClientError(f"Tushare API call failed after retries: {last_error}") from last_error

    def daily(
        self,
        *,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.daily,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def adj_factor(
        self,
        *,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.adj_factor,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def daily_basic(
        self,
        *,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.daily_basic,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def index_daily(
        self,
        *,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.index_daily,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def fina_indicator(
        self,
        *,
        ts_code: str,
        ann_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.fina_indicator,
            ts_code=ts_code,
            ann_date=ann_date,
            start_date=start_date,
            end_date=end_date,
            period=period,
            fields=fields,
        )

    def trade_cal(
        self,
        *,
        exchange: str = "SSE",
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: str | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        return self._call_with_retry(
            self._pro.trade_cal,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open=is_open,
            fields=fields,
        )

    def pro_bar(
        self,
        *,
        ts_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        adj: str | None = None,
        freq: str = "D",
        asset: str = "E",
    ) -> pd.DataFrame:
        return self._call_with_retry(
            ts.pro_bar,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            adj=adj,
            freq=freq,
            asset=asset,
        )
