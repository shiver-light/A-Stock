"""US market data and regime helpers.

Tushare interfaces used:
- us_daily: US daily market data, single request max 6000 rows.
- us_tradecal: US trading calendar.

Time alignment:
- US close for US trade date D is not used for A-share signal date D.
- Regime rows are aligned to the next A-share trading date by using the latest
  US trade date strictly earlier than the A-share trade_date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig

US_DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "vol",
    "amount",
]


@dataclass(frozen=True)
class USDailyRequest:
    ts_code: str
    start_date: str
    end_date: str
    fields: tuple[str, ...] = tuple(US_DAILY_FIELDS)
    refresh: bool = False


class USMarketDailyService:
    """Cached wrapper for Tushare us_daily."""

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_daily(self, request: USDailyRequest) -> pd.DataFrame:
        self._validate_request(request)
        cache_path = self.cache_dir / "us_daily" / f"{request.ts_code}.parquet"
        cached = self._read_cache(cache_path)
        missing_ranges = self._compute_missing_ranges(cached, request.start_date, request.end_date, request.refresh)

        frames = [cached] if not cached.empty else []
        for start_date, end_date in missing_ranges:
            fetched = self.client.us_daily(
                ts_code=request.ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(US_DAILY_FIELDS),
            )
            frames.append(self._normalize_frame(fetched))

        merged = self._merge_frames(frames)
        self._write_cache(cache_path, merged)
        result = self._slice_dates(merged, request.start_date, request.end_date)
        return self._select_fields(result, request.fields)

    def get_a_share_open_dates(self, *, start_date: str, end_date: str) -> list[str]:
        data = self.client.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
            fields="cal_date",
        )
        if data.empty:
            return []
        return sorted(data["cal_date"].astype(str).drop_duplicates().tolist())

    def _validate_request(self, request: USDailyRequest) -> None:
        if not request.ts_code:
            raise ValueError("ts_code is required.")
        if request.start_date > request.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")

    def _compute_missing_ranges(
        self,
        cached: pd.DataFrame,
        start_date: str,
        end_date: str,
        refresh: bool,
    ) -> list[tuple[str, str]]:
        if refresh or cached.empty:
            return [(start_date, end_date)]

        ranges: list[tuple[str, str]] = []
        min_cached = str(cached["trade_date"].min())
        max_cached = str(cached["trade_date"].max())
        if start_date < min_cached:
            ranges.append((start_date, min_cached))
        if end_date > max_cached:
            ranges.append((max_cached, end_date))
        return ranges

    def _normalize_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        normalized = data.copy()
        normalized["trade_date"] = normalized["trade_date"].astype(str)
        return (
            normalized.sort_values(["trade_date", "ts_code"])
            .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _merge_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        return (
            pd.concat(valid_frames, ignore_index=True)
            .sort_values(["trade_date", "ts_code"])
            .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            .reset_index(drop=True)
        )

    def _slice_dates(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
        return data.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def _select_fields(self, data: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
        missing = [field for field in fields if field not in data.columns]
        if missing:
            raise ValueError(f"Requested fields are unavailable: {missing}")
        return data.loc[:, list(fields)].copy()

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._normalize_frame(pd.read_parquet(path))

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)


def build_us_market_regime(
    prices: pd.DataFrame,
    a_share_trade_dates: list[str],
    *,
    spy_code: str = "SPY",
    qqq_code: str = "QQQ",
    iwm_code: str = "IWM",
    vix_proxy_code: str = "VXX",
    lookback: int = 20,
    tech_relative_threshold: float = 0.0,
    smallcap_relative_threshold: float = 0.0,
    vix_stress_threshold: float = 0.1,
) -> pd.DataFrame:
    """Build A-share-date-aligned US market regime from US ETF closes."""

    if not a_share_trade_dates:
        return _empty_regime_frame()
    if prices.empty:
        return _empty_regime_frame()
    missing_columns = {"ts_code", "trade_date", "close"}.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"US prices missing required columns: {sorted(missing_columns)}")
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    wide = _build_close_wide(prices)
    required_codes = [spy_code, qqq_code, iwm_code, vix_proxy_code]
    missing_codes = [code for code in required_codes if code not in wide.columns]
    if missing_codes:
        raise ValueError(f"US prices missing required ts_code data: {missing_codes}")

    qqq_return = wide[qqq_code].pct_change(lookback)
    spy_return = wide[spy_code].pct_change(lookback)
    iwm_return = wide[iwm_code].pct_change(lookback)
    vix_return = wide[vix_proxy_code].pct_change(lookback)
    qqq_spy_relative = (wide[qqq_code] / wide[spy_code]).pct_change(lookback)
    iwm_spy_relative = (wide[iwm_code] / wide[spy_code]).pct_change(lookback)

    us_regime = pd.DataFrame(
        {
            "us_trade_date": wide.index.astype(str),
            "spy_return_20d": spy_return.to_numpy(),
            "qqq_return_20d": qqq_return.to_numpy(),
            "iwm_return_20d": iwm_return.to_numpy(),
            "vix_return_20d": vix_return.to_numpy(),
            "qqq_spy_relative_20d": qqq_spy_relative.to_numpy(),
            "iwm_spy_relative_20d": iwm_spy_relative.to_numpy(),
        }
    )
    us_regime["us_risk_on"] = (us_regime["spy_return_20d"] >= 0.0) & (
        us_regime["vix_return_20d"] <= vix_stress_threshold
    )
    us_regime["us_tech_strong"] = us_regime["qqq_spy_relative_20d"] >= tech_relative_threshold
    us_regime["us_smallcap_strong"] = us_regime["iwm_spy_relative_20d"] >= smallcap_relative_threshold
    us_regime["vix_stress"] = us_regime["vix_return_20d"] > vix_stress_threshold

    aligned = _align_to_a_share_dates(us_regime, a_share_trade_dates)
    flag_cols = ["us_risk_on", "us_tech_strong", "us_smallcap_strong", "vix_stress"]
    for column in flag_cols:
        aligned[column] = aligned[column].fillna(False).astype(int)
    return aligned.loc[
        :,
        [
            "trade_date",
            "source_us_trade_date",
            "us_risk_on",
            "us_tech_strong",
            "us_smallcap_strong",
            "vix_stress",
            "spy_return_20d",
            "qqq_return_20d",
            "iwm_return_20d",
            "vix_return_20d",
            "qqq_spy_relative_20d",
            "iwm_spy_relative_20d",
        ],
    ].sort_values("trade_date").reset_index(drop=True)


def get_us_daily_prices(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    service = USMarketDailyService(cache_dir=cache_dir)
    request = USDailyRequest(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
    return service.get_daily(request)


def _build_close_wide(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.loc[:, ["trade_date", "ts_code", "close"]].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"])
    return (
        data.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
        .sort_index()
        .reset_index()
        .set_index("trade_date")
    )


def _align_to_a_share_dates(us_regime: pd.DataFrame, a_share_trade_dates: list[str]) -> pd.DataFrame:
    a_dates = pd.DataFrame({"trade_date": sorted({str(date) for date in a_share_trade_dates})})
    a_dates["_date"] = pd.to_datetime(a_dates["trade_date"], format="%Y%m%d")
    source = us_regime.copy()
    source["source_us_trade_date"] = source["us_trade_date"].astype(str)
    source["_date"] = pd.to_datetime(source["source_us_trade_date"], format="%Y%m%d")
    source = source.sort_values("_date")
    return (
        pd.merge_asof(
            a_dates.sort_values("_date"),
            source.drop(columns=["us_trade_date"]).sort_values("_date"),
            on="_date",
            direction="backward",
            allow_exact_matches=False,
        )
        .drop(columns=["_date"])
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _empty_regime_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "source_us_trade_date",
            "us_risk_on",
            "us_tech_strong",
            "us_smallcap_strong",
            "vix_stress",
            "spy_return_20d",
            "qqq_return_20d",
            "iwm_return_20d",
            "vix_return_20d",
            "qqq_spy_relative_20d",
            "iwm_spy_relative_20d",
        ]
    )
