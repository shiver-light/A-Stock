"""Financial indicator data aligned to trading days."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig

FUNDAMENTAL_FIELDS = ["ts_code", "ann_date", "end_date", "roe", "or_yoy"]
TRADE_CAL_FIELDS = ["exchange", "cal_date", "is_open", "pretrade_date"]


class AShareFundamentalService:
    """Minimal cached wrapper for fina_indicator and trade_cal."""

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_fundamental_history(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / "fina_indicator" / f"{ts_code}.parquet"
        cached = self._read_cache(cache_path, date_columns=("ann_date", "end_date"))
        need_fetch = refresh or cached.empty

        if need_fetch:
            fetched = self.client.fina_indicator(
                ts_code=ts_code,
                fields=",".join(FUNDAMENTAL_FIELDS),
            )
            history = self._normalize_fundamental_frame(fetched)
            self._write_cache(cache_path, history)
        else:
            history = cached

        # Use ann_date to filter the effective time window conservatively.
        result = history.loc[
            (history["ann_date"] >= start_date) & (history["ann_date"] <= end_date),
            FUNDAMENTAL_FIELDS,
        ].copy()
        return result.reset_index(drop=True)

    def get_trade_calendar(
        self,
        *,
        start_date: str,
        end_date: str,
        refresh: bool = False,
        exchange: str = "SSE",
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / "trade_cal" / f"{exchange}.parquet"
        cached = self._read_cache(cache_path, date_columns=("cal_date", "pretrade_date"))

        if refresh or cached.empty:
            fetched = self.client.trade_cal(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(TRADE_CAL_FIELDS),
            )
            calendar = self._normalize_trade_calendar(fetched)
            self._write_cache(cache_path, calendar)
        else:
            calendar = cached
            min_cached = str(calendar["cal_date"].min())
            max_cached = str(calendar["cal_date"].max())
            if start_date < min_cached or end_date > max_cached:
                fetched = self.client.trade_cal(
                    exchange=exchange,
                    start_date=min(start_date, min_cached),
                    end_date=max(end_date, max_cached),
                    fields=",".join(TRADE_CAL_FIELDS),
                )
                calendar = self._normalize_trade_calendar(fetched)
                self._write_cache(cache_path, calendar)

        mask = (calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)
        return calendar.loc[mask].reset_index(drop=True)

    def align_fundamental_to_daily(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        history = self.get_fundamental_history(
            ts_code=ts_code,
            start_date=self._shift_calendar_date(start_date, -500),
            end_date=end_date,
            refresh=refresh,
        )
        calendar = self.get_trade_calendar(
            start_date=self._shift_calendar_date(start_date, -30),
            end_date=self._shift_calendar_date(end_date, 30),
            refresh=refresh,
        )
        trading_days = calendar.loc[calendar["is_open"] == "1", ["cal_date"]].copy()
        trading_days = trading_days.rename(columns={"cal_date": "trade_date"})
        trading_days["ts_code"] = ts_code

        if history.empty:
            trading_days["roe"] = pd.NA
            trading_days["revenue_growth"] = pd.NA
            return trading_days.loc[
                (trading_days["trade_date"] >= start_date) & (trading_days["trade_date"] <= end_date),
                ["trade_date", "ts_code", "roe", "revenue_growth"],
            ].reset_index(drop=True)

        effective = history.copy()
        effective["effective_date"] = effective["ann_date"].apply(
            lambda ann_date: self._next_trade_date(calendar, ann_date)
        )
        effective = (
            effective.sort_values(["effective_date", "ann_date", "end_date"])
            .drop_duplicates(subset=["effective_date"], keep="last")
            .rename(columns={"or_yoy": "revenue_growth"})
        )
        trading_days["trade_date_key"] = pd.to_datetime(trading_days["trade_date"], format="%Y%m%d")
        effective["effective_date_key"] = pd.to_datetime(effective["effective_date"], format="%Y%m%d")

        aligned = pd.merge_asof(
            trading_days.sort_values("trade_date_key"),
            effective[["effective_date", "effective_date_key", "roe", "revenue_growth"]].sort_values("effective_date_key"),
            left_on="trade_date_key",
            right_on="effective_date_key",
            direction="backward",
        )
        result = aligned.drop(columns=["effective_date", "trade_date_key", "effective_date_key"])
        mask = (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
        return result.loc[mask, ["trade_date", "ts_code", "roe", "revenue_growth"]].reset_index(drop=True)

    def _normalize_fundamental_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_FIELDS)
        normalized = data.copy()
        normalized["ann_date"] = normalized["ann_date"].astype(str)
        normalized["end_date"] = normalized["end_date"].astype(str)
        return (
            normalized[FUNDAMENTAL_FIELDS]
            .sort_values(["ann_date", "end_date"])
            .drop_duplicates(subset=["ts_code", "ann_date", "end_date"], keep="last")
            .reset_index(drop=True)
        )

    def _normalize_trade_calendar(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=TRADE_CAL_FIELDS)
        normalized = data.copy()
        normalized["cal_date"] = normalized["cal_date"].astype(str)
        normalized["pretrade_date"] = normalized["pretrade_date"].fillna("").astype(str)
        normalized["is_open"] = normalized["is_open"].astype(str)
        return (
            normalized[TRADE_CAL_FIELDS]
            .sort_values("cal_date")
            .drop_duplicates(subset=["exchange", "cal_date"], keep="last")
            .reset_index(drop=True)
        )

    def _read_cache(self, path: Path, date_columns: tuple[str, ...]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        data = pd.read_parquet(path)
        for column in date_columns:
            if column in data.columns:
                data[column] = data[column].fillna("").astype(str)
        return data

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)

    def _next_trade_date(self, calendar: pd.DataFrame, ann_date: str) -> str:
        open_dates = calendar.loc[calendar["is_open"] == "1", "cal_date"]
        later = open_dates.loc[open_dates > ann_date]
        if later.empty:
            raise ValueError(f"No next trade date found after ann_date={ann_date}")
        return str(later.iloc[0])

    def _shift_calendar_date(self, date_text: str, days: int) -> str:
        return (pd.Timestamp(date_text) + pd.Timedelta(days=days)).strftime("%Y%m%d")


def get_a_share_fundamental_daily(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    service = AShareFundamentalService(cache_dir=cache_dir)
    return service.align_fundamental_to_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
