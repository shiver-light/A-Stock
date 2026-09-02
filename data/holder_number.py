"""Shareholder-number data aligned to trading days by announcement date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.tushare_client import TushareClient, TushareClientConfig

HOLDER_NUMBER_FIELDS = ["ts_code", "ann_date", "end_date", "holder_num"]


class AShareHolderNumberService:
    """Cached wrapper for Tushare stk_holdernumber aligned by ann_date availability."""

    def __init__(
        self,
        client: TushareClient | None = None,
        *,
        cache_dir: str | Path = "data/cache/tushare",
    ) -> None:
        self.client = client or TushareClient(TushareClientConfig())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_holder_number_history(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Load shareholder-number disclosures filtered by ann_date."""

        cache_path = self.cache_dir / "stk_holdernumber" / f"{ts_code}.parquet"
        metadata_path = cache_path.with_suffix(".json")
        cached = self._read_cache(cache_path)
        if refresh or not self._cache_covers_window(cached, metadata_path, start_date, end_date):
            fetched = self.client.stk_holdernumber(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(HOLDER_NUMBER_FIELDS),
            )
            history = self._normalize_holder_number(fetched)
            self._write_cache(cache_path, history)
            self._write_cache_metadata(metadata_path, start_date=start_date, end_date=end_date)
        else:
            history = cached

        result = history.loc[(history["ann_date"] >= start_date) & (history["ann_date"] <= end_date)].copy()
        return result.reset_index(drop=True)

    def align_holder_number_to_daily(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Align latest disclosed holder-number change ratio to each trading day.

        The factor becomes available from ann_date onward, so end_date is never used
        as the signal availability date.
        """

        history = self.get_holder_number_history(
            ts_code=ts_code,
            start_date=self._shift_calendar_date(start_date, -500),
            end_date=end_date,
            refresh=refresh,
        )
        calendar_start = self._holder_calendar_start(history, start_date)
        calendar = self.get_trade_calendar(
            start_date=calendar_start,
            end_date=end_date,
            refresh=refresh,
        )
        trading_days = calendar.loc[calendar["is_open"] == "1", ["cal_date"]].rename(columns={"cal_date": "trade_date"})
        trading_days["ts_code"] = ts_code
        if history.empty:
            return self._empty_aligned_frame(trading_days, start_date, end_date)

        effective = history.sort_values(["ann_date", "end_date"]).copy()
        effective["previous_holder_num"] = effective["holder_num"].shift(1)
        effective["holder_num_change"] = effective["holder_num"] - effective["previous_holder_num"]
        effective["holder_num_change_ratio"] = effective["holder_num_change"] / effective["previous_holder_num"]
        effective["holder_num_change_ratio_negative"] = -effective["holder_num_change_ratio"]
        effective["holder_report_lag_trading_days"] = self._report_lag_trading_days(
            effective,
            calendar,
        )
        effective = effective.dropna(subset=["holder_num_change_ratio_negative"]).copy()
        if effective.empty:
            return self._empty_aligned_frame(trading_days, start_date, end_date)

        trading_days["trade_date_key"] = pd.to_datetime(trading_days["trade_date"], format="%Y%m%d")
        effective["ann_date_key"] = pd.to_datetime(effective["ann_date"], format="%Y%m%d")
        aligned = pd.merge_asof(
            trading_days.sort_values("trade_date_key"),
            effective[
                [
                    "ann_date",
                    "end_date",
                    "ann_date_key",
                    "holder_num",
                    "previous_holder_num",
                    "holder_num_change",
                    "holder_num_change_ratio",
                    "holder_num_change_ratio_negative",
                    "holder_report_lag_trading_days",
                ]
            ].sort_values("ann_date_key"),
            left_on="trade_date_key",
            right_on="ann_date_key",
            direction="backward",
        )
        aligned = aligned.drop(columns=["trade_date_key", "ann_date_key"])
        mask = (aligned["trade_date"] >= start_date) & (aligned["trade_date"] <= end_date)
        return aligned.loc[
            mask,
            [
                "trade_date",
                "ts_code",
                "ann_date",
                "end_date",
                "holder_num",
                "previous_holder_num",
                "holder_num_change",
                "holder_num_change_ratio",
                "holder_num_change_ratio_negative",
                "holder_report_lag_trading_days",
            ],
        ].reset_index(drop=True)

    def get_trade_calendar(
        self,
        *,
        start_date: str,
        end_date: str,
        refresh: bool = False,
        exchange: str = "SSE",
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / "trade_cal" / f"{exchange}.parquet"
        cached = self._read_calendar_cache(cache_path)
        if refresh or cached.empty:
            fetched = self.client.trade_cal(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                fields="exchange,cal_date,is_open,pretrade_date",
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
                    fields="exchange,cal_date,is_open,pretrade_date",
                )
                calendar = self._normalize_trade_calendar(fetched)
                self._write_cache(cache_path, calendar)
        return calendar.loc[(calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)].reset_index(
            drop=True
        )

    def _empty_aligned_frame(self, trading_days: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        result = trading_days.copy()
        for column in [
            "ann_date",
            "end_date",
            "holder_num",
            "previous_holder_num",
            "holder_num_change",
            "holder_num_change_ratio",
            "holder_num_change_ratio_negative",
            "holder_report_lag_trading_days",
        ]:
            result[column] = pd.NA
        mask = (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
        return result.loc[mask].reset_index(drop=True)

    def _normalize_holder_number(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=HOLDER_NUMBER_FIELDS)
        normalized = data.copy()
        for column in ["ts_code", "ann_date", "end_date"]:
            normalized[column] = normalized[column].fillna("").astype(str)
        normalized["holder_num"] = pd.to_numeric(normalized["holder_num"], errors="coerce")
        normalized = normalized.dropna(subset=["ts_code", "ann_date", "end_date", "holder_num"])
        return (
            normalized.loc[:, HOLDER_NUMBER_FIELDS]
            .sort_values(["ann_date", "end_date"])
            .drop_duplicates(subset=["ts_code", "ann_date", "end_date"], keep="last")
            .reset_index(drop=True)
        )

    def _normalize_trade_calendar(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = ["exchange", "cal_date", "is_open", "pretrade_date"]
        if data is None or data.empty:
            return pd.DataFrame(columns=columns)
        normalized = data.copy()
        normalized["cal_date"] = normalized["cal_date"].fillna("").astype(str)
        normalized["is_open"] = normalized["is_open"].astype(str)
        normalized["pretrade_date"] = normalized["pretrade_date"].fillna("").astype(str)
        return (
            normalized.loc[:, columns]
            .sort_values("cal_date")
            .drop_duplicates(subset=["exchange", "cal_date"], keep="last")
            .reset_index(drop=True)
        )

    def _holder_calendar_start(self, history: pd.DataFrame, start_date: str) -> str:
        default_start = self._shift_calendar_date(start_date, -30)
        if history.empty or "end_date" not in history.columns:
            return default_start
        end_dates = history["end_date"].fillna("").astype(str)
        end_dates = end_dates.loc[end_dates.ne("")]
        if end_dates.empty:
            return default_start
        return min(default_start, str(end_dates.min()))

    def _report_lag_trading_days(self, disclosures: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
        if disclosures.empty or calendar.empty:
            return pd.Series(pd.NA, index=disclosures.index, dtype="Int64")
        open_dates = (
            calendar.loc[calendar["is_open"] == "1", "cal_date"]
            .dropna()
            .astype(str)
            .sort_values()
            .drop_duplicates()
            .tolist()
        )
        rows = []
        for row in disclosures.itertuples():
            end_date = str(row.end_date)
            ann_date = str(row.ann_date)
            lag = sum(1 for trade_date in open_dates if end_date < trade_date <= ann_date)
            rows.append(lag)
        return pd.Series(rows, index=disclosures.index, dtype="Int64")

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        data = pd.read_parquet(path)
        for column in ["ts_code", "ann_date", "end_date"]:
            if column in data.columns:
                data[column] = data[column].fillna("").astype(str)
        return data

    def _cache_covers_window(self, cached: pd.DataFrame, metadata_path: Path, start_date: str, end_date: str) -> bool:
        if cached.empty or "ann_date" not in cached.columns:
            return False
        metadata = self._read_cache_metadata(metadata_path)
        if metadata:
            return str(metadata.get("start_date", "")) <= start_date and str(metadata.get("end_date", "")) >= end_date
        return str(cached["ann_date"].min()) <= start_date and str(cached["ann_date"].max()) >= end_date

    def _read_cache_metadata(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            return pd.read_json(path, typ="series").fillna("").astype(str).to_dict()
        except ValueError:
            return {}

    def _write_cache_metadata(self, path: Path, *, start_date: str, end_date: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.Series({"start_date": start_date, "end_date": end_date}).to_json(path, force_ascii=False)

    def _read_calendar_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        data = pd.read_parquet(path)
        for column in ["cal_date", "pretrade_date"]:
            if column in data.columns:
                data[column] = data[column].fillna("").astype(str)
        return data

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)

    def _shift_calendar_date(self, date_text: str, days: int) -> str:
        return (pd.Timestamp(date_text) + pd.Timedelta(days=days)).strftime("%Y%m%d")


def get_a_share_holder_number_daily(
    *,
    ts_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache/tushare",
) -> pd.DataFrame:
    service = AShareHolderNumberService(cache_dir=cache_dir)
    return service.align_holder_number_to_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        refresh=refresh,
    )
