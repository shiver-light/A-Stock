"""Unified stock universe provider."""

from __future__ import annotations

import pandas as pd

from data import INDEX_CODE_MAP, get_index_constituents, get_stock_basic_history


SUPPORTED_UNIVERSES = {
    "all_a",
    "hs300",
    "zz500",
    "zz1000",
    "zz2000",
    "sse50",
    "main_board",
    "chinext",
    "custom",
}


def get_custom_universe(
    ts_codes: list[str],
    *,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    unique_codes = sorted(set(ts_codes))
    result = pd.DataFrame({"ts_code": unique_codes})
    result["as_of_date"] = as_of_date
    result["universe_name"] = "custom"
    result["in_universe"] = True
    return result.loc[:, ["as_of_date", "ts_code", "universe_name", "in_universe"]]


def get_universe(
    universe_name: str,
    as_of_date: str,
    *,
    ts_codes: list[str] | None = None,
    include_weights: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    if universe_name not in SUPPORTED_UNIVERSES:
        raise ValueError(f"Unsupported universe: {universe_name}")

    if universe_name == "custom":
        if not ts_codes:
            raise ValueError("custom universe requires ts_codes.")
        return get_custom_universe(ts_codes, as_of_date=as_of_date)

    if universe_name in {"hs300", "zz500", "zz1000", "zz2000", "sse50"}:
        result = get_index_constituents(
            index_code=INDEX_CODE_MAP[universe_name],
            as_of_date=as_of_date,
            refresh=refresh,
        )
        result["universe_name"] = universe_name
        ordered_columns = ["as_of_date", "ts_code", "universe_name", "in_universe"]
        if include_weights and "weight" in result.columns:
            ordered_columns.insert(3, "weight")
        return result.loc[:, ordered_columns].reset_index(drop=True)

    stock_basic = get_stock_basic_history(refresh=refresh)
    filtered = _filter_active_a_share(stock_basic, as_of_date)

    if universe_name == "all_a":
        result = filtered.copy()
    elif universe_name == "main_board":
        result = filtered.loc[filtered["market"] == "主板"].copy()
    elif universe_name == "chinext":
        result = filtered.loc[filtered["market"] == "创业板"].copy()
    else:
        raise ValueError(f"Universe is not implemented: {universe_name}")

    result["as_of_date"] = as_of_date
    result["universe_name"] = universe_name
    result["in_universe"] = True
    return result.loc[:, ["as_of_date", "ts_code", "universe_name", "in_universe"]].reset_index(drop=True)


def get_universe_history(
    universe_name: str,
    as_of_dates: list[str],
    *,
    ts_codes: list[str] | None = None,
    include_weights: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    unique_dates = sorted(set(as_of_dates))
    frames: list[pd.DataFrame] = []
    for as_of_date in unique_dates:
        frame = get_universe(
            universe_name,
            as_of_date,
            ts_codes=ts_codes,
            include_weights=include_weights,
            refresh=refresh,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["as_of_date", "ts_code", "universe_name", "in_universe"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["as_of_date", "ts_code"])
        .drop_duplicates(subset=["as_of_date", "ts_code", "universe_name"], keep="last")
        .reset_index(drop=True)
    )


def _filter_active_a_share(stock_basic: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    data = stock_basic.copy()
    data["list_date"] = data["list_date"].fillna("").astype(str)
    data["delist_date"] = data["delist_date"].fillna("").astype(str)
    listed_mask = data["list_date"].ne("") & (data["list_date"] <= as_of_date)
    not_delisted_mask = data["delist_date"].eq("") | (data["delist_date"] > as_of_date)
    exchange_mask = data["exchange"].isin(["SSE", "SZSE"])
    return (
        data.loc[listed_mask & not_delisted_mask & exchange_mask, :]
        .sort_values("ts_code")
        .drop_duplicates(subset=["ts_code"], keep="last")
        .reset_index(drop=True)
    )
