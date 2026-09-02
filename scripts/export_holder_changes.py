"""Export recent shareholder-number changes for an A-share universe.

Tushare interface: stk_holdernumber
Doc: https://tushare.pro/document/2?doc_id=166
Fields: ts_code, ann_date, end_date, holder_num
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import TushareClient, TushareClientConfig, get_stock_basic_history
from universe import get_universe


HOLDER_FIELDS = ["ts_code", "ann_date", "end_date", "holder_num"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export recent shareholder-number changes.")
    parser.add_argument("--universe-name", default="all_a_ex_chinext_star_st", help="Universe name to export.")
    parser.add_argument("--as-of-date", default=None, help="Universe snapshot date in YYYYMMDD. Defaults to today.")
    parser.add_argument("--start-date", default=None, help="Announcement start date in YYYYMMDD. Defaults to 365 days before end date.")
    parser.add_argument("--end-date", default=None, help="Announcement end date in YYYYMMDD. Defaults to today.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of latest-updated rows to print and export.")
    parser.add_argument("--output-dir", default="output/shareholder_changes", help="Output directory.")
    parser.add_argument("--cache-dir", default="data/cache/tushare/stk_holdernumber", help="Per-stock cache directory.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached holder-number data.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    end_date = args.end_date or datetime.now().strftime("%Y%m%d")
    start_date = args.start_date or (pd.Timestamp(end_date) - pd.DateOffset(years=1)).strftime("%Y%m%d")
    as_of_date = args.as_of_date or end_date

    universe = get_universe(args.universe_name, as_of_date=as_of_date)
    ts_codes = universe["ts_code"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    holder_data = load_holdernumber_data(
        ts_codes,
        start_date=start_date,
        end_date=end_date,
        cache_dir=Path(args.cache_dir),
        refresh=args.refresh,
    )
    changes = summarize_latest_holder_changes(holder_data, stock_basic=get_stock_basic_history(refresh=False))
    latest = changes.sort_values(["ann_date", "end_date", "ts_code"], ascending=[False, False, True]).head(args.top_n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{args.universe_name}_{start_date}_{end_date}_holder_changes.csv"
    top_path = output_dir / f"{args.universe_name}_{start_date}_{end_date}_latest_top{args.top_n}.csv"
    changes.to_csv(full_path, index=False)
    latest.to_csv(top_path, index=False)

    print(render_holder_changes(latest, universe_name=args.universe_name, start_date=start_date, end_date=end_date))
    print(f"\nfull_csv={full_path}")
    print(f"top_csv={top_path}")
    return 0


def load_holdernumber_data(
    ts_codes: list[str],
    *,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load stk_holdernumber data per stock with local parquet cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = TushareClient(TushareClientConfig())
    frames: list[pd.DataFrame] = []
    for index, ts_code in enumerate(ts_codes, start=1):
        cache_path = cache_dir / f"{ts_code}.parquet"
        if not refresh and cache_path.exists():
            data = pd.read_parquet(cache_path)
        else:
            data = client.stk_holdernumber(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=",".join(HOLDER_FIELDS),
            )
            data = normalize_holdernumber_data(data)
            data.to_parquet(cache_path, index=False)
        frames.append(data)
        if index % 200 == 0:
            print(f"loaded {index}/{len(ts_codes)} stocks")
    if not frames:
        return pd.DataFrame(columns=HOLDER_FIELDS)
    return normalize_holdernumber_data(pd.concat(frames, ignore_index=True))


def summarize_latest_holder_changes(holder_data: pd.DataFrame, *, stock_basic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calculate latest holder-number change versus the previous disclosure."""

    data = normalize_holdernumber_data(holder_data)
    columns = [
        "ts_code",
        "name",
        "industry",
        "market",
        "ann_date",
        "end_date",
        "holder_num",
        "previous_ann_date",
        "previous_end_date",
        "previous_holder_num",
        "holder_num_change",
        "holder_num_change_ratio",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    data = data.sort_values(["ts_code", "ann_date", "end_date"], ascending=[True, False, False])
    data["previous_ann_date"] = data.groupby("ts_code")["ann_date"].shift(-1)
    data["previous_end_date"] = data.groupby("ts_code")["end_date"].shift(-1)
    data["previous_holder_num"] = data.groupby("ts_code")["holder_num"].shift(-1)
    data["holder_num_change"] = data["holder_num"] - data["previous_holder_num"]
    data["holder_num_change_ratio"] = data["holder_num_change"] / data["previous_holder_num"]
    latest = data.groupby("ts_code", as_index=False).head(1).copy()
    latest = latest.dropna(subset=["previous_holder_num"]).copy()

    if stock_basic is not None and not stock_basic.empty:
        basic_cols = [column for column in ["ts_code", "name", "industry", "market"] if column in stock_basic.columns]
        if "ts_code" in basic_cols:
            basic = stock_basic.loc[:, basic_cols].drop_duplicates(subset=["ts_code"], keep="last").copy()
            basic["ts_code"] = basic["ts_code"].astype(str)
            latest = latest.merge(basic, on="ts_code", how="left")
    for column in ["name", "industry", "market"]:
        if column not in latest.columns:
            latest[column] = ""
    return latest.loc[:, columns].sort_values(["ann_date", "end_date", "ts_code"], ascending=[False, False, True]).reset_index(drop=True)


def normalize_holdernumber_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize holder-number rows and remove duplicate disclosure keys."""

    if data is None or data.empty:
        return pd.DataFrame(columns=HOLDER_FIELDS)
    result = data.copy()
    for column in ["ts_code", "ann_date", "end_date"]:
        result[column] = result[column].fillna("").astype(str)
    result["holder_num"] = pd.to_numeric(result["holder_num"], errors="coerce")
    result = result.dropna(subset=["ts_code", "ann_date", "end_date", "holder_num"])
    return (
        result.loc[:, HOLDER_FIELDS]
        .sort_values(["ts_code", "ann_date", "end_date"])
        .drop_duplicates(subset=["ts_code", "ann_date", "end_date"], keep="last")
        .reset_index(drop=True)
    )


def render_holder_changes(data: pd.DataFrame, *, universe_name: str, start_date: str, end_date: str) -> str:
    """Render a compact text table for latest shareholder-number changes."""

    lines = [
        "【最近一年股东人数变化】",
        f"股票池：{universe_name}",
        f"公告日期区间：{start_date} - {end_date}",
        "",
        "rank ts_code name ann_date end_date holder_num change change_ratio",
    ]
    for rank, row in enumerate(data.itertuples(index=False), start=1):
        ratio = "" if pd.isna(row.holder_num_change_ratio) else f"{row.holder_num_change_ratio * 100:.2f}%"
        change = "" if pd.isna(row.holder_num_change) else f"{row.holder_num_change:,.0f}"
        holder_num = "" if pd.isna(row.holder_num) else f"{row.holder_num:,.0f}"
        lines.append(
            f"{rank} {row.ts_code} {row.name or ''} {row.ann_date} {row.end_date} "
            f"{holder_num} {change} {ratio}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
