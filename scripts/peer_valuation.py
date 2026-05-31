"""Peer-industry valuation helpers using Tushare daily_basic data.

This is a market-derived fallback for display/research. It does not overwrite
the prospectus-disclosed ``comparable_pe_avg_ex_nonrecurring`` model input.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import market_source


def active_members_on_date(members: pd.DataFrame, trade_date: str | pd.Timestamp) -> pd.DataFrame:
    out = members.copy()
    if "con_code" not in out.columns and "ts_code" in out.columns:
        out = out.rename(columns={"ts_code": "con_code"})
    date = pd.to_datetime(str(trade_date), format="%Y%m%d", errors="coerce")
    if pd.isna(date):
        date = pd.to_datetime(trade_date)
    in_date = pd.to_datetime(out.get("in_date"), format="%Y%m%d", errors="coerce")
    out_date = pd.to_datetime(out.get("out_date"), format="%Y%m%d", errors="coerce")
    mask = (in_date.isna() | (in_date <= date)) & (out_date.isna() | (out_date > date))
    return out.loc[mask].reset_index(drop=True)


def peer_pe_stats(
    daily_basic: pd.DataFrame,
    members: pd.DataFrame,
    *,
    exclude_ts_codes: set[str] | None = None,
    pe_col: str = "pe_ttm",
) -> dict[str, float | int | None]:
    exclude_ts_codes = exclude_ts_codes or set()
    member_codes = set(members["con_code"].dropna().astype(str)) - set(exclude_ts_codes)
    base = daily_basic[daily_basic["ts_code"].astype(str).isin(member_codes)].copy()
    values = pd.to_numeric(base[pe_col], errors="coerce")
    values = values[values > 0].dropna()
    prefix = "peer_pe_ttm" if pe_col == "pe_ttm" else f"peer_{pe_col}"
    return {
        "peer_count": int(values.count()),
        f"{prefix}_mean": float(values.mean()) if not values.empty else None,
        f"{prefix}_median": float(values.median()) if not values.empty else None,
    }


def fetch_daily_basic(pro, trade_date: str) -> pd.DataFrame:
    return pro.daily_basic(
        trade_date=trade_date,
        fields="ts_code,trade_date,pe,pe_ttm,pb,total_mv,circ_mv",
    )


def fetch_index_members(pro, index_code: str) -> pd.DataFrame:
    df = pro.index_member_all(l1_code=index_code)
    if df is None:
        return pd.DataFrame(columns=["con_code", "in_date", "out_date", "is_new"])
    df = df.copy()
    if "con_code" not in df.columns and "ts_code" in df.columns:
        df = df.rename(columns={"ts_code": "con_code"})
    return df


def estimate_industry_peer_pe(
    pro,
    sw_level1_industry_code: str,
    trade_date: str,
    *,
    exclude_ts_codes: set[str] | None = None,
) -> dict[str, object]:
    mapping = market_source.fetch_sw_level1_mapping(pro)
    row = mapping.loc[mapping["sw_level1_industry_code"].astype(str) == str(sw_level1_industry_code)]
    if row.empty:
        raise ValueError(f"unknown SW level-1 industry code: {sw_level1_industry_code}")
    info = row.iloc[0].to_dict()
    members = active_members_on_date(fetch_index_members(pro, info["tushare_index_code"]), trade_date)
    daily = fetch_daily_basic(pro, trade_date)
    stats = peer_pe_stats(daily, members, exclude_ts_codes=exclude_ts_codes, pe_col="pe_ttm")
    return {**info, "trade_date": trade_date, **stats}


def _pro_api(token: str | None = None):
    import tushare as ts

    token = token or os.environ.get("TUSHARE_TOKEN")
    if token:
        ts.set_token(token)
    return ts.pro_api(token) if token else ts.pro_api()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate same-industry peer PE from Tushare")
    parser.add_argument("--sw-level1-code", required=True)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--exclude", action="append", default=[], help="ts_code to exclude; repeatable")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    from config import load_env

    load_env()
    result = estimate_industry_peer_pe(
        _pro_api(args.token),
        args.sw_level1_code,
        args.trade_date,
        exclude_ts_codes=set(args.exclude),
    )
    print(pd.Series(result).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
