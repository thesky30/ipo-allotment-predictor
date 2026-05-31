"""Tushare-backed market and SW-industry refresh helpers.

This module is intentionally side-effect free except for the explicit writer.
Tests pass fake ``pro`` clients, so no network/API token is needed in pytest.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from reference_data import SW_LEVEL1_INDUSTRY_NAME_BY_CODE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"

START_DATE = "20190101"
SSE_COMPOSITE = "000001.SH"
SZSE_COMPOSITE = "399106.SZ"
CSI300 = "000300.SH"
SW_DAILY_COLUMNS = [
    "trade_date", "sw_level1_industry_code", "sw_level1_industry_name",
    "tushare_index_code", "turnover_100m_yuan", "return_pct", "pe", "pb",
]


def _date_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def _index_daily(pro, ts_code: str, start_date: str, end_date: str, fields: str) -> pd.DataFrame:
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    if df is None or df.empty:
        raise RuntimeError(f"Tushare index_daily returned empty data for {ts_code}")
    return df.copy()


def fetch_market_daily(pro, start_date: str = START_DATE, end_date: str | None = None) -> pd.DataFrame:
    end_date = end_date or pd.Timestamp.today().strftime("%Y%m%d")
    sse = _index_daily(pro, SSE_COMPOSITE, start_date, end_date, "trade_date,amount")
    szse = _index_daily(pro, SZSE_COMPOSITE, start_date, end_date, "trade_date,amount")
    csi = _index_daily(pro, CSI300, start_date, end_date, "trade_date,close,pct_chg")

    sse["sse_amount_100m"] = pd.to_numeric(sse["amount"], errors="coerce") / 1e5
    szse["szse_amount_100m"] = pd.to_numeric(szse["amount"], errors="coerce") / 1e5
    csi = csi.rename(columns={"close": "csi300_close", "pct_chg": "csi300_pct_chg"})

    out = (
        sse[["trade_date", "sse_amount_100m"]]
        .merge(szse[["trade_date", "szse_amount_100m"]], on="trade_date", how="outer")
        .merge(csi[["trade_date", "csi300_close", "csi300_pct_chg"]], on="trade_date", how="outer")
    )
    out["total_turnover_100m"] = out["sse_amount_100m"].fillna(0) + out["szse_amount_100m"].fillna(0)
    out["trade_date"] = _date_col(out["trade_date"])
    return out[[
        "trade_date", "sse_amount_100m", "szse_amount_100m",
        "total_turnover_100m", "csi300_close", "csi300_pct_chg",
    ]].sort_values("trade_date").reset_index(drop=True)


def _norm_name(value: object) -> str:
    return str(value).replace("(申万)", "").strip()


def fetch_sw_level1_mapping(pro) -> pd.DataFrame:
    raw = pro.index_classify(src="SW2021", level="L1", fields="index_code,industry_name")
    if raw is None or raw.empty:
        raise RuntimeError("Tushare index_classify returned empty SW level-1 mapping")
    raw = raw.copy()
    raw["sw_level1_industry_name"] = raw["industry_name"].map(_norm_name)
    name_to_tushare = dict(zip(raw["sw_level1_industry_name"], raw["index_code"]))

    rows = []
    for wind_code, name in SW_LEVEL1_INDUSTRY_NAME_BY_CODE.items():
        rows.append({
            "sw_level1_industry_code": wind_code,
            "sw_level1_industry_name": name,
            "tushare_index_code": name_to_tushare.get(name),
        })
    return pd.DataFrame(rows).dropna(subset=["tushare_index_code"]).reset_index(drop=True)


def fetch_sw_level1_daily(
    pro,
    mapping: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    end_date = end_date or pd.Timestamp.today().strftime("%Y%m%d")
    frames = []
    for row in mapping.itertuples(index=False):
        df = pro.sw_daily(
            ts_code=row.tushare_index_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,amount,pct_change,pe,pb",
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        df["sw_level1_industry_code"] = row.sw_level1_industry_code
        df["sw_level1_industry_name"] = row.sw_level1_industry_name
        df["tushare_index_code"] = row.tushare_index_code
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=SW_DAILY_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = _date_col(out["trade_date"])
    out["turnover_100m_yuan"] = pd.to_numeric(out["amount"], errors="coerce") / 10000
    out["return_pct"] = pd.to_numeric(out["pct_change"], errors="coerce")
    return out[[
        "trade_date", "sw_level1_industry_code", "sw_level1_industry_name",
        "tushare_index_code", "turnover_100m_yuan", "return_pct", "pe", "pb",
    ]].sort_values(["trade_date", "sw_level1_industry_code"]).reset_index(drop=True)


def read_cached_sw_daily(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / "sw_level1_market_daily_tushare.csv"
    if not path.exists():
        return pd.DataFrame(columns=SW_DAILY_COLUMNS)
    df = pd.read_csv(path)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def choose_sw_daily_output(
    fresh: pd.DataFrame,
    cached: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Use cached SW daily data when the fresh refresh is empty.

    Tushare SW daily endpoints can be quota-limited. Keeping the previous
    non-empty cache avoids replacing useful data with an empty file.
    """
    if fresh is not None and not fresh.empty:
        return fresh, False
    if cached is not None and not cached.empty:
        return cached, True
    return pd.DataFrame(columns=SW_DAILY_COLUMNS), False


def latest_sw_industry_pe(
    sw_daily: pd.DataFrame,
    sw_level1_industry_code: str,
    trade_date: str | pd.Timestamp,
) -> dict[str, object] | None:
    if sw_daily is None or sw_daily.empty:
        return None
    df = sw_daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    dt = pd.to_datetime(str(trade_date), format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(dt):
        return None
    pe = pd.to_numeric(df.get("pe"), errors="coerce")
    mask = (
        (df["sw_level1_industry_code"].astype(str) == str(sw_level1_industry_code))
        & (df["trade_date"] <= dt)
        & pe.notna()
        & (pe > 0)
    )
    matched = df.loc[mask].sort_values("trade_date")
    if matched.empty:
        return None
    row = matched.iloc[-1]
    return {
        "sw_level1_industry_code": str(row["sw_level1_industry_code"]),
        "sw_level1_industry_name": row.get("sw_level1_industry_name"),
        "trade_date": row["trade_date"].strftime("%Y%m%d"),
        "pe": float(row["pe"]),
    }


def fetch_latest_sw_industry_pe(
    pro,
    sw_level1_industry_code: str,
    trade_date: str | pd.Timestamp,
    *,
    lookback_days: int = 90,
) -> dict[str, object] | None:
    """Fetch latest available SW level-1 industry PE directly from Tushare."""
    dt = pd.to_datetime(str(trade_date), format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(dt):
        return None
    mapping = fetch_sw_level1_mapping(pro)
    row = mapping.loc[mapping["sw_level1_industry_code"].astype(str) == str(sw_level1_industry_code)]
    if row.empty:
        raise ValueError(f"unknown SW level-1 industry code: {sw_level1_industry_code}")
    start_date = (dt - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = dt.strftime("%Y%m%d")
    daily = fetch_sw_level1_daily(pro, row, start_date=start_date, end_date=end_date)
    return latest_sw_industry_pe(daily, sw_level1_industry_code, end_date)


def write_outputs(
    market_daily: pd.DataFrame,
    sw_mapping: pd.DataFrame,
    sw_daily: pd.DataFrame,
    data_dir: Path = DATA_DIR,
    db_path: Path = DB_PATH,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    market_daily.to_csv(data_dir / "market_daily.csv", index=False, encoding="utf-8-sig")
    sw_mapping.to_csv(data_dir / "sw_level1_mapping.csv", index=False, encoding="utf-8-sig")
    sw_daily.to_csv(data_dir / "sw_level1_market_daily_tushare.csv", index=False, encoding="utf-8-sig")
    with sqlite3.connect(db_path) as conn:
        market_daily.to_sql("market_daily", conn, if_exists="replace", index=False)
        sw_mapping.to_sql("sw_level1_mapping", conn, if_exists="replace", index=False)
        sw_daily.to_sql("sw_level1_market_daily_tushare", conn, if_exists="replace", index=False)


def _pro_api(token: str | None = None):
    import tushare as ts

    token = token or os.environ.get("TUSHARE_TOKEN")
    if token:
        ts.set_token(token)
    return ts.pro_api(token) if token else ts.pro_api()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh market and SW industry data via Tushare")
    parser.add_argument("--token", default=None)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    from config import load_env

    load_env()
    pro = _pro_api(args.token)
    market = fetch_market_daily(pro, args.start_date, args.end_date)
    mapping = fetch_sw_level1_mapping(pro)
    try:
        sw_daily = fetch_sw_level1_daily(pro, mapping, args.start_date, args.end_date)
    except Exception as e:  # noqa: BLE001 - keep market/mapping refresh usable under low Tushare quotas
        print(f"WARNING: sw_daily refresh skipped: {e}", file=sys.stderr)
        sw_daily = pd.DataFrame(columns=[
            "trade_date", "sw_level1_industry_code", "sw_level1_industry_name",
            "tushare_index_code", "turnover_100m_yuan", "return_pct", "pe", "pb",
        ])
    sw_daily, used_cache = choose_sw_daily_output(sw_daily, read_cached_sw_daily())
    write_outputs(market, mapping, sw_daily)
    print(f"market_daily rows: {len(market)}")
    print(f"sw_level1_mapping rows: {len(mapping)}")
    print(f"sw_level1_market_daily_tushare rows: {len(sw_daily)}")
    if used_cache:
        print("sw_level1_market_daily_tushare used existing cache because fresh refresh was empty", file=sys.stderr)


if __name__ == "__main__":
    main()
