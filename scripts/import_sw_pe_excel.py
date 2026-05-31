"""Import a Wind-exported SW level-1 PE workbook into local cache files."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from market_source import DB_PATH, SW_DAILY_COLUMNS
from reference_data import SW_LEVEL1_INDUSTRY_CODE_BY_NAME

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_CSV = DATA_DIR / "sw_level1_market_daily_tushare.csv"


def _clean_sw_name(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("SW"):
        text = text[2:]
    return text.replace("(申万)", "").replace("（申万）", "").strip()


def _pe_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "市盈率" in str(c) or "PE" in str(c).upper()]
    if not candidates:
        raise ValueError("未找到市盈率/PE列")
    return candidates[0]


def build_sw_pe_frame(raw: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    pe_col = _pe_column(raw)
    name_col = raw.columns[0]
    df = raw[[name_col, pe_col]].copy()
    df["sw_level1_industry_name"] = df[name_col].map(_clean_sw_name)
    df["sw_level1_industry_code"] = df["sw_level1_industry_name"].map(SW_LEVEL1_INDUSTRY_CODE_BY_NAME)
    df["pe"] = pd.to_numeric(df[pe_col], errors="coerce")
    out = df.dropna(subset=["sw_level1_industry_code", "pe"])[
        ["sw_level1_industry_code", "sw_level1_industry_name", "pe"]
    ].copy()
    out["trade_date"] = pd.to_datetime(str(trade_date), format="%Y%m%d", errors="coerce")
    if out["trade_date"].isna().any():
        raise ValueError(f"无效交易日：{trade_date}")
    out["tushare_index_code"] = None
    out["turnover_100m_yuan"] = None
    out["return_pct"] = None
    out["pb"] = None
    return out[SW_DAILY_COLUMNS].sort_values(["trade_date", "sw_level1_industry_code"]).reset_index(drop=True)


def merge_sw_pe_frame(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.copy().reset_index(drop=True)
    old = existing.copy()
    old["trade_date"] = pd.to_datetime(old["trade_date"], errors="coerce")
    incoming = new.copy()
    incoming["trade_date"] = pd.to_datetime(incoming["trade_date"], errors="coerce")
    keys = set(zip(incoming["trade_date"], incoming["sw_level1_industry_code"].astype(str)))
    keep = ~old.apply(lambda r: (r["trade_date"], str(r["sw_level1_industry_code"])) in keys, axis=1)
    out = pd.concat([old.loc[keep], incoming], ignore_index=True)
    return out[SW_DAILY_COLUMNS].sort_values(["trade_date", "sw_level1_industry_code"]).reset_index(drop=True)


def import_sw_pe_excel(
    excel_path: Path,
    trade_date: str,
    *,
    csv_path: Path = DEFAULT_CSV,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    raw = pd.read_excel(excel_path)
    new = build_sw_pe_frame(raw, trade_date)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
    else:
        existing = pd.DataFrame(columns=SW_DAILY_COLUMNS)
    merged = merge_sw_pe_frame(existing, new)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with sqlite3.connect(db_path) as conn:
        merged.to_sql("sw_level1_market_daily_tushare", conn, if_exists="replace", index=False)
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Wind SW PE TTM Excel into local cache")
    parser.add_argument("excel_path", type=Path)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD, e.g. 20260529")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()

    new = import_sw_pe_excel(args.excel_path, args.trade_date, csv_path=args.csv_path, db_path=args.db_path)
    print(f"imported rows: {len(new)}")
    print(f"trade_date: {args.trade_date}")
    print(f"csv: {args.csv_path}")
    print(f"db: {args.db_path}")


if __name__ == "__main__":
    main()
