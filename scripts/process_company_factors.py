"""Clean Wind company factor export and attach coverage diagnostics.

Input:
  D:/wind导出数据/上市公司因子数据.xlsx

Outputs:
  data/processed/company_factor_data.csv
  data/processed/ipo_company_factor_joined.csv
  data/processed/ipo_offline.db tables:
    - company_factor_data
    - ipo_company_factor_joined
  outputs/initial_analysis/company_factor_data_coverage.csv
  outputs/initial_analysis/company_factor_data_report.md

The source workbook is not modified. Wind workbooks may contain invalid style
metadata, so this script strips xl/styles.xml in a temporary copy before
reading with pandas.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "initial_analysis"
DB_PATH = DATA_DIR / "ipo_offline.db"
DEFAULT_INPUT = Path("D:/wind导出数据/上市公司因子数据.xlsx")


COLUMN_MAP = {
    "证券代码": "security_code",
    "证券简称": "security_name_factor",
    "初步询价公告日": "inquiry_announcement_date_raw",
    "初步询价起始日": "inquiry_start_date_raw",
    "所属申万行业代码(2021)\n[交易日期] 最新收盘日\n[行业级别] 一级行业": "sw_level1_industry_code",
    "所属申万行业代码(2021)\n[交易日期] 最新收盘日\n[行业级别] 二级行业": "sw_level2_industry_code",
    "网下询价市值门槛\n[单位] 万元": "offline_market_value_threshold_10k_yuan",
    "首发预计募集资金\n[单位] 亿元": "expected_fundraising_100m_yuan",
    "发行价格上限\n[单位] 元": "offer_price_upper_yuan_factor",
    "发行价格下限(底价)\n[单位] 元": "offer_price_lower_yuan_factor",
    "首发主承销商": "lead_underwriter",
    "首发保荐机构": "sponsor",
    "首发市盈率(超额配售前)\n[单位] 倍": "ipo_pe_pre_overallotment",
    "首发时所属行业市盈率\n[单位] 倍": "industry_pe_at_ipo_factor",
    "可比上市公司PE均值(扣非后)": "comparable_pe_avg_ex_nonrecurring_factor",
    "近三年营收复合增长率\n[单位] %": "revenue_cagr_3y_pct",
    "近一年营收额\n[单位] 亿元": "latest_revenue_100m_yuan",
    "发行市净率\n[单位] 倍": "issue_pb_factor",
}


def strip_xlsx_styles(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "xl/styles.xml":
                continue
            zout.writestr(item, zin.read(item.filename))


def excel_serial_to_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce").dt.date


def normalize_security_code(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    out = out.str.replace(r"\.0$", "", regex=True)
    valid = out.str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)
    return out.where(valid)


def normalize_industry_code(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.round(0).astype("Int64").astype("string")
    return out.replace("<NA>", pd.NA)


def read_company_factor_file(path: Path) -> pd.DataFrame:
    with TemporaryDirectory(ignore_cleanup_errors=True) as td:
        safe = Path(td) / f"{path.stem}_nostyles.xlsx"
        strip_xlsx_styles(path, safe)
        raw = pd.read_excel(safe, sheet_name=0, dtype={"证券代码": str})

    missing_cols = sorted(set(COLUMN_MAP) - set(raw.columns))
    if missing_cols:
        raise ValueError(f"{path} missing columns: {missing_cols}")

    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df["security_code"] = normalize_security_code(df["security_code"])
    df = df.dropna(subset=["security_code"]).drop_duplicates(subset=["security_code"], keep="first")

    for raw_col, clean_col in [
        ("inquiry_announcement_date_raw", "inquiry_announcement_date"),
        ("inquiry_start_date_raw", "inquiry_start_date"),
    ]:
        df[clean_col] = excel_serial_to_date(df[raw_col])
        df = df.drop(columns=[raw_col])

    for col in ["sw_level1_industry_code", "sw_level2_industry_code"]:
        df[col] = normalize_industry_code(df[col])

    numeric_cols = [
        "offline_market_value_threshold_10k_yuan",
        "expected_fundraising_100m_yuan",
        "offer_price_upper_yuan_factor",
        "offer_price_lower_yuan_factor",
        "ipo_pe_pre_overallotment",
        "industry_pe_at_ipo_factor",
        "comparable_pe_avg_ex_nonrecurring_factor",
        "revenue_cagr_3y_pct",
        "latest_revenue_100m_yuan",
        "issue_pb_factor",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["security_name_factor", "lead_underwriter", "sponsor"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    ordered = [
        "security_code",
        "security_name_factor",
        "inquiry_announcement_date",
        "inquiry_start_date",
        "sw_level1_industry_code",
        "sw_level2_industry_code",
        "offline_market_value_threshold_10k_yuan",
        "expected_fundraising_100m_yuan",
        "offer_price_upper_yuan_factor",
        "offer_price_lower_yuan_factor",
        "lead_underwriter",
        "sponsor",
        "ipo_pe_pre_overallotment",
        "industry_pe_at_ipo_factor",
        "comparable_pe_avg_ex_nonrecurring_factor",
        "revenue_cagr_3y_pct",
        "latest_revenue_100m_yuan",
        "issue_pb_factor",
    ]
    return df[ordered].sort_values("security_code").reset_index(drop=True)


def build_coverage(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_codes = set(base["security_code"].dropna())
    matched = df["security_code"].isin(base_codes)
    rows.append({
        "metric": "rows_after_cleaning",
        "value": len(df),
        "note": "有效证券代码行数；已剔除 Wind 页脚/空行",
    })
    rows.append({
        "metric": "matched_to_ipo_sample",
        "value": int(matched.sum()),
        "note": "能与 ipo_offline_sample 按 security_code 匹配的行数",
    })
    rows.append({
        "metric": "ipo_sample_missing_company_factor",
        "value": int(base["security_code"].nunique() - matched.sum()),
        "note": "现有 IPO 样本中暂未在本因子表出现的代码数",
    })
    rows.append({
        "metric": "company_factor_only",
        "value": int((~matched).sum()),
        "note": "本因子表存在、但当前 IPO 样本库没有的代码数",
    })

    for col in df.columns:
        if col == "security_code":
            continue
        non_null = int(df[col].notna().sum())
        rows.append({
            "metric": f"non_null__{col}",
            "value": non_null,
            "note": f"覆盖率 {non_null / len(df):.1%}",
        })
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, joined: pd.DataFrame, coverage: pd.DataFrame, path: Path) -> None:
    def val(metric: str) -> int:
        return int(coverage.loc[coverage["metric"] == metric, "value"].iloc[0])

    key_fields = [
        "inquiry_announcement_date",
        "inquiry_start_date",
        "sw_level1_industry_code",
        "sw_level2_industry_code",
        "offline_market_value_threshold_10k_yuan",
        "expected_fundraising_100m_yuan",
        "lead_underwriter",
        "sponsor",
        "revenue_cagr_3y_pct",
        "latest_revenue_100m_yuan",
    ]
    field_rows = []
    for col in key_fields:
        n = int(df[col].notna().sum())
        field_rows.append(f"| `{col}` | {n} | {n / len(df):.1%} |")

    by_board = (
        joined.groupby("board", dropna=False)
        .agg(
            ipo_rows=("security_code", "count"),
            matched_factor=("has_company_factor", "sum"),
            inquiry_start_non_null=("inquiry_start_date", lambda x: int(x.notna().sum())),
            sw_l1_non_null=("sw_level1_industry_code", lambda x: int(x.notna().sum())),
            threshold_non_null=("offline_market_value_threshold_10k_yuan", lambda x: int(x.notna().sum())),
            underwriter_non_null=("lead_underwriter", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    by_board["matched_factor"] = by_board["matched_factor"].astype(int)
    board_headers = list(by_board.columns)
    board_lines = [
        "| " + " | ".join(board_headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(board_headers) - 1)) + " |",
    ]
    for _, row in by_board.iterrows():
        board_lines.append("| " + " | ".join(str(row[col]) for col in board_headers) + " |")
    board_md = "\n".join(board_lines)

    text = f"""# 上市公司因子数据处理报告

生成文件：`data/processed/company_factor_data.csv`

## 总览

- 清洗后有效行数：{val("rows_after_cleaning")}
- 与现有 IPO 样本库匹配：{val("matched_to_ipo_sample")}
- 现有 IPO 样本暂缺本因子表：{val("ipo_sample_missing_company_factor")}
- 因子表有、IPO 样本库没有：{val("company_factor_only")}

## 关键字段覆盖

| 字段 | 非空数 | 覆盖率 |
|---|---:|---:|
{chr(10).join(field_rows)}

## 分板块匹配情况

{board_md}

## 口径提示

- 本脚本只新增独立表 `company_factor_data` 和联表视图式产物 `ipo_company_factor_joined`，不覆盖 `ipo_offline_sample` 主表。
- `offer_price_upper_yuan_factor` / `offer_price_lower_yuan_factor` 在本次文件中全为空，暂不能形成价格区间因子。
- 当前文件给的是申万行业代码，不是行业名称；后续如要展示友好名称，需要再补申万行业代码-名称映射。
- 这些字段属于询价前因子候选，但仍建议在入模前逐项确认公告发布时间口径。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_company_factor_file(DEFAULT_INPUT)
    with sqlite3.connect(DB_PATH) as conn:
        base = pd.read_sql(
            "SELECT security_code, security_name, board, listing_date, inquiry_deadline_date, subscription_deadline_date "
            "FROM ipo_offline_sample WHERE security_code IS NOT NULL",
            conn,
        )
        joined = base.merge(df, on="security_code", how="left", validate="many_to_one")
        joined["has_company_factor"] = joined["security_name_factor"].notna()
        coverage = build_coverage(df, base)

        df.to_sql("company_factor_data", conn, if_exists="replace", index=False)
        joined.to_sql("ipo_company_factor_joined", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_company_factor_code ON company_factor_data(security_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_joined_company_factor_code ON ipo_company_factor_joined(security_code)")

    df.to_csv(DATA_DIR / "company_factor_data.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(DATA_DIR / "ipo_company_factor_joined.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT_DIR / "company_factor_data_coverage.csv", index=False, encoding="utf-8-sig")
    write_report(df, joined, coverage, OUT_DIR / "company_factor_data_report.md")

    print(f"company_factor_data rows: {len(df)}")
    print(f"matched rows: {int(joined['has_company_factor'].sum())} / {len(joined)}")
    print(f"wrote: {DATA_DIR / 'company_factor_data.csv'}")
    print(f"wrote: {OUT_DIR / 'company_factor_data_report.md'}")


if __name__ == "__main__":
    main()
