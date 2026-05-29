"""Clean Wind board and SW industry daily market exports.

Inputs:
  D:/wind导出数据/板块行情数据.xlsx
  D:/wind导出数据/申万一级行情数据.xlsx

Outputs:
  data/processed/board_market_daily.csv
  data/processed/sw_level1_market_daily.csv
  data/processed/ipo_offline.db tables:
    - board_market_daily
    - sw_level1_market_daily
  outputs/initial_analysis/board_industry_market_report.md

The source workbooks are not modified. Wind xlsx files may contain invalid
style metadata, so the loader strips xl/styles.xml in a temporary copy before
reading with pandas.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "initial_analysis"
DB_PATH = DATA_DIR / "ipo_offline.db"

BOARD_INPUT = Path("D:/wind导出数据/板块行情数据.xlsx")
INDUSTRY_INPUT = Path("D:/wind导出数据/申万一级行情数据.xlsx")

METRIC_MAP = {
    "成交额": "turnover_100m_yuan",
    "涨跌幅": "return_pct",
    "换手率": "turnover_rate_pct",
}

BOARD_NAME_MAP = {
    "万得主板": "主板",
    "创业板综": "创业板",
    "科创综指": "科创板",
}


def strip_xlsx_styles(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "xl/styles.xml":
                continue
            zout.writestr(item, zin.read(item.filename))


def read_wind_wide(path: Path) -> pd.DataFrame:
    with TemporaryDirectory(ignore_cleanup_errors=True) as td:
        safe = Path(td) / f"{path.stem}_nostyles.xlsx"
        strip_xlsx_styles(path, safe)
        return pd.read_excel(safe, sheet_name=0, header=None)


def excel_serial_to_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").floordiv(1)
    return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce").dt.date


def metric_from_header(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value)
    for key, clean in METRIC_MAP.items():
        if key in text:
            return clean
    return None


def wide_market_to_long(raw: pd.DataFrame, entity_col: str, entity_name_map: dict[str, str] | None = None) -> pd.DataFrame:
    headers = raw.iloc[0].ffill()
    entities = raw.iloc[1]
    records = []

    for col_idx in range(1, raw.shape[1]):
        metric = metric_from_header(headers.iloc[col_idx])
        entity = entities.iloc[col_idx]
        if metric is None or pd.isna(entity):
            continue
        entity = str(entity).strip()
        if entity_name_map:
            entity = entity_name_map.get(entity, entity)
        temp = pd.DataFrame({
            "trade_date": excel_serial_to_date(raw.iloc[2:, 0]),
            entity_col: entity,
            "metric": metric,
            "value": pd.to_numeric(raw.iloc[2:, col_idx], errors="coerce"),
        })
        records.append(temp)

    if not records:
        raise ValueError("No market columns parsed from Wind wide table.")

    long = pd.concat(records, ignore_index=True).dropna(subset=["trade_date", entity_col])
    out = (
        long
        .pivot_table(index=["trade_date", entity_col], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
        .sort_values(["trade_date", entity_col])
        .reset_index(drop=True)
    )

    for col in METRIC_MAP.values():
        if col not in out.columns:
            out[col] = pd.NA

    return out[["trade_date", entity_col, "turnover_100m_yuan", "return_pct", "turnover_rate_pct"]]


def coverage_frame(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    rows = []
    for entity, group in df.groupby(entity_col, dropna=False):
        rows.append({
            entity_col: entity,
            "rows": len(group),
            "date_min": group["trade_date"].min(),
            "date_max": group["trade_date"].max(),
            "turnover_non_null": int(group["turnover_100m_yuan"].notna().sum()),
            "return_non_null": int(group["return_pct"].notna().sum()),
            "turnover_rate_non_null": int(group["turnover_rate_pct"].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values(entity_col).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(board: pd.DataFrame, industry: pd.DataFrame, path: Path) -> None:
    board_cov = coverage_frame(board, "board")
    industry_cov = coverage_frame(industry, "sw_level1_industry_name")

    missing_boards = sorted(set(["主板", "创业板", "科创板", "北交所"]) - set(board["board"].dropna()))
    text = f"""# 板块与申万一级行业行情数据处理报告

## 板块行情

- 行数：{len(board)}
- 板块数：{board["board"].nunique()}
- 日期范围：{board["trade_date"].min()} 至 {board["trade_date"].max()}
- 当前缺失板块：{", ".join(missing_boards) if missing_boards else "无"}

{markdown_table(board_cov)}

## 申万一级行业行情

- 行数：{len(industry)}
- 行业数：{industry["sw_level1_industry_name"].nunique()}
- 日期范围：{industry["trade_date"].min()} 至 {industry["trade_date"].max()}

{markdown_table(industry_cov)}

## 口径提示

- `turnover_100m_yuan` 为成交额，单位亿元。
- `return_pct` 为日涨跌幅，单位 %。
- `turnover_rate_pct` 为日换手率，单位 %；当前板块行情文件没有换手率，因此板块表该列为空。
- 申万行业行情当前是行业名称，例如 `电子(申万)`；上市公司因子表目前是申万行业代码，后续需要补“代码-名称映射”后才能直接联接。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    board = wide_market_to_long(read_wind_wide(BOARD_INPUT), "board", BOARD_NAME_MAP)
    industry = wide_market_to_long(read_wind_wide(INDUSTRY_INPUT), "sw_level1_industry_name")

    board.to_csv(DATA_DIR / "board_market_daily.csv", index=False, encoding="utf-8-sig")
    industry.to_csv(DATA_DIR / "sw_level1_market_daily.csv", index=False, encoding="utf-8-sig")

    with sqlite3.connect(DB_PATH) as conn:
        board.to_sql("board_market_daily", conn, if_exists="replace", index=False)
        industry.to_sql("sw_level1_market_daily", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_board_market_daily ON board_market_daily(board, trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sw_l1_market_daily ON sw_level1_market_daily(sw_level1_industry_name, trade_date)")

    write_report(board, industry, OUT_DIR / "board_industry_market_report.md")

    print(f"board rows: {len(board)}, boards: {board['board'].nunique()}, range: {board['trade_date'].min()} to {board['trade_date'].max()}")
    print(f"industry rows: {len(industry)}, industries: {industry['sw_level1_industry_name'].nunique()}, range: {industry['trade_date'].min()} to {industry['trade_date'].max()}")
    print(f"wrote: {DATA_DIR / 'board_market_daily.csv'}")
    print(f"wrote: {DATA_DIR / 'sw_level1_market_daily.csv'}")


if __name__ == "__main__":
    main()
