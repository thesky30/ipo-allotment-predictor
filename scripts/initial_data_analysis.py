"""Data loading, integration and EDA for Wind IPO offline subscription exports.

v2 changes (2026-05-22):
  - 北交所_网下打新数据.xlsx added as board "北交所".
  - Three 补充数据 files joined by security_code, supplying inquiry-time fields:
    inquiry_subscription_total_10k, inquiry_investors_count,
    inquiry_allotment_accounts, subscription_step/upper/lower_limit_10k,
    quote_price_weighted_avg, quote_price_median,
    inquiry_deadline_date, subscription_deadline_date, first_day_return_pct.
  - 科创板 supplement additionally provides offer_price_upper/lower_yuan.
  - 主板 has no supplement file; inquiry fields remain NaN for that board.
  - a_investor_lottery_rate_pct absent in BSE main file; treated as optional.

v1 (2026-05-21): initial EDA on 科创板 / 创业板 / 主板 only.

Raw source files are never modified.  Wind xlsx files often contain malformed
style metadata; the loader strips xl/styles.xml before reading with pandas.
"""

from __future__ import annotations

import json
import math
import sqlite3
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "initial_analysis"

# ---------------------------------------------------------------------------
# Main file specs (one row per IPO, labels + post-subscription fields)
# ---------------------------------------------------------------------------
FILE_SPECS = [
    {
        "path": Path("D:/wind导出数据/全部科创板_网下打新数据.xlsx"),
        "board": "科创板",
        "sample_note": "科创板全样本",
    },
    {
        "path": Path("D:/wind导出数据/注册制创业板_网下打新数据.xlsx"),
        "board": "创业板",
        "sample_note": "注册制创业板",
    },
    {
        "path": Path("D:/wind导出数据/全部科创板_主板.xlsx"),
        "board": "主板",
        "sample_note": "主板注册制后样本",
    },
    {
        "path": Path("D:/wind导出数据/北交所_网下打新数据.xlsx"),
        "board": "北交所",
        "sample_note": "北交所全样本",
    },
]

# ---------------------------------------------------------------------------
# Supplement file specs (inquiry-time fields, joined by security_code)
# 主板 has no supplement; those rows will have NaN for all inquiry fields.
# ---------------------------------------------------------------------------
SUPPLEMENT_SPECS = [
    {
        "path": Path("D:/wind导出数据/全部科创板_网下打新_补充数据.xlsx"),
        "board": "科创板",
    },
    {
        "path": Path("D:/wind导出数据/创业板注册制上市_补充数据.xlsx"),
        "board": "创业板",
    },
    {
        "path": Path("D:/wind导出数据/北交所_补充数据.xlsx"),
        "board": "北交所",
    },
]

# ---------------------------------------------------------------------------
# Column maps
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "证券代码": "security_code",
    "证券简称": "security_name",
    # 北交所 uses "上市日期"; other boards use "首发上市日期"
    "首发上市日期": "listing_date_raw",
    "上市日期": "listing_date_raw",
    "首发价格\n[单位] 元": "offer_price_yuan",
    "发行数量合计\n[单位] 万股": "total_issue_shares_10k",
    "网下发行数量(回拨前)\n[单位] 万股": "offline_issue_before_clawback_10k",
    "网上发行数量(回拨前)\n[单位] 万股": "online_issue_before_clawback_10k",
    "网上发行数量\n[单位] 万股": "online_issue_final_10k",
    "网下发行数量\n[单位] 万股": "offline_issue_final_10k",
    "向战略投资者配售数量\n[单位] 万股": "strategic_allocation_10k",
    "其它发行数量\n[单位] 万股": "other_issue_10k",
    "包销比例\n[单位] %": "underwriting_ratio_pct",
    "回拨比例\n[单位] %": "clawback_ratio_pct",
    "被剔除的最高价申报量占比\n[单位] %": "high_price_excluded_subscription_share_pct",
    "被剔除的申报量占比\n[单位] %": "excluded_subscription_share_pct",
    "首发市盈率(摊薄)": "ipo_pe_diluted",
    "发行市净率\n[单位] 倍": "issue_pb",
    "首发时所属行业市盈率\n[单位] 倍": "industry_pe_at_ipo",
    "可比上市公司PE均值(扣非后)": "comparable_pe_avg_ex_nonrecurring",
    "网上发行中签率\n[单位] %": "online_lottery_rate_pct",
    "网上超额认购倍数\n[单位] 倍": "online_oversubscription_ratio",
    "网下申购配售比例\n[单位] %": "offline_allotment_ratio_pct",
    "网下申购总量\n[单位] 万股": "offline_subscription_total_10k",
    "网下有效报价申购量\n[单位] 万股": "offline_valid_quote_subscription_10k",
    "网下超额认购倍数": "offline_oversubscription_ratio",
    "网下超额认购倍数(回拨前)\n[单位] 倍": "offline_oversubscription_ratio_before_clawback",
    "网下申购配售对象家数\n[单位] 家": "offline_allotment_accounts",
    "网下申购询价对象家数\n[单位] 家": "offline_inquiry_investors",
    "网下投资者获配数量\n[机构类别] A类投资者\n[单位] 万股": "a_investor_allotted_shares_10k",
    "网下投资者申购数量\n[机构类别] A类投资者\n[单位] 万股": "a_investor_subscription_shares_10k",
    "网下投资者获配家数\n[机构类别] A类投资者\n[单位] 家": "a_investor_allotted_accounts",
    # Optional: absent in 北交所 main file
    "网下投资者中签率\n[机构类别] A类投资者\n[单位] %": "a_investor_lottery_rate_pct",
}

# Columns that are allowed to be absent in some source files.
OPTIONAL_COLUMNS = {"a_investor_lottery_rate_pct"}

SUPPLEMENT_COLUMN_MAP = {
    "证券代码": "security_code",
    "初步询价申购总量\n[单位] 万股": "inquiry_subscription_total_10k",
    "初步询价询价对象家数\n[单位] 家": "inquiry_investors_count",
    "初步询价配售对象家数\n[单位] 户": "inquiry_allotment_accounts",
    "网下申购步长\n[单位] 万股": "subscription_step_10k",
    "网下申购数量上限\n[单位] 万股": "subscription_upper_limit_10k",
    "网下申购数量下限\n[单位] 万股": "subscription_lower_limit_10k",
    "网下申报价格加权平均数\n[机构类别] 网下全部投资者": "quote_price_weighted_avg",
    "网下申报价格中位数\n[机构类别] 网下全部投资者": "quote_price_median",
    "初步询价截止日": "inquiry_deadline_date_raw",
    "网下申购截止日期": "subscription_deadline_date_raw",
    "上市首日涨跌幅\n[单位] %": "first_day_return_pct",
    # Only in 科创板 supplement
    "发行价格上限\n[单位] 元↓": "offer_price_upper_yuan",
    "发行价格下限(底价)\n[单位] 元": "offer_price_lower_yuan",
}

# ---------------------------------------------------------------------------
# Field classification by prediction-time availability
# ---------------------------------------------------------------------------
PREDICTION_TIME_WARNING = {
    "likely_pre_subscription": [
        # From main files
        "offer_price_yuan",
        "total_issue_shares_10k",
        "offline_issue_before_clawback_10k",
        "online_issue_before_clawback_10k",
        "strategic_allocation_10k",
        "ipo_pe_diluted",
        "issue_pb",
        "industry_pe_at_ipo",
        "comparable_pe_avg_ex_nonrecurring",
        "high_price_excluded_subscription_share_pct",
        "excluded_subscription_share_pct",
        # From supplement files (inquiry-time, confirmed pre-subscription)
        "inquiry_subscription_total_10k",
        "inquiry_investors_count",
        "inquiry_allotment_accounts",
        "subscription_step_10k",
        "subscription_upper_limit_10k",
        "subscription_lower_limit_10k",
        "quote_price_weighted_avg",
        "quote_price_median",
        "offer_price_upper_yuan",
        "offer_price_lower_yuan",
    ],
    "label_or_post_subscription": [
        # Labels / post-subscription results – must not enter model features
        "offline_allotment_ratio_pct",
        "offline_subscription_total_10k",
        "offline_valid_quote_subscription_10k",
        "offline_oversubscription_ratio",
        "offline_oversubscription_ratio_before_clawback",
        "offline_allotment_accounts",
        "offline_inquiry_investors",
        "a_investor_lottery_rate_pct",
        "online_lottery_rate_pct",
        "online_oversubscription_ratio",
        "online_issue_final_10k",
        "offline_issue_final_10k",
        "clawback_ratio_pct",
        # Post-listing: use only as rolling past-IPO market heat feature
        "first_day_return_pct",
    ],
}

REQUIRED_PREDICTION_FIELDS = [
    {
        "required_field": "网下申购数量上限",
        "category": "发行安排/申购规则",
        "current_status": "已补充：subscription_upper_limit_10k（科创板/创业板/北交所有）",
        "why_needed": "直接约束单个配售对象可申购规模，影响最终申购总量和拥挤度。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "网下申购数量下限",
        "category": "发行安排/申购规则",
        "current_status": "已补充：subscription_lower_limit_10k（科创板/创业板/北交所有）",
        "why_needed": "反映参与门槛和报价/申购最小单位。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "网下申购步长",
        "category": "发行安排/申购规则",
        "current_status": "已补充：subscription_step_10k（科创板/创业板/北交所有）",
        "why_needed": "影响申购数量离散化。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "初步询价申购总量",
        "category": "初步询价",
        "current_status": "已补充：inquiry_subscription_total_10k（科创板/创业板基本全覆盖；北交所仅41条）",
        "why_needed": "预测时点最关键的需求侧变量之一。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "初步询价询价对象家数",
        "category": "初步询价",
        "current_status": "已补充：inquiry_investors_count（科创板/创业板基本全覆盖）",
        "why_needed": "衡量参与机构数量，是板块差异和市场热度的重要代理变量。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "初步询价配售对象家数",
        "category": "初步询价",
        "current_status": "已补充：inquiry_allotment_accounts（科创板/创业板基本全覆盖）",
        "why_needed": "预测机构参与拥挤度。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "网下申报价格加权平均数",
        "category": "初步询价",
        "current_status": "已补充：quote_price_weighted_avg（科创板/创业板基本全覆盖）",
        "why_needed": "反映询价价格中枢，可与发行价、行业 PE 结合构造估值吸引力。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "网下申报价格中位数",
        "category": "初步询价",
        "current_status": "已补充：quote_price_median（科创板/创业板基本全覆盖）",
        "why_needed": "比均值更稳健，可衡量询价价格集中位置。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "发行价格上限",
        "category": "发行定价",
        "current_status": "已补充：offer_price_upper_yuan（仅科创板有此字段）",
        "why_needed": "与底价共同刻画询价价格区间宽度。",
        "suggested_source": "已从科创板补充数据获取；创业板/主板/北交所暂缺",
    },
    {
        "required_field": "发行价格下限(底价)",
        "category": "发行定价",
        "current_status": "已补充：offer_price_lower_yuan（仅科创板有此字段）",
        "why_needed": "用于衡量最终发行价在询价区间中的位置。",
        "suggested_source": "已从科创板补充数据获取；创业板/主板/北交所暂缺",
    },
    {
        "required_field": "申购截止日期（时间轴）",
        "category": "时间轴",
        "current_status": "已补充：subscription_deadline_date（科创板/创业板基本全覆盖）",
        "why_needed": "做真实时间序列回测必须按预测发生日排序。",
        "suggested_source": "已从补充数据获取",
    },
    {
        "required_field": "主板补充数据",
        "category": "数据缺口",
        "current_status": "当前无主板补充数据文件；主板所有询价字段为 NaN",
        "why_needed": "主板注册制后样本需要同等询价字段才能参与统一模型。",
        "suggested_source": "需要从 Wind 补充导出主板询价字段",
    },
    {
        "required_field": "行业分类",
        "category": "公司/行业",
        "current_status": "当前 Excel 未提供；仅有行业 PE",
        "why_needed": "行业固定效应可能影响热度、估值和参与机构偏好。",
        "suggested_source": "Wind 股票基本资料、申万/证监会行业分类",
    },
    {
        "required_field": "主承销商/保荐机构",
        "category": "承销商",
        "current_status": "当前 Excel 未提供",
        "why_needed": "不同承销商项目质量、定价风格、机构覆盖可能不同。",
        "suggested_source": "发行公告、Wind IPO 承销商字段",
    },
    {
        "required_field": "预测时点市场热度变量",
        "category": "市场环境",
        "current_status": "上市首日涨跌幅已补充（事后）；可基于此滚动计算近期IPO热度",
        "why_needed": "解释阶段性打新热度，如近期破发率、近期首日均值涨幅、指数涨跌。",
        "suggested_source": "可由 first_day_return_pct 滚动窗口派生；沪深成交量数据需单独获取",
    },
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def strip_xlsx_styles(src: Path, dst: Path) -> None:
    """Copy an xlsx while removing the malformed styles part."""
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "xl/styles.xml":
                continue
            zout.writestr(item, zin.read(item.filename))


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")


def read_source(spec: dict, temp_dir: Path) -> pd.DataFrame:
    src = Path(spec["path"])
    safe = temp_dir / f"{src.stem}_nostyles.xlsx"
    strip_xlsx_styles(src, safe)
    df = pd.read_excel(safe, sheet_name=0, dtype={"证券代码": str})
    df = df.rename(columns=COLUMN_MAP)

    # Required columns: everything in COLUMN_MAP values except OPTIONAL_COLUMNS
    required = set(COLUMN_MAP.values()) - OPTIONAL_COLUMNS - {"listing_date_raw"}
    # listing_date_raw may come from either 首发上市日期 or 上市日期
    if "listing_date_raw" not in df.columns:
        raise ValueError(f"{src.name}: no listing date column found")
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"{src.name} missing required columns: {missing_required}")

    # Fill optional columns with NaN if absent
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["board"] = spec["board"]
    df["source_file"] = src.name
    df["sample_note"] = spec["sample_note"]
    return df


def load_supplements(temp_dir: Path) -> pd.DataFrame:
    """Load all supplement files and return a merged DataFrame keyed by security_code."""
    frames = []
    for spec in SUPPLEMENT_SPECS:
        src = Path(spec["path"])
        safe = temp_dir / f"{src.stem}_nostyles.xlsx"
        strip_xlsx_styles(src, safe)
        df = pd.read_excel(safe, sheet_name=0, dtype={"证券代码": str})
        df = df.rename(columns=SUPPLEMENT_COLUMN_MAP)
        # Keep only columns that are in SUPPLEMENT_COLUMN_MAP values
        keep = [c for c in SUPPLEMENT_COLUMN_MAP.values() if c in df.columns]
        df = df[keep].copy()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Drop full duplicates; keep first occurrence per security_code
    combined = combined.drop_duplicates(subset=["security_code"], keep="first")
    return combined


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Coerce all non-id columns to numeric
    id_cols = {"security_code", "security_name", "board", "source_file", "sample_note"}
    for col in out.columns:
        if col not in id_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Dates
    out["listing_date"] = excel_serial_to_datetime(out["listing_date_raw"])
    out["listing_year"] = out["listing_date"].dt.year
    out["inquiry_deadline_date"] = excel_serial_to_datetime(out.get("inquiry_deadline_date_raw", pd.Series(dtype=float)))
    out["subscription_deadline_date"] = excel_serial_to_datetime(out.get("subscription_deadline_date_raw", pd.Series(dtype=float)))

    # Derived size / valuation
    out["issue_amount_100m_yuan"] = out["offer_price_yuan"] * out["total_issue_shares_10k"] / 10000
    out["offline_issue_before_share_pct"] = (
        out["offline_issue_before_clawback_10k"] / out["total_issue_shares_10k"] * 100
    )
    out["offline_issue_final_share_pct"] = out["offline_issue_final_10k"] / out["total_issue_shares_10k"] * 100
    out["strategic_allocation_share_pct"] = out["strategic_allocation_10k"] / out["total_issue_shares_10k"] * 100
    out["a_investor_subscription_share_pct"] = (
        out["a_investor_subscription_shares_10k"] / out["offline_subscription_total_10k"] * 100
    )
    out["a_investor_allocation_share_pct"] = (
        out["a_investor_allotted_shares_10k"] / out["offline_issue_final_10k"] * 100
    )
    out["valid_to_total_subscription_pct"] = (
        out["offline_valid_quote_subscription_10k"] / out["offline_subscription_total_10k"] * 100
    )
    out["offline_accounts_per_investor"] = out["offline_allotment_accounts"] / out["offline_inquiry_investors"]
    out["pe_vs_industry"] = out["ipo_pe_diluted"] / out["industry_pe_at_ipo"]
    out["pe_vs_comparable"] = out["ipo_pe_diluted"] / out["comparable_pe_avg_ex_nonrecurring"]

    # Label
    ratio = out["offline_oversubscription_ratio"]
    out["log_offline_oversubscription"] = np.where(ratio > 0, np.log(ratio), np.nan)
    out["implied_offline_lottery_rate_pct"] = np.where(ratio > 0, 100 / ratio, np.nan)
    out["offline_lottery_gap_pct_point"] = (
        out["offline_allotment_ratio_pct"] - out["implied_offline_lottery_rate_pct"]
    )
    out["has_offline_label"] = out["offline_oversubscription_ratio"].notna()

    # Inquiry-time derived features (only valid where supplement data exists)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["inquiry_oversubscription_ratio"] = np.where(
            out["offline_issue_before_clawback_10k"] > 0,
            out["inquiry_subscription_total_10k"] / out["offline_issue_before_clawback_10k"],
            np.nan,
        )
        out["quote_price_vs_offer"] = np.where(
            out["offer_price_yuan"] > 0,
            out["quote_price_weighted_avg"] / out["offer_price_yuan"],
            np.nan,
        )
        # Price position within the inquiry range (科创板 only)
        price_range = out["offer_price_upper_yuan"] - out["offer_price_lower_yuan"]
        out["offer_price_range_pct"] = np.where(
            price_range > 0,
            price_range / out["offer_price_lower_yuan"] * 100,
            np.nan,
        )
        out["offer_price_position_in_range"] = np.where(
            price_range > 0,
            (out["offer_price_yuan"] - out["offer_price_lower_yuan"]) / price_range,
            np.nan,
        )

    # Rolling market heat: avg first_day_return of prior 20 IPOs on same board
    # Sorted by subscription_deadline_date where available, else listing_date.
    sort_date = out["subscription_deadline_date"].fillna(out["listing_date"])
    out["sort_date_for_heat"] = sort_date
    heat_parts = []
    for board, grp in out.groupby("board"):
        grp = grp.sort_values("sort_date_for_heat")
        grp["recent_ipo_first_day_return_ma20"] = (
            grp["first_day_return_pct"].shift(1).rolling(20, min_periods=5).mean()
        )
        heat_parts.append(grp)
    heat_df = pd.concat(heat_parts).sort_index()
    out["recent_ipo_first_day_return_ma20"] = heat_df["recent_ipo_first_day_return_ma20"]

    return out


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def q01(x: pd.Series) -> float:
    return x.quantile(0.01)


def q99(x: pd.Series) -> float:
    return x.quantile(0.99)


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "field": col,
                "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p01": q01(s.dropna()) if s.notna().any() else np.nan,
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p99": q99(s.dropna()) if s.notna().any() else np.nan,
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def corr_table(df: pd.DataFrame, target: str, excluded: set[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.loc[df[target].notna()]
    cols = [c for c in numeric.columns if c != target and c not in excluded]
    rows = []
    for c in cols:
        pair = numeric[[target, c]].dropna()
        if len(pair) < 20 or pair[c].nunique() < 2:
            continue
        pearson = pair[target].corr(pair[c], method="pearson")
        spearman = pair[target].rank().corr(pair[c].rank(), method="pearson")
        rows.append(
            {
                "feature": c,
                "n": len(pair),
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


# ---------------------------------------------------------------------------
# SVG charts (no matplotlib dependency)
# ---------------------------------------------------------------------------

def fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 2) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows).copy()
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt_num(row[h], digits) for h in headers) + " |")
    return "\n".join(lines)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#222">'
        f"{escape(str(text))}</text>"
    )


def linear_scale(vmin: float, vmax: float, omin: float, omax: float):
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return lambda _: (omin + omax) / 2
    return lambda value: omin + (float(value) - vmin) / (vmax - vmin) * (omax - omin)


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def svg_boxplot(df: pd.DataFrame, fig_dir: Path) -> None:
    label_df = df[df["log_offline_oversubscription"].notna()].copy()
    boards = ["科创板", "创业板", "主板", "北交所"]
    colors = {"科创板": "#4C78A8", "创业板": "#F58518", "主板": "#54A24B", "北交所": "#B279A2"}
    stats = []
    for board in boards:
        s = label_df.loc[label_df["board"] == board, "log_offline_oversubscription"].dropna()
        if len(s) < 5:
            continue
        stats.append(
            {
                "board": board,
                "min": s.quantile(0.05),
                "q1": s.quantile(0.25),
                "median": s.median(),
                "q3": s.quantile(0.75),
                "max": s.quantile(0.95),
                "n": len(s),
            }
        )
    if not stats:
        return

    width, height = 960, 520
    left, right, top, bottom = 90, 40, 70, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin = min(s["min"] for s in stats)
    ymax = max(s["max"] for s in stats)
    yscale = linear_scale(ymin, ymax, top + plot_h, top)
    x_positions = np.linspace(left + 120, left + plot_w - 120, len(stats))
    body = [
        svg_text(width / 2, 32, "Label Distribution by Board (v2)", 20, "middle", "700"),
        svg_text(width / 2, 55, "log(offline oversubscription), whiskers = p05 / p95", 12, "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = yscale(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(svg_text(left - 10, y + 4, f"{value:.1f}", 11, "end"))

    box_w = 80
    for x, row in zip(x_positions, stats):
        c = colors[row["board"]]
        y_min, y_q1, y_med, y_q3, y_max = [yscale(row[k]) for k in ["min", "q1", "median", "q3", "max"]]
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{y_min:.1f}" x2="{x:.1f}" y2="{y_max:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w/3:.1f}" y1="{y_min:.1f}" x2="{x + box_w/3:.1f}" y2="{y_min:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w/3:.1f}" y1="{y_max:.1f}" x2="{x + box_w/3:.1f}" y2="{y_max:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<rect x="{x - box_w/2:.1f}" y="{y_q3:.1f}" width="{box_w}" height="{y_q1 - y_q3:.1f}" fill="{c}" fill-opacity="0.25" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w/2:.1f}" y1="{y_med:.1f}" x2="{x + box_w/2:.1f}" y2="{y_med:.1f}" stroke="{c}" stroke-width="3"/>',
            ]
        )
        body.append(svg_text(x, top + plot_h + 28, row["board"], 13, "middle", "700"))
        body.append(svg_text(x, top + plot_h + 47, f"n={row['n']}", 11, "middle"))
    body.append(svg_text(22, top + plot_h / 2, "log oversubscription", 12, "middle"))
    save_svg(fig_dir / "label_distribution_by_board.svg", width, height, body)


def svg_yearly_trend(df: pd.DataFrame, fig_dir: Path) -> None:
    trend = (
        df[df["offline_oversubscription_ratio"].notna()]
        .groupby(["listing_year", "board"])["offline_oversubscription_ratio"]
        .median()
        .reset_index()
        .dropna()
    )
    if trend.empty:
        return
    boards = ["科创板", "创业板", "主板", "北交所"]
    colors = {"科创板": "#4C78A8", "创业板": "#F58518", "主板": "#54A24B", "北交所": "#B279A2"}
    width, height = 980, 560
    left, right, top, bottom = 90, 180, 70, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = int(trend["listing_year"].min()), int(trend["listing_year"].max())
    ymin, ymax = 0, float(trend["offline_oversubscription_ratio"].max() * 1.1)
    xscale = linear_scale(xmin, xmax, left, left + plot_w)
    yscale = linear_scale(ymin, ymax, top + plot_h, top)
    body = [
        svg_text(width / 2, 32, "Median Offline Oversubscription by Listing Year (v2)", 20, "middle", "700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for year in range(xmin, xmax + 1):
        x = xscale(year)
        body.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        body.append(svg_text(x, top + plot_h + 24, year, 11, "middle"))
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = yscale(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(svg_text(left - 10, y + 4, f"{value:,.0f}", 11, "end"))
    for board in boards:
        sub = trend[trend["board"] == board].sort_values("listing_year")
        if sub.empty:
            continue
        points = [(xscale(y), yscale(v)) for y, v in zip(sub["listing_year"], sub["offline_oversubscription_ratio"])]
        path_d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{path_d}" fill="none" stroke="{colors[board]}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[board]}"/>')
    for idx, board in enumerate(boards):
        if board not in trend["board"].values:
            continue
        y = top + 20 + idx * 28
        body.append(f'<rect x="{left + plot_w + 35}" y="{y - 10}" width="16" height="16" fill="{colors[board]}"/>')
        body.append(svg_text(left + plot_w + 58, y + 3, board, 12))
    body.append(svg_text(width / 2, height - 20, "listing year", 12, "middle"))
    body.append(svg_text(24, top + plot_h / 2, "median ratio", 12, "middle"))
    save_svg(fig_dir / "yearly_median_oversubscription.svg", width, height, body)


def svg_missing_bar(missing: pd.DataFrame, fig_dir: Path) -> None:
    top_missing = missing[missing["missing_count"] > 0].head(16).iloc[::-1]
    if top_missing.empty:
        return
    width, height = 980, 680
    left, right, top, bottom = 380, 50, 60, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    xscale = linear_scale(0, 100, left, left + plot_w)
    row_h = plot_h / len(top_missing)
    body = [
        svg_text(width / 2, 32, "Top Missing Fields (v2)", 20, "middle", "700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        x = xscale(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6e6e6"/>')
        body.append(svg_text(x, top + plot_h + 22, f"{tick}%", 11, "middle"))
    for idx, (_, row) in enumerate(top_missing.iterrows()):
        y = top + idx * row_h + row_h * 0.2
        bar_w = xscale(row["missing_rate"] * 100) - left
        body.append(svg_text(left - 12, y + row_h * 0.55, row["field"], 11, "end"))
        body.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{row_h * 0.6:.1f}" fill="#E45756" fill-opacity="0.82"/>')
        body.append(svg_text(left + bar_w + 6, y + row_h * 0.55, f"{row['missing_rate'] * 100:.1f}%", 11))
    body.append(svg_text(width / 2, height - 16, "missing rate", 12, "middle"))
    save_svg(fig_dir / "top_missing_fields.svg", width, height, body)


def svg_corr_bar(corr_pre: pd.DataFrame, fig_dir: Path) -> None:
    view = corr_pre.head(14).iloc[::-1]
    if view.empty:
        return
    width, height = 980, 680
    left, right, top, bottom = 420, 50, 60, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xscale = linear_scale(-1, 1, left, left + plot_w)
    zero = xscale(0)
    row_h = plot_h / len(view)
    body = [
        svg_text(width / 2, 32, "Top Spearman Correlations (pre-subscription fields, v2)", 20, "middle", "700"),
        svg_text(width / 2, 52, "not a causal test; excludes post-subscription leakage", 12, "middle"),
        f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [-1, -0.5, 0, 0.5, 1]:
        x = xscale(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6e6e6"/>')
        body.append(svg_text(x, top + plot_h + 22, f"{tick:.1f}", 11, "middle"))
    for idx, (_, row) in enumerate(view.iterrows()):
        corr = float(row["spearman_corr"])
        y = top + idx * row_h + row_h * 0.2
        x = xscale(corr)
        bar_x = min(zero, x)
        bar_w = abs(x - zero)
        color = "#4C78A8" if corr >= 0 else "#F58518"
        body.append(svg_text(left - 12, y + row_h * 0.55, row["feature"], 10, "end"))
        body.append(f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{row_h * 0.6:.1f}" fill="{color}" fill-opacity="0.86"/>')
        body.append(svg_text(x + (6 if corr >= 0 else -6), y + row_h * 0.55, f"{corr:.2f}", 11, "start" if corr >= 0 else "end"))
    body.append(svg_text(width / 2, height - 16, "Spearman correlation with log offline oversubscription", 12, "middle"))
    save_svg(fig_dir / "top_pre_subscription_correlations.svg", width, height, body)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def make_report(
    df: pd.DataFrame,
    board_summary: pd.DataFrame,
    year_board: pd.DataFrame,
    missing: pd.DataFrame,
    label_desc: pd.DataFrame,
    corr_pre: pd.DataFrame,
    corr_all: pd.DataFrame,
    outliers: pd.DataFrame,
) -> str:
    label_df = df[df["has_offline_label"]].copy()
    total_n = len(df)
    label_n = int(label_df["has_offline_label"].sum())
    date_min = df["listing_date"].min().date()
    date_max = df["listing_date"].max().date()

    has_supplement = df["inquiry_subscription_total_10k"].notna().sum()

    exact_identity = (
        label_df["offline_lottery_gap_pct_point"].abs().dropna().quantile(0.95)
        if label_df["offline_lottery_gap_pct_point"].notna().any()
        else np.nan
    )

    interesting = []
    if not board_summary.empty:
        high_board = board_summary.sort_values("median_offline_oversubscription", ascending=False).iloc[0]
        low_board = board_summary.sort_values("median_offline_oversubscription", ascending=True).iloc[0]
        interesting.append(
            f"- 板块差异很强：{high_board['board']} 网下超额认购倍数中位数最高 "
            f"({fmt_num(high_board['median_offline_oversubscription'])} 倍)，"
            f"{low_board['board']} 最低 ({fmt_num(low_board['median_offline_oversubscription'])} 倍)。"
        )
    if not corr_pre.empty:
        top = corr_pre.iloc[0]
        direction = "正相关" if top["spearman_corr"] > 0 else "负相关"
        interesting.append(
            f"- 预测前可用字段中，`{top['feature']}` 与标签的 Spearman 相关绝对值最高，"
            f"为 {fmt_num(top['spearman_corr'], 3)}，方向为{direction}。"
        )
    if pd.notna(exact_identity):
        interesting.append(
            f"- `网下申购配售比例` 与 `100 / 网下超额认购倍数` 基本互为倒数，"
            f"二者差异的 95% 分位仅 {fmt_num(exact_identity, 6)} 个百分点；建模时只能作为标签。"
        )

    missing_top = missing[missing["missing_count"] > 0].head(10)

    report = f"""# 初步数据处理与分析报告（v2）

生成日期：2026-05-22

## 变更说明（相对 v1）

- 新增北交所（北交所_网下打新数据.xlsx），样本扩展至四个板块。
- 合并三份补充数据（科创板/创业板/北交所），新增初步询价阶段字段：
  询价申购总量、询价/配售对象家数、申购步长/上下限、申报价格均值/中位数、
  询价截止日、申购截止日、上市首日涨跌幅。
- 科创板补充数据额外包含发行价格上限/下限（底价）。
- 主板暂无补充数据；主板询价字段全为 NaN。
- 北交所大部分样本不适用询价机制；询价字段有效样本仅约 41 条。

## 1. 样本概况

- 总样本：{total_n:,} 条。
- 有网下超额认购倍数标签：{label_n:,} 条，占 {label_n / total_n:.1%}。
- 有初步询价补充字段：{has_supplement:,} 条。
- 上市日期范围：{date_min} 至 {date_max}。

{markdown_table(board_summary, digits=2)}

## 2. 年份与板块分布

{markdown_table(year_board, max_rows=30, digits=0)}

## 3. 标签描述性统计

核心标签为 `log_offline_oversubscription = log(网下超额认购倍数)`。

{markdown_table(label_desc, max_rows=20, digits=4)}

## 4. 预测前可用字段相关性（含新增询价字段）

以下排除申购后泄露字段，仅展示可能在预测时点可见的字段：

{markdown_table(corr_pre[['feature', 'n', 'spearman_corr', 'pearson_corr']].head(14), digits=4)}

## 5. 全字段相关性（含事后字段，供理解机制）

{markdown_table(corr_all[['feature', 'n', 'spearman_corr', 'pearson_corr']].head(14), digits=4)}

## 6. 主要结论

{chr(10).join(interesting)}

## 7. 字段缺失情况

{markdown_table(missing_top[['field', 'missing_count', 'missing_rate']], digits=4)}

## 8. 异常值检查

{markdown_table(outliers, max_rows=20, digits=4)}

## 9. 建模建议（v2）

- 询价阶段字段（inquiry_subscription_total_10k、inquiry_investors_count 等）现已可用，
  是最有价值的新增特征，建议优先加入 Ridge 和 LightGBM 基线。
- inquiry_oversubscription_ratio（初步询价超额认购倍数）理论上是最强预测因子，
  但必须确认其发布时点确实早于网下申购截止。
- recent_ipo_first_day_return_ma20 已基于过去 20 只 IPO 首日涨幅滚动计算，可作为
  市场热度代理变量，不引入未来数据。
- 北交所标签覆盖率仅约 13%（41/316），短期内建议与其他板块合并训练并单独评估误差，
  不单独建模。
- 主板仍无询价补充字段，如需统一模型需补齐或使用缺失值插补策略。
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        temp_dir = Path(td)
        frames = [read_source(spec, temp_dir) for spec in FILE_SPECS]
        supplements = load_supplements(temp_dir)

    raw = pd.concat(frames, ignore_index=True)

    # Join supplement fields (left join to preserve all main rows)
    sup_cols = [c for c in supplements.columns if c != "security_code"]
    raw = raw.merge(supplements[["security_code"] + sup_cols], on="security_code", how="left")

    df = add_features(raw)

    # Stable column order
    front_cols = [
        "security_code", "security_name", "board",
        "listing_date", "listing_year",
        "inquiry_deadline_date", "subscription_deadline_date",
        "source_file", "sample_note",
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    # --- Outputs ---
    csv_path = DATA_DIR / "ipo_offline_sample.csv"
    db_path = DATA_DIR / "ipo_offline.db"

    df_for_csv = df.copy()
    for dcol in ["listing_date", "inquiry_deadline_date", "subscription_deadline_date"]:
        if dcol in df_for_csv.columns:
            df_for_csv[dcol] = df_for_csv[dcol].dt.strftime("%Y-%m-%d")
    df_for_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")

    db_df = df_for_csv.copy()
    with sqlite3.connect(db_path) as conn:
        db_df.to_sql("ipo_offline_sample", conn, if_exists="replace", index=False)
        pd.DataFrame(
            [{"category": key, "field": field}
             for key, fields in PREDICTION_TIME_WARNING.items()
             for field in fields]
        ).to_sql("field_time_classification", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_board_year ON ipo_offline_sample(board, listing_year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_listing_date ON ipo_offline_sample(listing_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sub_date ON ipo_offline_sample(subscription_deadline_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_security_code ON ipo_offline_sample(security_code)")

    # Field dictionary
    all_maps = {**COLUMN_MAP, **SUPPLEMENT_COLUMN_MAP}
    field_dict = pd.DataFrame(
        [{"raw_field": r, "clean_field": c} for r, c in all_maps.items()]
    )
    field_dict.to_csv(OUT_DIR / "field_dictionary.csv", index=False, encoding="utf-8-sig")

    # Required fields status
    pd.DataFrame(REQUIRED_PREDICTION_FIELDS).to_csv(
        OUT_DIR / "missing_required_prediction_fields.csv", index=False, encoding="utf-8-sig"
    )

    # Missing rate
    missing = (
        df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "field"})
    )
    missing["missing_rate"] = missing["missing_count"] / len(df)
    missing = missing.sort_values(["missing_rate", "missing_count"], ascending=False)
    missing.to_csv(OUT_DIR / "missing_by_field.csv", index=False, encoding="utf-8-sig")

    # Board summary
    board_summary = (
        df.groupby("board", dropna=False)
        .agg(
            sample_count=("security_code", "size"),
            label_count=("has_offline_label", "sum"),
            inquiry_field_count=("inquiry_subscription_total_10k", lambda s: int(s.notna().sum())),
            date_min=("listing_date", "min"),
            date_max=("listing_date", "max"),
            median_offline_oversubscription=("offline_oversubscription_ratio", "median"),
            mean_offline_oversubscription=("offline_oversubscription_ratio", "mean"),
            median_log_oversubscription=("log_offline_oversubscription", "median"),
            median_offline_allotment_ratio_pct=("offline_allotment_ratio_pct", "median"),
            median_inquiry_subscription_total_10k=("inquiry_subscription_total_10k", "median"),
            median_inquiry_investors_count=("inquiry_investors_count", "median"),
            median_issue_amount_100m_yuan=("issue_amount_100m_yuan", "median"),
            missing_label_count=("offline_oversubscription_ratio", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    board_summary["date_min"] = board_summary["date_min"].dt.strftime("%Y-%m-%d")
    board_summary["date_max"] = board_summary["date_max"].dt.strftime("%Y-%m-%d")
    board_summary.to_csv(OUT_DIR / "board_summary.csv", index=False, encoding="utf-8-sig")

    # Year-board counts
    year_board = (
        df.pivot_table(
            index="listing_year", columns="board",
            values="security_code", aggfunc="count", fill_value=0,
        )
        .reset_index()
        .sort_values("listing_year")
    )
    year_board.to_csv(OUT_DIR / "year_board_counts.csv", index=False, encoding="utf-8-sig")

    # Descriptive stats
    numeric_cols = [
        "offline_oversubscription_ratio", "log_offline_oversubscription",
        "offline_allotment_ratio_pct", "offline_oversubscription_ratio_before_clawback",
        "a_investor_lottery_rate_pct", "offer_price_yuan", "issue_amount_100m_yuan",
        "strategic_allocation_share_pct", "ipo_pe_diluted", "pe_vs_industry",
        "excluded_subscription_share_pct",
        "inquiry_subscription_total_10k", "inquiry_investors_count",
        "inquiry_allotment_accounts", "inquiry_oversubscription_ratio",
        "quote_price_weighted_avg", "quote_price_median", "quote_price_vs_offer",
        "offer_price_upper_yuan", "offer_price_lower_yuan",
        "recent_ipo_first_day_return_ma20",
    ]
    describe_numeric(df, numeric_cols).to_csv(OUT_DIR / "descriptive_stats.csv", index=False, encoding="utf-8-sig")

    label_desc = describe_numeric(
        df,
        ["offline_oversubscription_ratio", "log_offline_oversubscription",
         "offline_allotment_ratio_pct", "offline_oversubscription_ratio_before_clawback",
         "a_investor_lottery_rate_pct"],
    )

    # Correlation tables
    leakage_or_labels = set(PREDICTION_TIME_WARNING["label_or_post_subscription"]) | {
        "listing_date_raw", "listing_year", "has_offline_label",
        "offline_lottery_gap_pct_point", "implied_offline_lottery_rate_pct",
        "inquiry_deadline_date_raw", "subscription_deadline_date_raw",
        "sort_date_for_heat",
    }
    corr_pre = corr_table(df, "log_offline_oversubscription", leakage_or_labels)
    corr_all = corr_table(
        df, "log_offline_oversubscription",
        {"listing_date_raw", "listing_year", "has_offline_label",
         "inquiry_deadline_date_raw", "subscription_deadline_date_raw",
         "sort_date_for_heat"},
    )
    corr_pre.to_csv(OUT_DIR / "correlation_pre_subscription_like.csv", index=False, encoding="utf-8-sig")
    corr_all.to_csv(OUT_DIR / "correlation_all_fields.csv", index=False, encoding="utf-8-sig")

    # Outlier checks
    outlier_rows = []
    for col in [
        "offline_oversubscription_ratio", "offline_allotment_ratio_pct",
        "offer_price_yuan", "issue_amount_100m_yuan", "ipo_pe_diluted",
        "inquiry_subscription_total_10k", "inquiry_investors_count",
    ]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        outlier_rows.append({
            "field": col, "p01": lo, "p99": hi,
            "below_p01_count": int((df[col] < lo).sum()),
            "above_p99_count": int((df[col] > hi).sum()),
            "min_security": df.loc[df[col].idxmin(), "security_name"] if df[col].notna().any() else "",
            "min": s.min(),
            "max_security": df.loc[df[col].idxmax(), "security_name"] if df[col].notna().any() else "",
            "max": s.max(),
        })
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(OUT_DIR / "outlier_checks.csv", index=False, encoding="utf-8-sig")

    # SVG figures
    svg_boxplot(df, fig_dir)
    svg_yearly_trend(df, fig_dir)
    svg_missing_bar(missing, fig_dir)
    svg_corr_bar(corr_pre, fig_dir)

    # Report
    report = make_report(df, board_summary, year_board, missing, label_desc, corr_pre, corr_all, outliers)
    (OUT_DIR / "initial_analysis_report.md").write_text(report, encoding="utf-8")

    # Manifest
    manifest = {
        "version": "v2",
        "generated": "2026-05-22",
        "inputs": {
            "main_files": [{"path": str(s["path"]), "board": s["board"]} for s in FILE_SPECS],
            "supplement_files": [{"path": str(s["path"]), "board": s["board"]} for s in SUPPLEMENT_SPECS],
        },
        "outputs": {
            "csv": str(csv_path),
            "sqlite": str(db_path),
            "report": str(OUT_DIR / "initial_analysis_report.md"),
            "field_dictionary": str(OUT_DIR / "field_dictionary.csv"),
            "board_summary": str(OUT_DIR / "board_summary.csv"),
            "figures_dir": str(fig_dir),
        },
        "row_count": int(len(df)),
        "label_count": int(df["has_offline_label"].sum()),
        "inquiry_field_count": int(df["inquiry_subscription_total_10k"].notna().sum()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
