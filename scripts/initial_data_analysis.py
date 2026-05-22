"""Initial processing and EDA for Wind IPO offline subscription exports.

Inputs are the three Wind-exported Excel files listed in FILE_SPECS.  Some
Wind xlsx files contain malformed style metadata; the loader therefore creates
temporary no-style copies before reading with pandas.  Raw source files are not
modified.
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
]

COLUMN_MAP = {
    "证券代码": "security_code",
    "证券简称": "security_name",
    "首发上市日期": "listing_date_raw",
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
    "网下投资者中签率\n[机构类别] A类投资者\n[单位] %": "a_investor_lottery_rate_pct",
}

PREDICTION_TIME_WARNING = {
    "likely_pre_subscription": [
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
    ],
    "label_or_post_subscription": [
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
    ],
}

REQUIRED_PREDICTION_FIELDS = [
    {
        "required_field": "网下申购数量上限",
        "category": "发行安排/申购规则",
        "current_status": "当前 Excel 未提供",
        "why_needed": "直接约束单个配售对象可申购规模，影响最终申购总量和拥挤度。",
        "suggested_source": "发行公告、初步询价及推介公告、Wind IPO 发行明细字段",
    },
    {
        "required_field": "网下申购数量下限",
        "category": "发行安排/申购规则",
        "current_status": "当前 Excel 未提供",
        "why_needed": "反映参与门槛和报价/申购最小单位，可能影响小规模参与者数量。",
        "suggested_source": "发行公告、初步询价及推介公告、Wind IPO 发行明细字段",
    },
    {
        "required_field": "网下申购步长",
        "category": "发行安排/申购规则",
        "current_status": "当前 Excel 未提供",
        "why_needed": "影响申购数量离散化，后续可用于解释申购量分布。",
        "suggested_source": "发行公告、初步询价及推介公告",
    },
    {
        "required_field": "网下配售数量",
        "category": "发行规模",
        "current_status": "已有相近字段：网下发行数量(回拨前)、网下发行数量",
        "why_needed": "分母端核心变量；正式预测建议优先使用回拨前网下配售数量，避免回拨后泄露。",
        "suggested_source": "当前 Excel + 发行安排字段校验",
    },
    {
        "required_field": "战略配售获配股份数",
        "category": "战略配售",
        "current_status": "已有相近字段：向战略投资者配售数量；缺失率约 34.0%",
        "why_needed": "战略配售越多，留给网下/网上的份额越少，可能抬高拥挤度。",
        "suggested_source": "当前 Excel、战略配售结果公告、发行公告",
    },
    {
        "required_field": "战略配售获配股份占比",
        "category": "战略配售",
        "current_status": "已派生 strategic_allocation_share_pct；缺失率约 34.0%",
        "why_needed": "比绝对股数更便于跨规模样本比较。",
        "suggested_source": "由战略配售股数 / 发行总股数派生",
    },
    {
        "required_field": "主承销商战略获配股份数",
        "category": "战略配售/承销商",
        "current_status": "当前 Excel 未提供",
        "why_needed": "可衡量跟投安排和保荐承销商约束，科创板/创业板制度差异可能明显。",
        "suggested_source": "战略配售结果公告、保荐机构跟投公告、Wind 战略配售明细",
    },
    {
        "required_field": "主承销商战略获配股份占比",
        "category": "战略配售/承销商",
        "current_status": "当前 Excel 未提供",
        "why_needed": "用于跨发行规模比较承销商跟投强度。",
        "suggested_source": "由主承销商战略获配股数 / 发行总股数派生",
    },
    {
        "required_field": "网下投资者分类限售配售方式",
        "category": "限售安排",
        "current_status": "当前 Excel 未提供",
        "why_needed": "限售安排会影响机构参与意愿和报价/申购行为。",
        "suggested_source": "发行公告、网下发行初步配售结果公告",
    },
    {
        "required_field": "网下投资者分类配售限售比例",
        "category": "限售安排",
        "current_status": "当前 Excel 未提供",
        "why_needed": "量化限售约束强弱，适合做板块制度差异特征。",
        "suggested_source": "发行公告、网下发行初步配售结果公告",
    },
    {
        "required_field": "初步询价申报价格",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "价格分布能反映机构认可度和分歧程度。",
        "suggested_source": "初步询价结果及推迟发行公告、Wind 询价明细",
    },
    {
        "required_field": "网下申报价格加权平均数",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "反映询价价格中枢，可与发行价、行业 PE 结合构造估值吸引力。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "网下申报价格中位数",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "比均值更稳健，可衡量询价价格集中位置。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "初步询价申报数量",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "预测时点最关键的需求侧变量之一。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "初步询价配售对象家数",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供；现有网下申购配售对象家数更像申购后字段",
        "why_needed": "预测机构参与拥挤度，但必须使用询价阶段口径，避免泄露。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "初步询价询价对象家数",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供；现有网下申购询价对象家数需确认是否为申购后口径",
        "why_needed": "衡量参与机构数量，是板块差异和市场热度的重要代理变量。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "初步询价申购总量",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供；现有网下申购总量属于标签/事后字段",
        "why_needed": "若能在网下申购前获得，是预测超额认购倍数的核心变量。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "初步询价申购倍数(回拨前)",
        "category": "初步询价",
        "current_status": "当前 Excel 未提供；现有网下超额认购倍数(回拨前)属于标签",
        "why_needed": "可作为更早阶段的需求强度特征，但要严格确认公告时间。",
        "suggested_source": "初步询价结果公告、由初步询价申购总量 / 回拨前网下发行量派生",
    },
    {
        "required_field": "网下询价市值门槛",
        "category": "参与门槛",
        "current_status": "当前 Excel 未提供",
        "why_needed": "市值门槛影响可参与账户池大小，可能显著影响配售对象数量。",
        "suggested_source": "发行公告、初步询价及推介公告",
    },
    {
        "required_field": "网下询价市值门槛(A类)",
        "category": "参与门槛",
        "current_status": "当前 Excel 未提供",
        "why_needed": "A 类投资者口径可能与整体门槛不同，适合解释 A 类中签率。",
        "suggested_source": "发行公告、初步询价及推介公告",
    },
    {
        "required_field": "网下询价市值门槛(主题与战略)",
        "category": "参与门槛",
        "current_status": "当前 Excel 未提供",
        "why_needed": "主题/战略配售相关门槛有助于解释特殊发行安排。",
        "suggested_source": "发行公告、初步询价及推介公告",
    },
    {
        "required_field": "发行价格下限(底价)",
        "category": "发行定价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "用于衡量最终发行价在询价区间中的位置。",
        "suggested_source": "招股意向书、发行公告、询价公告",
    },
    {
        "required_field": "发行价格上限",
        "category": "发行定价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "与底价共同刻画询价价格区间宽度。",
        "suggested_source": "招股意向书、发行公告、询价公告",
    },
    {
        "required_field": "剔除无效和最高报价后申购总量",
        "category": "剔除后询价",
        "current_status": "当前 Excel 未提供；仅有剔除申报量占比",
        "why_needed": "比剔除比例更直接描述有效需求基数。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "剔除无效和最高报价后配售对象",
        "category": "剔除后询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "衡量剔除后真实参与账户数量，适合预测拥挤度。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
    },
    {
        "required_field": "剔除无效和最高报价后询价对象",
        "category": "剔除后询价",
        "current_status": "当前 Excel 未提供",
        "why_needed": "衡量剔除后真实参与机构数量，适合做需求侧特征。",
        "suggested_source": "初步询价结果公告、Wind 询价统计",
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
        "required_field": "发行公告日/申购日/询价截止日",
        "category": "时间轴",
        "current_status": "当前 Excel 只有上市日期",
        "why_needed": "做真实时间序列回测必须按预测发生日排序，而不是上市日。",
        "suggested_source": "发行公告、Wind IPO 日程字段",
    },
    {
        "required_field": "预测时点市场热度变量",
        "category": "市场环境",
        "current_status": "当前 Excel 未提供",
        "why_needed": "可用近期 IPO 破发率、上市首日收益、指数涨跌、成交额等解释阶段性热度。",
        "suggested_source": "Wind 市场数据、历史 IPO 首日表现、指数行情",
    },
]


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


def read_source(spec: dict[str, object], temp_dir: Path) -> pd.DataFrame:
    src = Path(spec["path"])
    safe = temp_dir / f"{src.stem}_nostyles.xlsx"
    strip_xlsx_styles(src, safe)
    df = pd.read_excel(safe, sheet_name=0, dtype={"证券代码": str})
    df = df.rename(columns=COLUMN_MAP)
    missing_mapped = sorted(set(COLUMN_MAP.values()) - set(df.columns))
    if missing_mapped:
        raise ValueError(f"{src.name} missing expected columns: {missing_mapped}")
    df["board"] = spec["board"]
    df["source_file"] = src.name
    df["sample_note"] = spec["sample_note"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in {"security_code", "security_name", "board", "source_file", "sample_note"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["listing_date"] = excel_serial_to_datetime(out["listing_date_raw"])
    out["listing_year"] = out["listing_date"].dt.year
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

    ratio = out["offline_oversubscription_ratio"]
    out["log_offline_oversubscription"] = np.where(ratio > 0, np.log(ratio), np.nan)
    out["implied_offline_lottery_rate_pct"] = np.where(ratio > 0, 100 / ratio, np.nan)
    out["offline_lottery_gap_pct_point"] = (
        out["offline_allotment_ratio_pct"] - out["implied_offline_lottery_rate_pct"]
    )
    out["has_offline_label"] = out["offline_oversubscription_ratio"].notna()
    return out


def q01(x: pd.Series) -> float:
    return x.quantile(0.01)


def q99(x: pd.Series) -> float:
    return x.quantile(0.99)


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
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
        # pandas delegates Spearman to scipy; rank correlation avoids that
        # optional dependency and is equivalent for this EDA purpose.
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
    boards = ["科创板", "创业板", "主板"]
    colors = {"科创板": "#4C78A8", "创业板": "#F58518", "主板": "#54A24B"}
    stats = []
    for board in boards:
        s = label_df.loc[label_df["board"] == board, "log_offline_oversubscription"].dropna()
        if s.empty:
            continue
        stats.append(
            {
                "board": board,
                "min": s.quantile(0.01),
                "q1": s.quantile(0.25),
                "median": s.median(),
                "q3": s.quantile(0.75),
                "max": s.quantile(0.99),
                "n": len(s),
            }
        )
    if not stats:
        return

    width, height = 900, 520
    left, right, top, bottom = 90, 40, 70, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin = min(s["min"] for s in stats)
    ymax = max(s["max"] for s in stats)
    yscale = linear_scale(ymin, ymax, top + plot_h, top)
    x_positions = np.linspace(left + 120, left + plot_w - 120, len(stats))
    body = [
        svg_text(width / 2, 32, "Label Distribution by Board", 20, "middle", "700"),
        svg_text(width / 2, 55, "log(offline oversubscription), whiskers = p01 / p99", 12, "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = yscale(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(svg_text(left - 10, y + 4, f"{value:.1f}", 11, "end"))

    box_w = 90
    for x, row in zip(x_positions, stats):
        c = colors[row["board"]]
        y_min, y_q1, y_med, y_q3, y_max = [yscale(row[k]) for k in ["min", "q1", "median", "q3", "max"]]
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{y_min:.1f}" x2="{x:.1f}" y2="{y_max:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 3:.1f}" y1="{y_min:.1f}" x2="{x + box_w / 3:.1f}" y2="{y_min:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 3:.1f}" y1="{y_max:.1f}" x2="{x + box_w / 3:.1f}" y2="{y_max:.1f}" stroke="{c}" stroke-width="2"/>',
                f'<rect x="{x - box_w / 2:.1f}" y="{y_q3:.1f}" width="{box_w}" height="{y_q1 - y_q3:.1f}" fill="{c}" fill-opacity="0.25" stroke="{c}" stroke-width="2"/>',
                f'<line x1="{x - box_w / 2:.1f}" y1="{y_med:.1f}" x2="{x + box_w / 2:.1f}" y2="{y_med:.1f}" stroke="{c}" stroke-width="3"/>',
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
    boards = ["科创板", "创业板", "主板"]
    colors = {"科创板": "#4C78A8", "创业板": "#F58518", "主板": "#54A24B"}
    width, height = 980, 560
    left, right, top, bottom = 90, 160, 70, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = int(trend["listing_year"].min()), int(trend["listing_year"].max())
    ymin, ymax = 0, float(trend["offline_oversubscription_ratio"].max() * 1.1)
    xscale = linear_scale(xmin, xmax, left, left + plot_w)
    yscale = linear_scale(ymin, ymax, top + plot_h, top)
    body = [
        svg_text(width / 2, 32, "Median Offline Oversubscription by Listing Year", 20, "middle", "700"),
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
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{path}" fill="none" stroke="{colors[board]}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[board]}"/>')
    for idx, board in enumerate(boards):
        y = top + 20 + idx * 28
        body.append(f'<rect x="{left + plot_w + 35}" y="{y - 10}" width="16" height="16" fill="{colors[board]}"/>')
        body.append(svg_text(left + plot_w + 58, y + 3, board, 12))
    body.append(svg_text(width / 2, height - 20, "listing year", 12, "middle"))
    body.append(svg_text(24, top + plot_h / 2, "median ratio", 12, "middle"))
    save_svg(fig_dir / "yearly_median_oversubscription.svg", width, height, body)


def svg_missing_bar(missing: pd.DataFrame, fig_dir: Path) -> None:
    top_missing = missing[missing["missing_count"] > 0].head(14).iloc[::-1]
    if top_missing.empty:
        return
    width, height = 980, 620
    left, right, top, bottom = 360, 50, 60, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    xscale = linear_scale(0, 100, left, left + plot_w)
    row_h = plot_h / len(top_missing)
    body = [
        svg_text(width / 2, 32, "Top Missing Fields", 20, "middle", "700"),
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
    view = corr_pre.head(12).iloc[::-1]
    if view.empty:
        return
    width, height = 980, 620
    left, right, top, bottom = 390, 50, 60, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xscale = linear_scale(-1, 1, left, left + plot_w)
    zero = xscale(0)
    row_h = plot_h / len(view)
    body = [
        svg_text(width / 2, 32, "Top Single-Field Spearman Correlations", 20, "middle", "700"),
        svg_text(width / 2, 52, "pre-subscription-like fields only; not a causal test", 12, "middle"),
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
            f"- 在较像预测前可用的字段里，`{top['feature']}` 与标签的 Spearman 相关绝对值最高，"
            f"为 {fmt_num(top['spearman_corr'], 3)}，方向为{direction}。"
        )
    if pd.notna(exact_identity):
        interesting.append(
            f"- `网下申购配售比例` 与 `100 / 网下超额认购倍数` 基本互为倒数，"
            f"二者差异的 95% 分位仅 {fmt_num(exact_identity, 6)} 个百分点；建模时二者只能作为标签/校验，不能同时做输入。"
        )
    if "offline_allotment_accounts" in corr_all["feature"].values:
        row = corr_all[corr_all["feature"] == "offline_allotment_accounts"].iloc[0]
        interesting.append(
            f"- 配售对象家数越多，竞争越拥挤：`offline_allotment_accounts` 与 log 超额认购倍数的 Spearman "
            f"约 {fmt_num(row['spearman_corr'], 3)}。但这个字段更像申购后结果，需确认预测时点。"
        )

    missing_top = missing[missing["missing_count"] > 0].head(8)

    report = f"""# 初步数据处理与分析报告

生成日期：2026-05-21

## 1. SQL / SQLite 是否值得用

值得用，但定位应是“中间层和回测查询层”，不是替代 pandas / sklearn 的建模层。

- 当前 3 个 Excel 合计 {total_n:,} 行、字段不多，单机 pandas 已经足够快。
- SQLite `.db` 的价值在于：统一字段名、保留原始来源、方便按板块/年份/时间窗口滚动取数、让后续网页或回测服务直接查询。
- 建议流程：Excel 原始层 -> 清洗后的 `ipo_offline_sample` 表 -> 特征/标签视图 -> pandas/sklearn 建模。
- 不建议一开始把所有特征工程都写进 SQL；复杂统计、缺失处理、分位裁剪、时间序列回测仍用 Python 更顺手。

本次已生成 SQLite 数据库：`data/processed/ipo_offline.db`，核心表为 `ipo_offline_sample`。

如环境安装了 matplotlib，脚本会在 `outputs/initial_analysis/figures/` 额外生成可视化图片；当前 bundled Python 未包含该依赖时会自动跳过。

## 2. 样本概况

- 总样本：{total_n:,} 条。
- 有网下超额认购倍数标签：{label_n:,} 条，占 {label_n / total_n:.1%}。
- 上市日期范围：{date_min} 至 {date_max}。
- 当前覆盖：科创板、注册制创业板、主板注册制后样本；尚未包含北交所。

{markdown_table(board_summary, digits=2)}

## 3. 年份与板块分布

{markdown_table(year_board, max_rows=30, digits=0)}

## 4. 标签描述性统计

核心标签为 `log_offline_oversubscription = log(网下超额认购倍数)`。

{markdown_table(label_desc, max_rows=20, digits=4)}

## 5. 可能对预测有用的规律

### 预测前较可能可用字段的相关性

以下仅是单变量相关，不代表因果，也没有经过严格时间序列验证：

{markdown_table(corr_pre[['feature', 'n', 'spearman_corr', 'pearson_corr']].head(12), digits=4)}

### 全字段相关性，包括明显事后字段

这些字段可用于理解标签形成机制，但正式预测时要谨慎排除泄露：

{markdown_table(corr_all[['feature', 'n', 'spearman_corr', 'pearson_corr']].head(12), digits=4)}

## 6. 有趣的数据结论

{chr(10).join(interesting)}

## 7. 缺少哪些数据

当前 Excel 缺少项目文档中许多“初步询价结束后、网下申购前”更理想的输入字段，尤其是：

- 网下申购数量上限、下限、步长。
- 初步询价申报价格明细、加权平均数、中位数。
- 初步询价申报数量、初步询价配售对象家数、询价对象家数。
- 初步询价申购总量、初步询价申购倍数（回拨前）。
- 网下询价市值门槛及 A 类/主题与战略门槛。
- 发行价格下限/上限。
- 剔除无效和最高报价后的申购总量、配售对象、询价对象。
- 主承销商战略获配股份数/占比。
- 行业、保荐机构/主承销商、发行日/申购日、发行阶段市场热度等上下文变量。

当前字段缺失率最高的字段如下：

{markdown_table(missing_top[['field', 'missing_count', 'missing_rate']], digits=4)}

## 8. 异常值与数据质量提示

{markdown_table(outliers, max_rows=20, digits=4)}

## 9. 建模建议

- 第一版基线可以用 `log_offline_oversubscription` 做标签，并同时保留 `board` 类别特征。
- 先建立三个不泄露的基线：板块滚动均值、年份/板块滚动均值、只用发行规模/价格/PE/战略配售/剔除比例的 Ridge 或树模型。
- `offline_subscription_total_10k`、`offline_allotment_accounts`、`offline_allotment_ratio_pct` 等字段预测力很强，但多数属于申购后/配售后信息，应作为标签或事后解释，不应进入正式预测输入。
- 科创板、创业板、主板中位水平差异较明显，后续必须分板块评估；统一模型要加入板块特征，且建议做板块残差校准。
- 下一步最关键不是换模型，而是补齐真正预测时点可见的初步询价字段。
"""
    return report


def make_figures(df: pd.DataFrame, missing: pd.DataFrame, corr_pre: pd.DataFrame) -> None:
    """Create lightweight chart files for reports and slides.

    SVG charts use only the Python standard library.  PNG charts are attempted
    only when matplotlib exists in the runtime.
    """
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_boxplot(df, fig_dir)
    svg_yearly_trend(df, fig_dir)
    svg_missing_bar(missing, fig_dir)
    svg_corr_bar(corr_pre, fig_dir)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        (fig_dir / "README.txt").write_text(
            "SVG figures were generated with the standard-library fallback.\n"
            f"PNG figures were skipped because matplotlib is unavailable: {exc}\n",
            encoding="utf-8",
        )
        print(f"Generated SVG figures. Skip PNG figures because matplotlib is unavailable: {exc}")
        return

    board_order = ["科创板", "创业板", "主板"]
    board_label = {"科创板": "STAR", "创业板": "ChiNext", "主板": "Main"}

    label_df = df[df["log_offline_oversubscription"].notna()].copy()
    data = [
        label_df.loc[label_df["board"] == board, "log_offline_oversubscription"].dropna()
        for board in board_order
        if (label_df["board"] == board).any()
    ]
    labels = [board_label[b] for b in board_order if (label_df["board"] == b).any()]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("log offline oversubscription")
    plt.title("Label Distribution by Board")
    plt.tight_layout()
    plt.savefig(fig_dir / "label_distribution_by_board.png", dpi=160)
    plt.close()

    trend = (
        label_df.groupby(["listing_year", "board"])["offline_oversubscription_ratio"]
        .median()
        .reset_index()
        .dropna()
    )
    plt.figure(figsize=(9, 5))
    for board in board_order:
        sub = trend[trend["board"] == board].sort_values("listing_year")
        if sub.empty:
            continue
        plt.plot(
            sub["listing_year"],
            sub["offline_oversubscription_ratio"],
            marker="o",
            label=board_label[board],
        )
    plt.ylabel("median offline oversubscription")
    plt.xlabel("listing year")
    plt.title("Median Offline Oversubscription by Year")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "yearly_median_oversubscription.png", dpi=160)
    plt.close()

    top_missing = missing[missing["missing_count"] > 0].head(12).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top_missing["field"], top_missing["missing_rate"] * 100)
    plt.xlabel("missing rate (%)")
    plt.title("Top Missing Fields")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_missing_fields.png", dpi=160)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        temp_dir = Path(td)
        frames = [read_source(spec, temp_dir) for spec in FILE_SPECS]
    raw = pd.concat(frames, ignore_index=True)
    df = add_features(raw)

    # Keep a stable, human-readable column order for downstream work.
    front_cols = [
        "security_code",
        "security_name",
        "board",
        "listing_date",
        "listing_year",
        "source_file",
        "sample_note",
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    csv_path = DATA_DIR / "ipo_offline_sample.csv"
    db_path = DATA_DIR / "ipo_offline.db"
    field_dict_path = OUT_DIR / "field_dictionary.csv"
    missing_path = OUT_DIR / "missing_by_field.csv"
    desc_path = OUT_DIR / "descriptive_stats.csv"
    corr_pre_path = OUT_DIR / "correlation_pre_subscription_like.csv"
    corr_all_path = OUT_DIR / "correlation_all_fields.csv"
    board_summary_path = OUT_DIR / "board_summary.csv"
    year_board_path = OUT_DIR / "year_board_counts.csv"
    report_path = OUT_DIR / "initial_analysis_report.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    db_df = df.copy()
    db_df["listing_date"] = db_df["listing_date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        db_df.to_sql("ipo_offline_sample", conn, if_exists="replace", index=False)
        pd.DataFrame(
            [
                {"category": key, "field": field}
                for key, fields in PREDICTION_TIME_WARNING.items()
                for field in fields
            ]
        ).to_sql("field_time_classification_draft", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ipo_board_year ON ipo_offline_sample(board, listing_year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ipo_listing_date ON ipo_offline_sample(listing_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ipo_security_code ON ipo_offline_sample(security_code)")

    field_dict = pd.DataFrame(
        [
            {"raw_field": raw_col, "clean_field": clean_col}
            for raw_col, clean_col in COLUMN_MAP.items()
        ]
    )
    field_dict.to_csv(field_dict_path, index=False, encoding="utf-8-sig")

    required_missing = pd.DataFrame(REQUIRED_PREDICTION_FIELDS)
    required_missing_path = OUT_DIR / "missing_required_prediction_fields.csv"
    required_missing.to_csv(required_missing_path, index=False, encoding="utf-8-sig")

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "field"})
    )
    missing["missing_rate"] = missing["missing_count"] / len(df)
    missing = missing.sort_values(["missing_rate", "missing_count"], ascending=False)
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "offline_oversubscription_ratio",
        "log_offline_oversubscription",
        "offline_allotment_ratio_pct",
        "implied_offline_lottery_rate_pct",
        "offline_oversubscription_ratio_before_clawback",
        "offline_subscription_total_10k",
        "offline_valid_quote_subscription_10k",
        "offline_issue_before_clawback_10k",
        "offline_issue_final_10k",
        "offer_price_yuan",
        "issue_amount_100m_yuan",
        "strategic_allocation_share_pct",
        "clawback_ratio_pct",
        "ipo_pe_diluted",
        "pe_vs_industry",
        "offline_allotment_accounts",
        "offline_inquiry_investors",
        "a_investor_lottery_rate_pct",
    ]
    desc = describe_numeric(df, numeric_cols)
    desc.to_csv(desc_path, index=False, encoding="utf-8-sig")

    board_summary = (
        df.groupby("board", dropna=False)
        .agg(
            sample_count=("security_code", "size"),
            label_count=("has_offline_label", "sum"),
            date_min=("listing_date", "min"),
            date_max=("listing_date", "max"),
            median_offline_oversubscription=("offline_oversubscription_ratio", "median"),
            mean_offline_oversubscription=("offline_oversubscription_ratio", "mean"),
            median_log_oversubscription=("log_offline_oversubscription", "median"),
            median_offline_allotment_ratio_pct=("offline_allotment_ratio_pct", "median"),
            median_offline_subscription_total_10k=("offline_subscription_total_10k", "median"),
            median_offline_accounts=("offline_allotment_accounts", "median"),
            median_inquiry_investors=("offline_inquiry_investors", "median"),
            median_issue_amount_100m_yuan=("issue_amount_100m_yuan", "median"),
            missing_label_count=("offline_oversubscription_ratio", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    board_summary["date_min"] = board_summary["date_min"].dt.strftime("%Y-%m-%d")
    board_summary["date_max"] = board_summary["date_max"].dt.strftime("%Y-%m-%d")
    board_summary.to_csv(board_summary_path, index=False, encoding="utf-8-sig")

    year_board = (
        df.pivot_table(
            index="listing_year",
            columns="board",
            values="security_code",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .sort_values("listing_year")
    )
    year_board.to_csv(year_board_path, index=False, encoding="utf-8-sig")

    label_desc = describe_numeric(
        df,
        [
            "offline_oversubscription_ratio",
            "log_offline_oversubscription",
            "offline_allotment_ratio_pct",
            "offline_oversubscription_ratio_before_clawback",
            "a_investor_lottery_rate_pct",
        ],
    )

    leakage_or_labels = set(PREDICTION_TIME_WARNING["label_or_post_subscription"]) | {
        "listing_date_raw",
        "listing_year",
        "has_offline_label",
        "offline_lottery_gap_pct_point",
        "implied_offline_lottery_rate_pct",
    }
    corr_pre = corr_table(df, "log_offline_oversubscription", leakage_or_labels)
    corr_all = corr_table(
        df,
        "log_offline_oversubscription",
        {"listing_date_raw", "listing_year", "has_offline_label"},
    )
    corr_pre.to_csv(corr_pre_path, index=False, encoding="utf-8-sig")
    corr_all.to_csv(corr_all_path, index=False, encoding="utf-8-sig")

    outlier_rows = []
    for col in [
        "offline_oversubscription_ratio",
        "offline_allotment_ratio_pct",
        "offline_subscription_total_10k",
        "offer_price_yuan",
        "issue_amount_100m_yuan",
        "ipo_pe_diluted",
        "clawback_ratio_pct",
    ]:
        s = df[col].dropna()
        if s.empty:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        outlier_rows.append(
            {
                "field": col,
                "p01": lo,
                "p99": hi,
                "below_p01_count": int((df[col] < lo).sum()),
                "above_p99_count": int((df[col] > hi).sum()),
                "min_security": df.loc[df[col].idxmin(), "security_name"] if df[col].notna().any() else "",
                "min": s.min(),
                "max_security": df.loc[df[col].idxmax(), "security_name"] if df[col].notna().any() else "",
                "max": s.max(),
            }
        )
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(OUT_DIR / "outlier_checks.csv", index=False, encoding="utf-8-sig")
    make_figures(df, missing, corr_pre)

    report = make_report(df, board_summary, year_board, missing, label_desc, corr_pre, corr_all, outliers)
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "inputs": [{"path": str(spec["path"]), "board": spec["board"]} for spec in FILE_SPECS],
        "outputs": {
            "csv": str(csv_path),
            "sqlite": str(db_path),
            "report": str(report_path),
            "field_dictionary": str(field_dict_path),
            "missing_required_prediction_fields": str(required_missing_path),
            "missing_by_field": str(missing_path),
            "descriptive_stats": str(desc_path),
            "board_summary": str(board_summary_path),
            "year_board_counts": str(year_board_path),
            "correlation_pre_subscription_like": str(corr_pre_path),
            "correlation_all_fields": str(corr_all_path),
            "outlier_checks": str(OUT_DIR / "outlier_checks.csv"),
            "figures_dir": str(OUT_DIR / "figures"),
        },
        "row_count": int(len(df)),
        "label_count": int(df["has_offline_label"].sum()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
