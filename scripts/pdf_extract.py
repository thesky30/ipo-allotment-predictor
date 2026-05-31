"""Extract询价前 raw fields from a cninfo「发行安排及初步询价公告」PDF.

Two stages: PDF bytes -> text (pdfplumber); text -> structured fields (LLM).
The LLM call is injectable so tests are deterministic without network."""
from __future__ import annotations

import io
from dataclasses import dataclass, field as _field
from typing import Any, Callable

import llm_client

# Extraction contract: key -> 中文描述 (also drives the LLM prompt). These keys
# are exactly the raw inputs feature_assembly.assemble_t6 consumes.
FIELD_SCHEMA: dict[str, str] = {
    "board": "板块（取值之一：科创板/创业板/主板/北交所）",
    "subscription_deadline_date": "网下申购截止日（格式 YYYY-MM-DD）",
    "lead_underwriter": '主承销商全称（法人全称，如“中信证券股份有限公司”）',
    "sw_level1_industry_code": "申万一级行业代码（若公告中无，留空）",
    "total_issue_shares_10k": "本次发行总股数（万股，纯数字）",
    "offline_issue_before_clawback_10k": "网下初始发行数量（回拨前，万股）",
    "online_issue_before_clawback_10k": "网上初始发行数量（回拨前，万股）",
    "strategic_allocation_10k": "战略配售股数（万股）",
    "subscription_upper_limit_10k": "网下单个配售对象申购上限（万股）",
    "subscription_lower_limit_10k": "网下单个配售对象申购下限（万股）",
    "subscription_step_10k": "网下申购步长（万股）",
    "offline_market_value_threshold_10k_yuan": "网下投资者市值门槛（万元）",
    "industry_pe_at_ipo": "发行时所属行业最近一个月平均静态市盈率",
    "comparable_pe_avg_ex_nonrecurring": "可比公司扣非后市盈率平均值",
    "expected_fundraising_100m_yuan": "预计募集资金总额（亿元）",
    "latest_revenue_100m_yuan": "最近一年营业收入（亿元）",
    "revenue_cagr_3y_pct": "最近三年营业收入复合增长率（%）",
}

SYSTEM_PROMPT = (
    "你是严谨的金融公告信息抽取助手。从给定的 A 股新股『发行安排及初步询价公告』文本中，"
    "抽取以下字段并只输出一个 JSON 对象（不要任何解释文字）。"
    "数值字段输出纯数字（去掉单位、千分位逗号），找不到的字段输出 null。\n"
    "字段说明：\n"
    + "\n".join(f"- {k}: {v}" for k, v in FIELD_SCHEMA.items())
)

MAX_CHARS = 20000  # cap text sent to the model


@dataclass
class ExtractResult:
    fields: dict[str, Any]
    text_chars: int
    warnings: list[str] = _field(default_factory=list)


def extract_text(pdf_bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def extract_ipo_fields(
    pdf_bytes: bytes,
    *,
    extract_json: Callable[[str, str], dict] | None = None,
    text: str | None = None,
) -> ExtractResult:
    """Return only the FIELD_SCHEMA keys with non-null values."""
    if text is None:
        text = extract_text(pdf_bytes)
    warnings: list[str] = []
    if len(text) > MAX_CHARS:
        warnings.append(f"公告文本较长（{len(text)} 字），仅取前 {MAX_CHARS} 字送抽取。")
    fn = extract_json or llm_client.extract_json
    raw = fn(SYSTEM_PROMPT, text[:MAX_CHARS])
    fields = {k: raw.get(k) for k in FIELD_SCHEMA if raw.get(k) not in (None, "")}
    if not fields:
        warnings.append("未从公告中抽取到任何字段，请改为手动输入。")
    return ExtractResult(fields=fields, text_chars=len(text), warnings=warnings)
