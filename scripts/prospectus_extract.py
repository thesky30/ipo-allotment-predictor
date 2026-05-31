"""从招股书 PDF 补全发行公告拿不到的 T-6 财务/估值字段。

招股书几百页：先逐页取文本（无 LLM 成本），按关键词给页面打分，只把
得分最高的若干页（封顶页数+字数）喂给 LLM，最大限度省 token。
"""
from __future__ import annotations

import io
from typing import Callable

import llm_client
from pdf_extract import ExtractResult

PROSPECTUS_FIELD_SCHEMA: dict[str, str] = {
    "latest_revenue_100m_yuan": "最近一个完整会计年度的营业收入（亿元，纯数字）",
    "revenue_cagr_3y_pct": "最近三年营业收入的复合年增长率（%）；无现成数字则用最近三年营收推算",
    "comparable_pe_avg_ex_nonrecurring": "招股书披露的同行业可比上市公司『扣非后市盈率』平均值",
    "expected_fundraising_100m_yuan": "本次发行『拟』募集资金总额（亿元）；不是发行后最终募资",
}

_ANCHORS: tuple[str, ...] = (
    "营业收入", "主要财务数据", "扣除非经常性损益", "归属于母公司",
    "募集资金", "拟募集", "募集资金运用",
    "可比公司", "同行业可比上市公司",
)

SYSTEM_PROMPT = (
    "你是严谨的招股书信息抽取助手。下面给你的是从某 A 股招股说明书中"
    "『定位到的若干相关页』（非全文）。请抽取以下字段并只输出一个 JSON 对象"
    "（不要解释）。数值输出纯数字（去单位、去千分位逗号），找不到的输出 null。\n"
    + "\n".join(f"- {k}: {v}" for k, v in PROSPECTUS_FIELD_SCHEMA.items())
)

MAX_PAGES = 8
MAX_CHARS = 16000


def extract_pages(pdf_bytes: bytes) -> list[str]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def locate_relevant_pages(
    pages: list[str],
    max_pages: int = MAX_PAGES,
    max_chars: int = MAX_CHARS,
) -> tuple[str, list[int]]:
    """按锚点关键词给每页打分，挑得分最高的页，按文档顺序拼接。"""
    scored = []
    for i, txt in enumerate(pages):
        score = sum(txt.count(kw) for kw in _ANCHORS)
        if score:
            scored.append((score, i))

    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[int] = []
    total = 0
    for _score, i in scored:
        if len(chosen) >= max_pages or total >= max_chars:
            break
        chosen.append(i)
        total += len(pages[i])

    chosen.sort()
    text = "\n\n".join(pages[i] for i in chosen)[:max_chars]
    return text, chosen


def extract_prospectus_fields(
    pdf_bytes: bytes,
    *,
    extract_json: Callable[[str, str], dict] | None = None,
    pages: list[str] | None = None,
) -> ExtractResult:
    if pages is None:
        pages = extract_pages(pdf_bytes)

    text, used = locate_relevant_pages(pages)
    warnings: list[str] = []
    if not text:
        warnings.append("未定位到财务/募资/可比公司章节，请改为手动输入这些字段。")
        return ExtractResult(fields={}, text_chars=0, warnings=warnings)

    fn = extract_json or llm_client.extract_json
    raw = fn(SYSTEM_PROMPT, text)
    fields = {
        k: raw.get(k)
        for k in PROSPECTUS_FIELD_SCHEMA
        if raw.get(k) not in (None, "")
    }
    warnings.append(f"招股书仅取定位到的页 {used}（共 {len(text)} 字送抽取，省 token）。")
    return ExtractResult(fields=fields, text_chars=len(text), warnings=warnings)
