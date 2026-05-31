"""从招股书 PDF 补全发行公告拿不到的 T-6 财务/估值字段。

招股书几百页：先逐页取文本（无 LLM 成本），按关键词给页面打分，只把
得分最高的若干页（封顶页数+字数）喂给 LLM，最大限度省 token。
"""
from __future__ import annotations

import io
import re
from typing import Callable

import llm_client
from pdf_extract import ExtractResult
from reference_data import normalize_sw_level1_industry_code, sw_level1_industry_name

PROSPECTUS_FIELD_SCHEMA: dict[str, str] = {
    "latest_revenue_100m_yuan": "最近一个完整会计年度的营业收入（亿元，纯数字）",
    "revenue_cagr_3y_pct": "最近三年营业收入的复合年增长率（%）；无现成数字则用最近三年营收推算",
    "sw_level1_industry_code": "申万一级行业代码；仅填 16 位 Wind 风格申万一级代码；不要填 C39 这类证监会行业代码，找不到则输出 null",
    "sw_level1_industry_name": "申万一级行业名称，例如电子、通信、机械设备；不要输出证监会行业大类名称，找不到输出 null",
    "comparable_company_names": "招股书披露的同行业可比上市公司名称列表（数组）；只取上市公司，找不到输出空数组",
    "expected_fundraising_100m_yuan": "本次发行『拟』募集资金总额（亿元）；不是发行后最终募资",
}

_ANCHOR_GROUPS: dict[str, tuple[str, ...]] = {
    "financial": ("营业收入", "主要财务数据", "扣除非经常性损益", "归属于母公司"),
    "fundraising": ("募集资金", "拟募集", "募集资金运用"),
    "peer": (
        "可比公司", "可比上市公司", "同行业可比上市公司", "同行业上市公司",
        "主要竞争对手", "竞争对手", "证券简称",
    ),
    "industry": ("所属行业", "行业分类", "申万一级", "申万一级行业", "所属申万"),
}
_ANCHORS: tuple[str, ...] = tuple(kw for group in _ANCHOR_GROUPS.values() for kw in group)

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
    """按锚点关键词给每页打分，挑得分最高的页，按文档顺序拼接。

    招股书财务页可能反复出现“营业收入”，全局 top-N 容易挤掉可比公司
    或申万行业页；先按类别各保留一页，再用总分补齐。
    """
    scored: list[tuple[int, int]] = []
    by_group: dict[str, list[tuple[int, int]]] = {}
    for i, txt in enumerate(pages):
        score = sum(txt.count(kw) for kw in _ANCHORS)
        if score:
            scored.append((score, i))
        for group, anchors in _ANCHOR_GROUPS.items():
            group_score = sum(txt.count(kw) for kw in anchors)
            if group_score:
                by_group.setdefault(group, []).append((group_score, i))

    chosen: list[int] = []
    total = 0

    def add_page(i: int) -> None:
        nonlocal total
        if 0 <= i < len(pages) and i not in chosen and len(chosen) < max_pages and total < max_chars:
            chosen.append(i)
            total += len(pages[i])

    for group in ("financial", "fundraising", "peer", "industry"):
        group_scores = sorted(by_group.get(group, []), key=lambda t: (-t[0], t[1]))
        if group_scores:
            idx = group_scores[0][1]
            add_page(idx)
            if group in ("peer", "industry"):
                add_page(idx + 1)

    scored.sort(key=lambda t: (-t[0], t[1]))
    for _score, i in scored:
        if len(chosen) >= max_pages or total >= max_chars:
            break
        add_page(i)

    chosen.sort()
    text = "\n\n".join(pages[i] for i in chosen)[:max_chars]
    return text, chosen


def _clean_sw_industry_name(value: object) -> str:
    text = str(value or "").strip()
    return text.replace("(申万)", "").replace("（申万）", "").strip()


def _normalize_peer_names(value: object) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        parts = re.split(r"[、,，;；\n/\s]+", str(item))
        for part in parts:
            name = re.sub(r"[（(][0-9A-Za-z.\-]+[）)]", "", part).strip()
            if re.fullmatch(r"[0-9]{6}(?:\.[A-Za-z]{2})?", name):
                continue
            for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团", "股份"):
                name = name.replace(suffix, "")
            name = name.replace(" ", "")
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


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
    if "sw_level1_industry_name" in fields:
        fields["sw_level1_industry_name"] = _clean_sw_industry_name(fields["sw_level1_industry_name"])
    raw_sw_code = fields.get("sw_level1_industry_code")
    raw_sw_name = fields.get("sw_level1_industry_name")
    code = normalize_sw_level1_industry_code(raw_sw_code, raw_sw_name)
    if code:
        fields["sw_level1_industry_code"] = code
        fields["sw_level1_industry_name"] = sw_level1_industry_name(code)
    elif raw_sw_code:
        fields.pop("sw_level1_industry_code", None)
        warnings.append(f"忽略无法映射为申万一级行业的代码：{raw_sw_code}。")
    if "comparable_company_names" in fields:
        fields["comparable_company_names"] = _normalize_peer_names(fields["comparable_company_names"])
    warnings.append(f"招股书仅取定位到的页 {used}（共 {len(text)} 字送抽取，省 token）。")
    return ExtractResult(fields=fields, text_chars=len(text), warnings=warnings)
