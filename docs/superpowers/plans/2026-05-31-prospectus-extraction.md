# 招股书提取（补全 T-6 财务/估值字段）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** 让用户在「发行安排及初步询价公告」之外，可选地上传**招股书 PDF**，自动补全发行公告里拿不到的 T-6 字段（营收、3 年营收 CAGR、可比公司 PE、拟募资额），并 **只把定位到的相关页喂 LLM** 以省 token。

**Architecture:** 招股书几百页，绝不整本喂 LLM。先 pdfplumber 逐页取文本（无 LLM 成本），按「财务/募资/可比公司」关键词给每页打分，挑出得分最高的若干页（封顶页数 + 封顶字数），按文档顺序拼接后**只把这几页**交给 `llm_client.extract_json`，按一个**只含招股书字段**的窄 schema 抽取。结果并入现有 `pdf_prefill`（发行公告填结构字段，招股书填财务/估值字段）。

**Tech Stack:** Python, pdfplumber, 既有 `llm_client`（provider-agnostic），pytest（LLM 注入 + 页面定位用合成数据，确定性）。

**Spec/前置:** 建立在 Phase 2 之上（`scripts/pdf_extract.py`、`scripts/llm_client.py`、`app.py` 手动 tab 的 `pdf_prefill` 机制）。背景与字段来源见 `AGENTS.md`「冷启动 T-6 字段取数来源」。

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `scripts/prospectus_extract.py` | `extract_pages`, `locate_relevant_pages`(打分+封顶), `PROSPECTUS_FIELD_SCHEMA`, `extract_prospectus_fields`(注入 LLM)。复用 `pdf_extract.ExtractResult`。 | Create |
| `app.py` | 手动 tab 第二个 `file_uploader`（招股书，可选）；抽取后**并入** `st.session_state["pdf_prefill"]`（不覆盖发行公告已填的结构字段）。 | Modify |
| `tests/test_prospectus_extract.py` | 页面打分定位（合成页）+ 字段抽取（注入 LLM）+ 真实 PDF `skipif`。 | Create |

招股书 schema（**只**补发行公告拿不到的字段；结构字段仍由发行公告负责）：
`latest_revenue_100m_yuan, revenue_cagr_3y_pct, comparable_pe_avg_ex_nonrecurring, expected_fundraising_100m_yuan`。

---

## Task 1: prospectus_extract — 页面定位 + 窄字段抽取

**Files:** Create `scripts/prospectus_extract.py`; Test `tests/test_prospectus_extract.py`

- [ ] **Step 1: 写失败测试** `tests/test_prospectus_extract.py`：
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import prospectus_extract as pe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_locate_picks_anchor_pages_and_caps():
    pages = [
        "目录 第一节 释义 ……无关内容" * 50,                 # 0: 无锚点
        "第八节 财务会计信息 营业收入 营业收入 扣除非经常性损益" * 5,  # 1: 财务锚点多
        "无关页" * 50,                                       # 2: 无锚点
        "募集资金运用 本次拟募集资金 可比公司 同行业可比上市公司",      # 3: 募资+可比
    ]
    text, used = pe.locate_relevant_pages(pages, max_pages=2, max_chars=100000)
    assert used == [1, 3]                  # 选中有锚点的两页，按文档顺序
    assert "营业收入" in text and "募集资金" in text
    assert 0 not in used and 2 not in used


def test_locate_respects_char_cap():
    pages = ["营业收入 " * 1000, "募集资金 " * 1000]
    text, used = pe.locate_relevant_pages(pages, max_pages=8, max_chars=500)
    assert len(text) <= 500


def test_extract_prospectus_fields_with_injected_llm():
    canned = {
        "latest_revenue_100m_yuan": 7.2,
        "revenue_cagr_3y_pct": 25.0,
        "comparable_pe_avg_ex_nonrecurring": 41.3,
        "expected_fundraising_100m_yuan": 11.0,
        "board": "科创板",          # 非招股书 schema 键 → 丢弃
    }
    res = pe.extract_prospectus_fields(
        b"", extract_json=lambda system, user: canned,
        pages=["营业收入 募集资金 可比公司"],
    )
    assert res.fields["latest_revenue_100m_yuan"] == 7.2
    assert res.fields["comparable_pe_avg_ex_nonrecurring"] == 41.3
    assert "board" not in res.fields


@pytest.mark.skipif(
    not list(FIXTURES.glob("*招股*.pdf")),
    reason="把真实招股书 PDF（文件名含『招股』）放进 tests/fixtures/ 才跑此测试",
)
def test_extract_text_pages_real_prospectus():
    pdf = next(FIXTURES.glob("*招股*.pdf"))
    pages = pe.extract_pages(pdf.read_bytes())
    assert len(pages) > 20            # 招股书通常很多页
    text, used = pe.locate_relevant_pages(pages)
    assert used and len(text) > 200
```

- [ ] **Step 2: 跑，确认失败** `python3 -m pytest tests/test_prospectus_extract.py -v` → FAIL（无模块）。

- [ ] **Step 3: 实现** `scripts/prospectus_extract.py`：
```python
"""从招股书 PDF 补全发行公告拿不到的 T-6 财务/估值字段。

招股书几百页：先逐页取文本（无 LLM 成本），按关键词给页面打分，只把
得分最高的若干页（封顶页数+字数）喂给 LLM，最大限度省 token。"""
from __future__ import annotations

import io
from typing import Any, Callable

import llm_client
from pdf_extract import ExtractResult  # 复用同一结果类型

# 招股书才有、发行公告没有的 T-6 字段（窄 schema）
PROSPECTUS_FIELD_SCHEMA: dict[str, str] = {
    "latest_revenue_100m_yuan": "最近一个完整会计年度的营业收入（亿元，纯数字）",
    "revenue_cagr_3y_pct": "最近三年营业收入的复合年增长率（%）；无现成数字则用最近三年营收推算",
    "comparable_pe_avg_ex_nonrecurring": "招股书披露的同行业可比上市公司『扣非后市盈率』平均值",
    "expected_fundraising_100m_yuan": "本次发行『拟』募集资金总额（亿元）；不是发行后最终募资",
}

# 定位相关章节的锚点关键词（财务 / 募资 / 可比公司）
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
    pages: list[str], max_pages: int = MAX_PAGES, max_chars: int = MAX_CHARS,
) -> tuple[str, list[int]]:
    """按锚点关键词给每页打分，挑得分最高的页（封顶页数/字数），按文档顺序拼接。"""
    scored = []
    for i, txt in enumerate(pages):
        score = sum(txt.count(kw) for kw in _ANCHORS)
        if score:
            scored.append((score, i))
    scored.sort(key=lambda t: (-t[0], t[1]))          # 分高优先，平分按页序
    chosen: list[int] = []
    total = 0
    for _score, i in scored:
        if len(chosen) >= max_pages or total >= max_chars:
            break
        chosen.append(i)
        total += len(pages[i])
    chosen.sort()                                     # 还原文档顺序
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
    fields = {k: raw.get(k) for k in PROSPECTUS_FIELD_SCHEMA if raw.get(k) not in (None, "")}
    warnings.append(
        f"招股书仅取定位到的页 {used}（共 {len(text)} 字送抽取，省 token）。"
    )
    return ExtractResult(fields=fields, text_chars=len(text), warnings=warnings)
```

- [ ] **Step 4: 跑，确认通过** `python3 -m pytest tests/test_prospectus_extract.py -v` → 3 passed, 1 skipped。再 `python3 -m pytest tests/ -q` 确认全绿。

- [ ] **Step 5: 提交**
```bash
git add scripts/prospectus_extract.py tests/test_prospectus_extract.py
git commit -m "feat: prospectus_extract — locate relevant pages, extract T-6 financial/valuation fields"
```

---

## Task 2: app.py — 第二个上传器（招股书，可选），并入 prefill

**Files:** Modify `app.py`（手动 tab，紧接发行公告上传器之后）

- [ ] **Step 1: 在发行公告上传器之后、`_pf = ...` 之前，加招股书上传器**
```python
    up2 = st.file_uploader("（可选）上传招股书 PDF 补充财务/估值字段（营收/CAGR/可比PE/拟募资）", type="pdf", key="prospectus_pdf")
    if up2 is not None and st.button("识别招股书字段", key="run_prospectus_extract"):
        import prospectus_extract, llm_client
        if not llm_client.is_configured():
            st.error("未配置 LLM（见 .env / Secrets：LLM_API_KEY），无法识别招股书。")
        else:
            with st.spinner("定位章节并识别招股书中…"):
                try:
                    res2 = prospectus_extract.extract_prospectus_fields(up2.read())
                    merged = dict(st.session_state.get("pdf_prefill", {}))
                    merged.update(res2.fields)          # 招股书补财务字段；不动发行公告已填的结构字段
                    st.session_state["pdf_prefill"] = merged
                    for w in res2.warnings:
                        st.info(w)
                    st.success(f"招股书补充了 {len(res2.fields)} 个字段，请在下方核对。")
                except Exception as e:
                    st.error(f"招股书识别失败：{e}。请手动输入这些字段。")
```
（`merged.update(res2.fields)`：招股书 schema 只含财务/估值字段，与发行公告的结构字段不重叠，所以是纯补充。）

- [ ] **Step 2: 验证**
1. `python3 -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"`
2. 启动：`timeout 25 streamlit run app.py --server.headless true --server.port 8606 >/tmp/st7.log 2>&1; grep -iE "error|traceback" /tmp/st7.log && echo "STARTUP ERROR" || echo "clean"`
3. 合并逻辑 headless：
```bash
PYTHONPATH=scripts python3 -c "
import prospectus_extract as pe
r=pe.extract_prospectus_fields(b'', extract_json=lambda s,u:{'latest_revenue_100m_yuan':7.2,'expected_fundraising_100m_yuan':11.0}, pages=['营业收入 募集资金'])
print('prospectus fields:', r.fields)"
```
4. `python3 -m pytest tests/ -q` 全绿。

- [ ] **Step 3: 提交**
```bash
git add app.py
git commit -m "feat: optional prospectus PDF upload merges financial/valuation fields into prefill"
```

---

## Task 3: 文档

- [ ] **Step 1:** 在 `AGENTS.md`「冷启动 T-6 字段取数来源」节把招股书方案标为已实现（`scripts/prospectus_extract.py`，页面定位省 token）。在 `README.md` 的 Phase 2 用法旁补一句「可另传招股书补全财务字段」。提交。

---

## Self-Review

- **省 token（用户核心诉求）**：`locate_relevant_pages` 按锚点打分 + `max_pages`/`max_chars` 双封顶，只把相关页喂 LLM —— Task 1 实现 + `test_locate_*` 验证。✓
- **字段边界**：招股书 schema 只含发行公告拿不到的 4 个财务/估值字段，与发行公告 schema 不重叠，`merged.update` 纯补充。✓
- **provider-agnostic / 人工确认**：复用 `llm_client` 与既有 `pdf_prefill` 表单（仍需点「预测」）。✓
- **测试确定性**：页面定位用合成页、抽取注入 LLM；真实招股书 `skipif`。✓
- **Placeholder 扫描**：无；真实 PDF 的 skip 是带明确条件的有意跳过。
- **类型一致**：`extract_prospectus_fields(..., extract_json, pages) -> ExtractResult`（复用 `pdf_extract.ExtractResult`）；`locate_relevant_pages(pages, max_pages, max_chars) -> (text, used)` 在测试与实现一致。

## Done criteria
- `pytest tests/ -q` 全绿（含招股书 3 passed + 1 skipped）。
- 网页可分别/同时上传发行公告 + 招股书；招股书只喂定位到的若干页；财务字段并入表单、人工核对后预测。
- 放一份真实招股书进 `tests/fixtures/`（文件名含「招股」）即可跑真实页定位测试。
