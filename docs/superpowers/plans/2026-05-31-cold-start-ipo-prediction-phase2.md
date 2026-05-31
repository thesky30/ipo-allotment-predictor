# Cold-Start IPO Prediction — Phase 2 (PDF → LLM Extraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user drop a 巨潮(cninfo)「发行安排及初步询价公告」PDF into the web app, auto-extract the询价前 raw fields with a **provider-agnostic LLM**, and prefill the existing manual form for human confirmation before predicting.

**Architecture:** Two new pure modules — `llm_client` (OpenAI-compatible chat client, fully configured by env vars so any provider works) and `pdf_extract` (PDF→text via pdfplumber, then text→structured fields via an *injectable* LLM call). `app.py` gains a `file_uploader` above the manual form; on upload it extracts fields into `st.session_state`, and the form widgets read those as editable defaults. Extraction never auto-predicts — the user reviews/edits, then submits the same `predict_new_ipo` path from Phase 1.

**Tech Stack:** Python, pdfplumber, `openai` SDK (used only as an OpenAI-compatible transport; `base_url`/`api_key`/`model` from env), pytest (LLM + PDF mocked for deterministic CI), Streamlit.

**Spec:** `docs/superpowers/specs/2026-05-31-cold-start-ipo-prediction-design.md` §6 (PDF抽取与人工确认), §2 decision 5 (provider-agnostic), §3 modules. Builds on Phase 1 (`feature_assembly`, `predict_new_ipo`, the manual tab).

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `scripts/llm_client.py` | OpenAI-compatible chat client; `is_configured()`, `extract_json(system, user, model=None)`. Config: `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`. No provider hardcoded. | Create |
| `scripts/pdf_extract.py` | `FIELD_SCHEMA`, `SYSTEM_PROMPT`, `extract_text(bytes)`, `extract_ipo_fields(pdf_bytes, *, extract_json=None, text=None) -> ExtractResult`. LLM call injectable for tests. | Create |
| `app.py` | `tab_manual`: add PDF `file_uploader` above the form; on upload populate `st.session_state["pdf_prefill"]`; form widgets default from it; "已识别请核对" banner + warnings. | Modify (`tab_manual` ~355-420) |
| `requirements.txt` | add `pdfplumber>=0.11`, `openai>=1.30` | Modify |
| `tests/test_llm_client.py` | `is_configured`, `extract_json` with a monkeypatched client. | Create |
| `tests/test_pdf_extract.py` | field extraction with injected `extract_json` + text; real-PDF path `skipif` no fixture. | Create |
| `tests/fixtures/` | optional real `公告.pdf` drop-in for the skipped integration test. | Create dir |

Field contract (the ~16 raw keys `feature_assembly.assemble_t6` consumes, defined once in `pdf_extract.FIELD_SCHEMA`):
`board, subscription_deadline_date, lead_underwriter, sw_level1_industry_code, total_issue_shares_10k, offline_issue_before_clawback_10k, online_issue_before_clawback_10k, strategic_allocation_10k, subscription_upper_limit_10k, subscription_lower_limit_10k, subscription_step_10k, offline_market_value_threshold_10k_yuan, industry_pe_at_ipo, comparable_pe_avg_ex_nonrecurring, expected_fundraising_100m_yuan, latest_revenue_100m_yuan, revenue_cagr_3y_pct`.

---

## Task 1: Dependencies

**Files:** Modify `requirements.txt`

- [ ] **Step 1: Add the two libraries**

Append to `requirements.txt`:
```text
pdfplumber>=0.11
openai>=1.30
```

- [ ] **Step 2: Install and verify import**

Run: `pip install -r requirements.txt && python3 -c "import pdfplumber, openai; print('deps ok', pdfplumber.__version__, openai.__version__)"`
Expected: prints `deps ok <ver> <ver>`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add pdfplumber + openai (provider-agnostic LLM transport) for PDF extraction"
```

---

## Task 2: llm_client — provider-agnostic OpenAI-compatible JSON extraction

**Files:** Create `scripts/llm_client.py`; Test `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_client.py`:
```python
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import llm_client


def test_is_configured(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert llm_client.is_configured() is False
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert llm_client.is_configured() is True


def test_extract_json_parses_model_output(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    class _Msg:    content = '{"board": "科创板", "industry_pe_at_ipo": 38.5}'
    class _Choice: message = _Msg()
    class _Resp:   choices = [_Choice()]

    class _FakeClient:
        def __init__(self, *a, **k): self.chat = self
        @property
        def completions(self): return self
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["response_format"] == {"type": "json_object"}
            return _Resp()

    monkeypatch.setattr(llm_client, "OpenAI", _FakeClient)
    out = llm_client.extract_json("sys", "user text")
    assert out == {"board": "科创板", "industry_pe_at_ipo": 38.5}


def test_extract_json_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMConfigError):
        llm_client.extract_json("sys", "user")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_client'`.

- [ ] **Step 3: Implement**

`scripts/llm_client.py`:
```python
"""Provider-agnostic LLM client over the OpenAI-compatible chat API.

Configured ENTIRELY by environment variables so any OpenAI-compatible provider
works (DeepSeek / Qwen / Kimi / Zhipu / OpenAI / a self-hosted gateway):
  LLM_API_KEY   (required)
  LLM_BASE_URL  (optional; provider's base url, e.g. https://api.deepseek.com/v1)
  LLM_MODEL     (optional; default 'gpt-4o-mini')
No provider is hard-coded. No network at import time."""
from __future__ import annotations

import json
import os

from openai import OpenAI


class LLMConfigError(RuntimeError):
    """Raised when the LLM is called without an API key configured."""


def is_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _client() -> OpenAI:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise LLMConfigError(
            "LLM_API_KEY 未配置。请设置 LLM_API_KEY（及可选 LLM_BASE_URL / LLM_MODEL）。"
        )
    base = os.environ.get("LLM_BASE_URL") or None
    return OpenAI(api_key=key, base_url=base)


def extract_json(system_prompt: str, user_text: str, model: str | None = None) -> dict:
    """Ask the model to return a single JSON object; parse and return it."""
    client = _client()
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_llm_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_client.py tests/test_llm_client.py
git commit -m "feat: provider-agnostic OpenAI-compatible llm_client.extract_json"
```

---

## Task 3: pdf_extract — PDF → fields

**Files:** Create `scripts/pdf_extract.py`; Test `tests/test_pdf_extract.py`; Create dir `tests/fixtures/`

- [ ] **Step 1: Write the failing test**

`tests/test_pdf_extract.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import pdf_extract

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_extract_ipo_fields_with_injected_llm():
    canned = {
        "board": "科创板",
        "subscription_deadline_date": "2025-03-18",
        "lead_underwriter": "中信证券股份有限公司",
        "total_issue_shares_10k": 4000,
        "industry_pe_at_ipo": 38.5,
        "not_a_field": "ignored",   # extra keys dropped
        "expected_fundraising_100m_yuan": None,  # null dropped
    }
    res = pdf_extract.extract_ipo_fields(
        b"", extract_json=lambda system, user: canned, text="一些公告文本"
    )
    assert res.fields["board"] == "科创板"
    assert res.fields["total_issue_shares_10k"] == 4000
    assert "not_a_field" not in res.fields          # only schema keys kept
    assert "expected_fundraising_100m_yuan" not in res.fields  # null pruned
    assert res.text_chars == len("一些公告文本")


def test_schema_keys_match_assemble_contract():
    # The extraction schema must be a subset of what assemble_t6 accepts.
    import feature_assembly  # noqa: F401
    assert "subscription_deadline_date" in pdf_extract.FIELD_SCHEMA
    assert "lead_underwriter" in pdf_extract.FIELD_SCHEMA
    assert "sw_level1_industry_code" in pdf_extract.FIELD_SCHEMA


@pytest.mark.skipif(
    not list(FIXTURES.glob("*.pdf")),
    reason="drop a real 巨潮 公告 PDF into tests/fixtures/ to run the real-PDF text test",
)
def test_extract_text_reads_real_pdf():
    pdf = next(FIXTURES.glob("*.pdf"))
    text = pdf_extract.extract_text(pdf.read_bytes())
    assert len(text) > 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_pdf_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_extract'`.

- [ ] **Step 3: Implement**

`scripts/pdf_extract.py`:
```python
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
    "lead_underwriter": "主承销商全称（法人全称，如“中信证券股份有限公司”）",
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_pdf_extract.py -v`
Expected: PASS (2 passed, 1 skipped — the real-PDF test).

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/fixtures && touch tests/fixtures/.gitkeep
git add scripts/pdf_extract.py tests/test_pdf_extract.py tests/fixtures/.gitkeep
git commit -m "feat: pdf_extract — cninfo announcement PDF -> raw IPO fields via injectable LLM"
```

---

## Task 4: app.py — PDF upload + prefill the manual form (human-confirm)

**Files:** Modify `app.py` (`tab_manual`)

- [ ] **Step 1: Add the uploader + extraction above the form**

In `app.py`, inside `with tab_manual:` and BEFORE `with st.form("manual_form"):`, insert:
```python
    st.markdown("##### 可选：上传巨潮『发行安排及初步询价公告』PDF 自动识别")
    up = st.file_uploader("上传 PDF（识别后回填下方表单，请务必人工核对）", type="pdf")
    if up is not None and st.button("识别 PDF 字段", key="run_pdf_extract"):
        import pdf_extract, llm_client
        if not llm_client.is_configured():
            st.error("未配置抽取用 LLM。请在部署环境设置 LLM_API_KEY（及可选 LLM_BASE_URL / LLM_MODEL）后重试，或直接手动输入。")
        else:
            with st.spinner("识别中…"):
                try:
                    res = pdf_extract.extract_ipo_fields(up.read())
                    st.session_state["pdf_prefill"] = res.fields
                    for w in res.warnings:
                        st.warning(w)
                    st.success(f"已识别 {len(res.fields)} 个字段，已回填下方表单，请核对后再预测。")
                except Exception as e:
                    st.error(f"识别失败：{e}。请改为手动输入。")
    _pf = st.session_state.get("pdf_prefill", {})
```

- [ ] **Step 2: Make the form widgets default from `_pf`**

In the same `tab_manual` form, set each widget's default from `_pf` (extracted values), keeping them editable. Example replacements (apply the same pattern to every field):
```python
        board_sel = c1.selectbox(
            "板块 *", ["科创板", "创业板", "主板", "北交所"],
            index=(["科创板", "创业板", "主板", "北交所"].index(_pf["board"])
                   if _pf.get("board") in ["科创板", "创业板", "主板", "北交所"] else 0),
        )
        total_shares = c5.number_input(
            "发行总股数（万股）", min_value=0.0,
            value=float(_pf.get("total_issue_shares_10k") or 0.0),
        )
        industry_pe = c13.number_input(
            "行业PE", min_value=0.0, step=0.1,
            value=float(_pf.get("industry_pe_at_ipo") or 0.0),
        )
```
Do this for ALL numeric inputs (`offline_issue_before_clawback_10k`, `online_issue_before_clawback_10k`, `subscription_upper_limit_10k`, `subscription_lower_limit_10k`, `subscription_step_10k`, `offline_market_value_threshold_10k_yuan`, `expected_fundraising_100m_yuan`, `latest_revenue_100m_yuan`, `revenue_cagr_3y_pct`, `strategic_allocation` if present) using `value=float(_pf.get(<key>) or 0.0)`. For the 申购截止日 date input use:
```python
        import datetime as _dt
        _dd = _pf.get("subscription_deadline_date")
        try:
            _dd = _dt.date.fromisoformat(_dd) if _dd else None
        except (TypeError, ValueError):
            _dd = None
        deadline_date = c11.date_input("申购截止日 *", value=_dd)
```
For the 主承销商 / 申万行业 selectboxes (sourced from DB lists), set `index` to the extracted value's position if present in the list, else 0 (the "（未知/其他）" option). If the extracted `lead_underwriter` / `sw_level1_industry_code` is not in the DB list, leave it at the "（未知/其他）" default and the existing missing-prior warning will fire.

- [ ] **Step 3: Verify (UI checks)**

1. Syntax: `python3 -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"`
2. App starts clean: `timeout 25 streamlit run app.py --server.headless true --server.port 8602 > /tmp/st3.log 2>&1; grep -iE "error|traceback" /tmp/st3.log && echo "STARTUP ERROR" || echo "clean"`
3. Headless prefill logic (no real PDF/LLM): 
```bash
PYTHONPATH=scripts python3 -c "
import pdf_extract
res = pdf_extract.extract_ipo_fields(b'', extract_json=lambda s,u: {'board':'科创板','industry_pe_at_ipo':38.5,'total_issue_shares_10k':4000}, text='x')
print('prefill fields:', res.fields)
"
```
4. `python3 -m pytest tests/ -q` still all pass.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: manual tab — upload cninfo PDF to auto-prefill the form (human-confirm before predict)"
```

---

## Task 5: Docs sync

**Files:** Modify `AGENTS.md`, `README.md`, and the spec status.

- [ ] **Step 1: Mark Phase 2 implemented**

In `AGENTS.md` under the「未入库新股的冷启动预测」section, change the Phase 2 backlog line to done and note the env config:
```text
- [x] Phase 2：巨潮 PDF → provider-agnostic LLM 抽取 → 回填可编辑表单（`scripts/pdf_extract.py` + `scripts/llm_client.py`，强制人工确认）。运行时配置 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（任意 OpenAI 兼容供应商）。
```

- [ ] **Step 2: README — add a short Phase 2 usage note**

In `README.md` near the cold-start / 运行方式 section, add:
```markdown
#### 上传 PDF 自动识别（Phase 2）
在网页「手动输入特征」tab 上传巨潮「发行安排及初步询价公告」PDF → 自动识别询价前字段并回填表单（请人工核对后再预测）。需配置环境变量 `LLM_API_KEY`（及可选 `LLM_BASE_URL` / `LLM_MODEL`，任意 OpenAI 兼容供应商；Streamlit 部署放 secrets）。未配置时回退为纯手动输入。
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: mark Phase 2 (PDF->LLM extraction) implemented; usage + env config"
```

---

## Self-Review

**Spec coverage (§6 PDF抽取与人工确认, §2.5 provider-agnostic, §3 modules):**
- 锁定发行公告、pdfplumber 取文本 → Task 3 (`extract_text`, MAX_CHARS cap). ✓
- LLM 按 JSON schema 抽取 → Task 2 (`extract_json`) + Task 3 (`SYSTEM_PROMPT`/`FIELD_SCHEMA`). ✓
- 强制人工确认（回填可编辑表单，不自动预测）→ Task 4 (prefill defaults; predict still behind the existing 预测 button). ✓
- provider-agnostic（base_url/key/model 配置化，接自备 API）→ Task 2 (env-only, no hardcoded provider). ✓
- 失败/未配置降级到手填 → Task 4 (is_configured guard + try/except → 手动输入). ✓
- 测试确定性（LLM mock）→ Tasks 2,3 (monkeypatched client / injected `extract_json`); real-PDF test `skipif`. ✓

**Placeholder scan:** none — every code step has full code; the only "drop a real PDF" is an intentional, documented skip with a concrete condition.

**Type consistency:** `extract_json(system, user, model=None) -> dict` used identically in `llm_client`, the injected test lambda `(system, user)`, and `pdf_extract` (calls `fn(SYSTEM_PROMPT, text)`). `ExtractResult(fields, text_chars, warnings)` consumed consistently in Task 4. `FIELD_SCHEMA` keys == `assemble_t6` raw keys (asserted in Task 3 Step 1).

---

## Done criteria

- `pytest tests/ -q` all green (Phase 1 suite + `test_llm_client` 3 + `test_pdf_extract` 2, 1 skipped).
- `streamlit run app.py` → upload a cninfo PDF (with `LLM_API_KEY` set) → fields prefill the form → user edits → 预测 works; without the key, a clear fallback message.
- Provider-agnostic: switching providers is purely `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` env changes, no code edits.
- Real-PDF validation: drop a sample 巨潮 公告 into `tests/fixtures/` to exercise `extract_text` end-to-end (needs a real PDF from the user).
