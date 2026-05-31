# Cold-Start IPO Prediction — Phase 1 Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict the网下 over-subscription / 中签率 for a brand-new IPO **not in the database**, from only询价前 raw fields, by auto-assembling the full 42-feature T-6 vector as-of the subscription date, and surface it honestly (no per-stock accuracy) in the Streamlit app.

**Architecture:** `feature_assembly` builds the new stock's raw row, **appends it to the historical reference frames, and re-runs the existing training-time feature builders** (`initial_data_analysis.add_features` for market/concurrent/heat/break; `build_new_factor_research` panel builders for board-rolling + underwriter/industry priors), then reads off the new row. Reusing the exact training code path makes train/serve skew structurally near-impossible; `baseline_models.load_modeling_data()` is the consistency oracle.

**Tech Stack:** Python, pandas, numpy, LightGBM (existing), pytest (new dev dep), Streamlit (existing).

**Scope note:** This is Plan 1 of 3 derived from the spec `docs/superpowers/specs/2026-05-31-cold-start-ipo-prediction-design.md`. Plan 2 = Phase 2 PDF→LLM extraction (`pdf_extract`, `llm_client`, upload UI). Plan 3 = market-data refresh (`market_source`, Tushare/AkShare). Plans 2 and 3 build on this one and are written separately.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `scripts/reference_data.py` | Load historical reference frames once; expose `data_as_of()` (max ref date in market_daily). Pure read, no network. | Create |
| `scripts/feature_assembly.py` | `assemble_t6(raw: dict) -> AssemblyResult`: append new row to history, run existing builders, return 42-feature dict + `data_as_of` + `warnings`. | Create |
| `scripts/predict.py` | Add `predict_new_ipo(raw, stage="T6")` thin wrapper: `assemble_t6` → `predict_from_dict`, passing through `data_as_of`/`warnings`. | Modify (append function) |
| `app.py` | Manual tab: call `predict_new_ipo`; show `data_as_of`, no-label note, model-level backtest caveat, same-board historical percentile. | Modify (`tab_manual` block ~327-402) |
| `tests/conftest.py` | pytest fixtures: load `load_modeling_data()` once. | Create |
| `tests/test_feature_assembly.py` | Consistency (skew) test vs `load_modeling_data()`; derived-field math; missing-key handling. | Create |
| `requirements-dev.txt` | `pytest>=8.0` | Create |

Reused existing functions (import as-is, do NOT reimplement):
- `initial_data_analysis.add_features(df)` — master builder (market×4, concurrent×3, heat×1, break×1, derived).
- `initial_data_analysis.load_market_daily()` — daily market frame with rolling aggregates.
- `build_new_factor_research.add_board_market_factors(panel, board_market)` — board_turnover×4.
- `build_new_factor_research.prior_group_stats(panel, key_col, prefix)` — `underwriter`/`sw_l1` priors×4 each.
- `build_new_factor_research.primary_underwriter(value)`, `build_new_factor_research.load_tables()`.
- `baseline_models.load_modeling_data()`, `baseline_models.FEATS_T6`.

---

## Task 0: Dev test harness

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest dev dependency**

`requirements-dev.txt`:
```text
-r requirements.txt
pytest>=8.0
```

- [ ] **Step 2: Create conftest with a cached modeling-data fixture**

`tests/conftest.py`:
```python
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def modeling_data():
    """Full training matrix (ipo_offline_sample ⋈ new_factor_panel)."""
    from baseline_models import load_modeling_data
    return load_modeling_data()
```

- [ ] **Step 3: Verify collection works**

Run: `pip install -r requirements-dev.txt && pytest tests/ --collect-only -q`
Expected: collects 0 tests, exit 0 (no errors importing conftest).

- [ ] **Step 4: Commit**

```bash
git add requirements-dev.txt tests/conftest.py
git commit -m "test: add pytest harness and modeling_data fixture"
```

---

## Task 1: reference_data — load history + data_as_of

**Files:**
- Create: `scripts/reference_data.py`
- Test: `tests/test_reference_data.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reference_data.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
from reference_data import load_history, data_as_of


def test_load_history_returns_nonempty_sample_and_panel():
    hist = load_history()
    assert len(hist.sample) > 100
    assert "security_code" in hist.sample.columns
    assert "board_turnover_ma20" in hist.panel.columns
    assert "primary_underwriter" in hist.panel.columns


def test_data_as_of_is_a_date():
    d = data_as_of()
    assert isinstance(d, pd.Timestamp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reference_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reference_data'`.

- [ ] **Step 3: Implement reference_data.py**

`scripts/reference_data.py`:
```python
"""Read-only loaders for the historical reference frames used to assemble
T-6 features for a brand-new IPO. No network, no Streamlit."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"


@dataclass(frozen=True)
class History:
    sample: pd.DataFrame        # ipo_offline_sample (raw + market/concurrent/heat baked in)
    panel: pd.DataFrame         # new_factor_panel (board rolling + priors)
    board_market: pd.DataFrame  # board_market_daily


def load_history() -> History:
    with sqlite3.connect(DB_PATH) as conn:
        sample = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)
        panel = pd.read_sql("SELECT * FROM new_factor_panel", conn)
        board_market = pd.read_sql("SELECT * FROM board_market_daily", conn)
    for df, col in [(sample, "subscription_deadline_date"),
                    (sample, "listing_date"),
                    (panel, "prediction_date")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return History(sample=sample, panel=panel, board_market=board_market)


def data_as_of() -> pd.Timestamp:
    """Latest trading day available in market_daily (freshness boundary)."""
    from initial_data_analysis import load_market_daily
    md = load_market_daily()
    if md is None or md.empty:
        return pd.Timestamp.min
    return pd.Timestamp(md["trade_date"].max())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reference_data.py -v`
Expected: PASS (2 passed). If `board_market_daily` is not a table, change the read to `pd.read_csv(DATA_DIR / "board_market_daily.csv")` and re-run.

- [ ] **Step 5: Commit**

```bash
git add scripts/reference_data.py tests/test_reference_data.py
git commit -m "feat: reference_data loaders for cold-start feature assembly"
```

---

## Task 2: feature_assembly — assemble the 42-feature T-6 vector

**Files:**
- Create: `scripts/feature_assembly.py`
- Test: extended in Task 3.

- [ ] **Step 1: Define the input contract and result type**

`scripts/feature_assembly.py` (part 1 — types & raw-row construction):
```python
"""Assemble the full T-6 feature vector for a brand-new IPO not in the DB.

Strategy: build the new stock's raw row, append it to the historical frames,
re-run the EXACT training-time builders, then read the new row back. Reusing
the training code path makes train/serve skew structurally near-impossible."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import reference_data
from initial_data_analysis import add_features
from build_new_factor_research import (
    add_board_market_factors,
    prior_group_stats,
    primary_underwriter,
)
from baseline_models import FEATS_T6

NEW_CODE = "__NEW_IPO__"

# Raw input keys the caller (form/PDF) must supply. Keys are optional; missing
# numeric inputs become NaN and the model imputes them.
RAW_INPUT_KEYS = [
    "board", "subscription_deadline_date", "lead_underwriter",
    "sw_level1_industry_code",
    "total_issue_shares_10k", "offline_issue_before_clawback_10k",
    "online_issue_before_clawback_10k", "strategic_allocation_10k",
    "subscription_upper_limit_10k", "subscription_lower_limit_10k",
    "subscription_step_10k", "offline_market_value_threshold_10k_yuan",
    "industry_pe_at_ipo", "comparable_pe_avg_ex_nonrecurring",
    "expected_fundraising_100m_yuan", "latest_revenue_100m_yuan",
    "revenue_cagr_3y_pct", "offer_price_upper_yuan", "offer_price_lower_yuan",
]


@dataclass
class AssemblyResult:
    features: dict[str, Any]          # 42 T-6 features for predict_from_dict
    data_as_of: pd.Timestamp
    warnings: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Implement the assembly function**

`scripts/feature_assembly.py` (part 2 — append, run builders, read back):
```python
def assemble_t6(raw: dict[str, Any]) -> AssemblyResult:
    hist = reference_data.load_history()
    warnings: list[str] = []

    sub_date = pd.to_datetime(raw.get("subscription_deadline_date"), errors="coerce")
    if pd.isna(sub_date):
        raise ValueError("subscription_deadline_date is required (申购截止日)")

    as_of = reference_data.data_as_of()
    if sub_date > as_of:
        warnings.append(
            f"申购截止日 {sub_date.date()} 晚于参考数据截止 {as_of.date()}，"
            f"上下文因子用最新可得数据近似。"
        )

    # 1) market/concurrent/heat/break + derived — reuse add_features on history+new.
    new_sample = {c: np.nan for c in hist.sample.columns}
    new_sample["security_code"] = NEW_CODE
    new_sample["board"] = raw.get("board")
    new_sample["subscription_deadline_date"] = sub_date
    for k in ("total_issue_shares_10k", "offline_issue_before_clawback_10k",
              "online_issue_before_clawback_10k", "strategic_allocation_10k",
              "subscription_upper_limit_10k", "subscription_lower_limit_10k",
              "subscription_step_10k", "industry_pe_at_ipo",
              "comparable_pe_avg_ex_nonrecurring", "offer_price_upper_yuan",
              "offer_price_lower_yuan"):
        if k in new_sample and raw.get(k) is not None:
            new_sample[k] = raw[k]
    sample_aug = pd.concat([hist.sample, pd.DataFrame([new_sample])],
                           ignore_index=True)
    built = add_features(sample_aug)
    new_built = built.iloc[-1]

    # 2) board rolling + underwriter/sw_l1 priors — reuse panel builders.
    new_panel = {c: np.nan for c in hist.panel.columns}
    new_panel["security_code"] = NEW_CODE
    new_panel["board"] = raw.get("board")
    new_panel["prediction_date"] = sub_date
    new_panel["primary_underwriter"] = primary_underwriter(raw.get("lead_underwriter"))
    new_panel["sw_level1_industry_code"] = raw.get("sw_level1_industry_code")
    panel_aug = pd.concat([hist.panel, pd.DataFrame([new_panel])],
                          ignore_index=True)
    panel_aug = add_board_market_factors(panel_aug, hist.board_market)
    panel_aug = prior_group_stats(panel_aug, "primary_underwriter", "underwriter")
    panel_aug = prior_group_stats(panel_aug, "sw_level1_industry_code", "sw_l1")
    new_prior = panel_aug.iloc[-1]

    if new_panel["primary_underwriter"] is None or pd.isna(new_prior.get("underwriter_prior_ipo_count")):
        warnings.append("该主承销商无历史样本，承销商先验按缺失处理。")
    if raw.get("sw_level1_industry_code") is None or pd.isna(new_prior.get("sw_l1_prior_ipo_count")):
        warnings.append("该申万一级行业无历史样本，行业先验按缺失处理。")

    # 3) Pass-through raw features that add_features does not derive.
    feats: dict[str, Any] = {}
    for f in FEATS_T6:
        if f in built.columns:
            feats[f] = new_built.get(f)
        elif f in panel_aug.columns:
            feats[f] = new_prior.get(f)
        else:
            feats[f] = raw.get(f)
    for k in ("expected_fundraising_100m_yuan", "latest_revenue_100m_yuan",
              "revenue_cagr_3y_pct", "offline_market_value_threshold_10k_yuan"):
        if raw.get(k) is not None:
            feats[k] = raw[k]
    if raw.get("expected_fundraising_100m_yuan") is not None:
        feats["log_expected_fundraising"] = float(np.log1p(raw["expected_fundraising_100m_yuan"]))
    if raw.get("latest_revenue_100m_yuan") is not None:
        feats["log_latest_revenue"] = float(np.log1p(raw["latest_revenue_100m_yuan"]))
    feats["board"] = raw.get("board")

    feats = {k: (None if (not isinstance(v, str) and pd.isna(v)) else v)
             for k, v in feats.items()}
    return AssemblyResult(features=feats, data_as_of=as_of, warnings=warnings)
```

- [ ] **Step 3: Smoke-run the assembly once**

Run:
```bash
python -c "import sys; sys.path.insert(0,'scripts'); from feature_assembly import assemble_t6; \
r=assemble_t6({'board':'科创板','subscription_deadline_date':'2024-09-01','lead_underwriter':'中信证券', \
'sw_level1_industry_code':'730000','total_issue_shares_10k':5000,'offline_issue_before_clawback_10k':3000, \
'industry_pe_at_ipo':35.0,'expected_fundraising_100m_yuan':12.0,'latest_revenue_100m_yuan':8.0}); \
print('n_features', len(r.features)); print('as_of', r.data_as_of.date()); print('warnings', r.warnings); \
print('market_turnover_ma20', r.features.get('market_turnover_ma20'))"
```
Expected: prints `n_features` ≥ 42, a date, possibly warnings, and a non-None `market_turnover_ma20`.

- [ ] **Step 4: Commit**

```bash
git add scripts/feature_assembly.py
git commit -m "feat: feature_assembly.assemble_t6 reusing training builders"
```

---

## Task 3: Consistency (train-serve skew) test — the linchpin

**Files:**
- Create: `tests/test_feature_assembly.py`

- [ ] **Step 1: Write the failing test**

`tests/test_feature_assembly.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest

from feature_assembly import assemble_t6

# Context features that assemble_t6 recomputes; must match the stored training row.
CONTEXT_FEATS = [
    "market_turnover_ma20", "market_turnover_pct_rank_1y",
    "market_turnover_ma20_over_ma60", "market_return_ma20",
    "recent_ipo_first_day_return_ma20", "same_board_break_rate_ma10",
    "concurrent_ipo_count", "same_board_concurrent_ipo_count",
    "concurrent_offline_issue_sum_10k",
    "board_turnover_ma20", "board_turnover_pct_rank_1y",
    "board_turnover_ma20_over_ma60", "board_return_ma20",
    "underwriter_prior_ipo_count", "underwriter_prior_log_oversub_mean",
    "underwriter_prior_first_day_return_mean", "underwriter_prior_break_rate",
    "sw_l1_prior_ipo_count", "sw_l1_prior_log_oversub_mean",
    "sw_l1_prior_first_day_return_mean", "sw_l1_prior_break_rate",
]


def _raw_from_row(row: pd.Series) -> dict:
    return {
        "board": row["board"],
        "subscription_deadline_date": row["subscription_deadline_date"],
        "lead_underwriter": row.get("primary_underwriter"),
        "sw_level1_industry_code": row.get("sw_level1_industry_code"),
        "total_issue_shares_10k": row.get("total_issue_shares_10k"),
        "offline_issue_before_clawback_10k": row.get("offline_issue_before_clawback_10k"),
        "online_issue_before_clawback_10k": row.get("online_issue_before_clawback_10k"),
        "strategic_allocation_10k": row.get("strategic_allocation_10k"),
        "industry_pe_at_ipo": row.get("industry_pe_at_ipo"),
        "expected_fundraising_100m_yuan": row.get("expected_fundraising_100m_yuan"),
        "latest_revenue_100m_yuan": row.get("latest_revenue_100m_yuan"),
    }


@pytest.mark.parametrize("offset", [200, 400, 600])
def test_assembly_matches_training_context_features(modeling_data, offset):
    """For an in-DB stock, re-deriving context features must reproduce the
    stored training values (no train/serve skew)."""
    df = modeling_data.dropna(subset=["subscription_deadline_date"]).reset_index(drop=True)
    row = df.iloc[offset % len(df)]
    result = assemble_t6(_raw_from_row(row))
    for f in CONTEXT_FEATS:
        if f not in df.columns or pd.isna(row[f]):
            continue
        got = result.features.get(f)
        assert got is not None, f"{f} missing"
        assert np.isclose(float(got), float(row[f]), rtol=1e-6, atol=1e-6), (
            f"{f}: assembled {got} != training {row[f]}"
        )
```

- [ ] **Step 2: Run test to verify it fails (or surfaces skew)**

Run: `pytest tests/test_feature_assembly.py -v`
Expected initially: FAIL on at least one feature, OR pass. If a feature mismatches, the message names it — investigate that builder's as-of anchor (likely `prediction_date` vs `subscription_deadline_date` difference) before changing anything.

- [ ] **Step 3: Reconcile the as-of anchor if needed**

If priors/board mismatch: the panel builders anchor on `prediction_date` (询价开始日), not `subscription_deadline_date`. In `assemble_t6`, set `new_panel["prediction_date"]` from the same field training used. For the test, pass `row["prediction_date"]` through `_raw_from_row` and into `assemble_t6` as an optional `prediction_date` key; default it to `subscription_deadline_date` when absent. Minimal change in `feature_assembly.py`:
```python
    pred_date = pd.to_datetime(raw.get("prediction_date"), errors="coerce")
    if pd.isna(pred_date):
        pred_date = sub_date
    new_panel["prediction_date"] = pred_date
```
And add `"prediction_date": row.get("prediction_date")` to `_raw_from_row`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_assembly.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_feature_assembly.py scripts/feature_assembly.py
git commit -m "test: assert feature_assembly reproduces training features (no skew)"
```

---

## Task 4: Derived-field + missing-key unit tests

**Files:**
- Modify: `tests/test_feature_assembly.py` (append)

- [ ] **Step 1: Write the tests**

Append to `tests/test_feature_assembly.py`:
```python
def test_log_fields_and_missing_keys():
    r = assemble_t6({
        "board": "创业板",
        "subscription_deadline_date": "2024-06-03",
        "expected_fundraising_100m_yuan": 10.0,
        "latest_revenue_100m_yuan": 5.0,
    })
    assert np.isclose(r.features["log_expected_fundraising"], np.log1p(10.0))
    assert np.isclose(r.features["log_latest_revenue"], np.log1p(5.0))
    # No underwriter / industry supplied → priors missing, with a warning.
    assert r.features["underwriter_prior_ipo_count"] is None
    assert any("承销商" in w for w in r.warnings)


def test_requires_subscription_date():
    with pytest.raises(ValueError):
        assemble_t6({"board": "主板"})
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_feature_assembly.py -v`
Expected: PASS (5 passed total).

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_assembly.py
git commit -m "test: derived-field math and missing-key handling"
```

---

## Task 5: predict.py — predict_new_ipo wrapper

**Files:**
- Modify: `scripts/predict.py` (append near `predict_from_dict`, ~line 430)
- Modify: `tests/test_feature_assembly.py` (append integration test)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_feature_assembly.py`:
```python
def test_predict_new_ipo_returns_prediction():
    from predict import predict_new_ipo
    res = predict_new_ipo({
        "board": "科创板",
        "subscription_deadline_date": "2024-09-01",
        "lead_underwriter": "中信证券",
        "sw_level1_industry_code": "730000",
        "total_issue_shares_10k": 5000,
        "offline_issue_before_clawback_10k": 3000,
        "industry_pe_at_ipo": 35.0,
        "expected_fundraising_100m_yuan": 12.0,
    })
    assert "predicted_oversubscription" in res or "predicted" in res
    assert res["data_as_of"] is not None
    assert isinstance(res["warnings"], list)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_feature_assembly.py::test_predict_new_ipo_returns_prediction -v`
Expected: FAIL — `ImportError: cannot import name 'predict_new_ipo'`.

- [ ] **Step 3: Implement predict_new_ipo**

Append to `scripts/predict.py`:
```python
def predict_new_ipo(raw: dict[str, Any], stage: str = "T6") -> dict[str, Any]:
    """Cold-start prediction for an IPO not in the DB.

    Assembles the full T-6 feature vector as-of the subscription date, then
    predicts. Adds ``data_as_of`` and ``warnings`` to the result for honest UI.
    """
    from feature_assembly import assemble_t6
    asm = assemble_t6(raw)
    result = predict_from_dict(asm.features, stage=stage, prefer_board_model=False)
    result["data_as_of"] = asm.data_as_of
    result["warnings"] = asm.warnings
    result["security_name"] = raw.get("security_name", "新IPO")
    return result
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_feature_assembly.py::test_predict_new_ipo_returns_prediction -v`
Expected: PASS. (If the result key differs, adjust the assert to the actual key printed by `predict_from_dict`; inspect with the smoke command from Task 2 Step 3.)

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py tests/test_feature_assembly.py
git commit -m "feat: predict_new_ipo cold-start wrapper with data_as_of/warnings"
```

---

## Task 6: app.py — wire manual tab to cold-start + honest display

**Files:**
- Modify: `app.py` — `tab_manual` submit block (~lines 332-402) and result area.

- [ ] **Step 1: Add the missing raw inputs to the manual form**

In `app.py` `with st.form("manual_form"):` add fields so the form collects the keys `feature_assembly` needs. After the existing `c9,c10` block insert:
```python
        c11, c12 = st.columns(2)
        sub_deadline = c11.date_input("申购截止日 *", help="自动补算市场/板块/历史因子的时点锚点")
        lead_uw = c12.text_input("主承销商", help="用于承销商历史先验")
        c13, c14 = st.columns(2)
        sw_l1_code = c13.text_input("申万一级行业代码", help="如 730000；用于行业历史先验")
        exp_fund = c14.number_input("预计募资额（亿元）", min_value=0.0, value=0.0, step=0.1)
        c15, c16 = st.columns(2)
        latest_rev = c15.number_input("近一年营收（亿元）", min_value=0.0, value=0.0, step=0.1)
        rev_cagr = c16.number_input("3年营收CAGR（%）", value=0.0, step=0.1)
        mkt_thr = st.number_input("网下询价市值门槛（万元）", min_value=0.0, value=0.0)
```

- [ ] **Step 2: Replace the submit handler to use predict_new_ipo**

Replace the `if submitted:` block (currently building `raw` and calling `predict_from_dict`) with:
```python
    if submitted:
        raw = {
            "board": board_sel,
            "subscription_deadline_date": str(sub_deadline),
            "lead_underwriter": lead_uw or None,
            "sw_level1_industry_code": sw_l1_code or None,
            "total_issue_shares_10k": total_shares or None,
            "offline_issue_before_clawback_10k": offline_before or None,
            "online_issue_before_clawback_10k": online_before or None,
            "strategic_allocation_10k": None,
            "strategic_allocation_share_pct": strategic_pct or None,
            "subscription_upper_limit_10k": sub_upper or None,
            "subscription_lower_limit_10k": sub_lower or None,
            "subscription_step_10k": sub_step or None,
            "offline_market_value_threshold_10k_yuan": mkt_thr or None,
            "industry_pe_at_ipo": industry_pe or None,
            "expected_fundraising_100m_yuan": exp_fund or None,
            "latest_revenue_100m_yuan": latest_rev or None,
            "revenue_cagr_3y_pct": rev_cagr if rev_cagr != 0.0 else None,
        }
        with st.spinner("组装因子并预测中…"):
            try:
                from predict import predict_new_ipo
                res = predict_new_ipo(raw, stage="T6")
                show_result(res)
                _render_no_label_note(res)
                _try_explain(res.get("features", raw), "T6")
            except Exception as e:
                st.error(f"预测出错：{e}")
```

- [ ] **Step 2b: Restrict the manual tab to T-6**

Because cold-start has no询价 data, T-1/T+1 research inputs are not meaningful here. Above the form add:
```python
    st.info("新股预测固定使用 T-6 询价前正式模型（无询价/回拨数据）。")
```
and remove the `if stage != OFFICIAL_STAGE:` research-field block from `tab_manual` (lines ~354-373) and the `compute_t1_features` call.

- [ ] **Step 3: Add the honest no-label note helper**

Add near the other helpers in `app.py` (after `show_explanation`):
```python
def _render_no_label_note(res: dict) -> None:
    as_of = res.get("data_as_of")
    for w in res.get("warnings", []):
        st.warning(w)
    st.caption(
        "⚠️ 本股暂无真实披露的网下中签率，**无法计算本股准确率**。\n\n"
        "下列为**模型整体回测水平**（OOS Spearman 0.62 / MAE 0.31，模型级，非本股）；"
        f"市场/参考数据截至 **{pd.Timestamp(as_of).date() if as_of is not None else '—'}**。"
    )
```

- [ ] **Step 4: Manual smoke test**

Run: `streamlit run app.py`
Then in the browser: 手动输入特征 tab → fill 板块=科创板, 申购截止日, 主承销商=中信证券, 申万代码=730000, a few 发行字段 → 预测.
Expected: a prediction renders; a "数据截至 YYYY-MM-DD" caption and the no-label warning show; any missing-prior warnings appear. No exception.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: manual tab does cold-start T-6 prediction with honest no-label display"
```

---

## Self-Review

**Spec coverage:**
- §2.1 全自动补算 → Tasks 2,3 (assemble + skew test). ✓
- §2.2 无标签诚实展示 → Task 6 Step 3 (`_render_no_label_note`). ✓
- §4 输入 schema (16 raw + 3 keys + derived + 21 context) → Tasks 2,6. ✓
- §5 train-serve skew 对策 (reuse builders + consistency test) → Tasks 2,3. ✓ (Reuse replaces the "extract shared functions" refactor — same guarantee, less churn; the consistency test enforces it.)
- §7 结果展示 (percentile, contributions, data_as_of) → Task 6; **gap:** same-board historical percentile not yet implemented. **Added below as Task 7.**
- §8 数据刷新, §6 PDF — out of Plan 1 scope (Plans 2 & 3). ✓
- §10 错误处理 (missing prior, future date) → Task 2 warnings, Task 4 test. ✓
- §11 测试 → Tasks 0,1,3,4,5. ✓

**Placeholder scan:** none ("appropriate error handling" etc. absent; all steps show code/commands).

**Type consistency:** `assemble_t6` → `AssemblyResult(features, data_as_of, warnings)` used consistently in Tasks 2-5; `predict_new_ipo` adds `data_as_of`/`warnings` consumed in Task 6. ✓

---

## Task 7: Same-board historical percentile (spec §7 gap)

**Files:**
- Modify: `scripts/predict.py` (append helper)
- Modify: `app.py` (call in `_render_no_label_note`)
- Modify: `tests/test_feature_assembly.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feature_assembly.py`:
```python
def test_oversub_percentile_in_unit_range():
    from predict import oversub_percentile
    p = oversub_percentile(predicted_oversub=2000.0, board="科创板")
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_feature_assembly.py::test_oversub_percentile_in_unit_range -v`
Expected: FAIL — `ImportError: cannot import name 'oversub_percentile'`.

- [ ] **Step 3: Implement**

Append to `scripts/predict.py`:
```python
def oversub_percentile(predicted_oversub: float, board: str) -> float:
    """Percentile of a predicted over-subscription ratio among same-board
    historical disclosed values (0..1). Falls back to all boards if sparse."""
    import reference_data
    hist = reference_data.load_history()
    s = hist.sample
    col = "offline_oversubscription_ratio"
    vals = pd.to_numeric(s.loc[s["board"] == board, col], errors="coerce").dropna()
    if len(vals) < 20:
        vals = pd.to_numeric(s[col], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float((vals < predicted_oversub).mean())
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_feature_assembly.py::test_oversub_percentile_in_unit_range -v`
Expected: PASS.

- [ ] **Step 5: Show percentile in the UI**

In `app.py` `_render_no_label_note`, after the caption add:
```python
    pred = res.get("predicted_oversubscription") or res.get("predicted")
    if pred is not None:
        from predict import oversub_percentile
        pct = oversub_percentile(float(pred), res.get("board", ""))
        if pct == pct:  # not NaN
            st.caption(f"该预测超额认购倍数处于同板块历史 **{pct*100:.0f}%** 分位。")
```

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py app.py tests/test_feature_assembly.py
git commit -m "feat: same-board historical percentile for cold-start result"
```

---

## Done criteria

- `pytest tests/ -v` all green (reference_data, assembly skew ×3, derived/missing, integration, percentile).
- `streamlit run app.py` → 手动输入特征 tab predicts a never-seen IPO, shows data-as-of, no-label caveat, missing-prior warnings, and same-board percentile.
- Next: Plan 2 (Phase 2 PDF→LLM extraction) and Plan 3 (market-data refresh).
