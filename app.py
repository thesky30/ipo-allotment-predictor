"""Streamlit demo — A股IPO网下中签率预测

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── sys.path so we can import scripts/ ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

# Load LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / TUSHARE_TOKEN from .env (local)
# or st.secrets (Streamlit Cloud) into the environment. See .env.example.
from config import load_env                 # noqa: E402
load_env()

from predict import (                       # noqa: E402
    predict_from_code,
    predict_from_dict,
    resolve_code_by_name,
    explain_prediction,
    DB_PATH,
)

OFFICIAL_STAGE = "T6"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO网下中签率预测",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .big-rate {
    font-size: 3rem; font-weight: 700;
    color: #1a6e3b; text-align: center;
    margin: 0.2em 0;
  }
  .sub-label {
    text-align: center; color: #555;
    font-size: 0.9rem; margin-bottom: 0.8em;
  }
  .card {
    background: #f0f7f2;
    border-radius: 10px;
    padding: 1.2em 1.5em;
    margin-bottom: 1em;
  }
  .conf-high   { color: #1a6e3b; font-weight: 600; }
  .conf-medium { color: #a06000; font-weight: 600; }
  .conf-low    { color: #b00000; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="加载模型…")
def _warmup():
    """Pre-load models into the predict module cache."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # touch global models via a dummy call; errors are fine here
        try:
            from predict import _load_model
            for s in ["T6", "T1", "T1PLUS"]:
                try:
                    _load_model(s)
                except Exception:
                    pass
        except Exception:
            pass

_warmup()


# ── DB helper ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_recent(n: int = 30) -> pd.DataFrame:
    """Last n IPOs with labels for reference table."""
    import sqlite3
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            df = pd.read_sql(f"""
                SELECT security_code, security_name, board, listing_date,
                       offline_oversubscription_ratio, offline_allotment_ratio_pct
                FROM ipo_offline_sample
                WHERE offline_oversubscription_ratio IS NOT NULL
                ORDER BY listing_date DESC
                LIMIT {n}
            """, conn)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _fetch_row(code: str) -> pd.DataFrame:
    """Fetch a full DB row by security code (for SHAP explanation)."""
    import sqlite3
    code_bare = code.split(".")[0]
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            return pd.read_sql(
                "SELECT * FROM ipo_offline_sample "
                "WHERE security_code = ? OR security_code LIKE ? LIMIT 1",
                conn, params=(code, code_bare + ".%"),
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _known_underwriters_and_industries():
    import reference_data
    p = reference_data.load_history().panel
    uws = sorted(p["primary_underwriter"].dropna().astype(str).unique().tolist())
    inds = sorted(
        p["sw_level1_industry_code"].dropna().astype(str).unique().tolist(),
        key=reference_data.sw_level1_industry_name,
    )
    return uws, inds


def _to_tushare_trade_date(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        dt = pd.Timestamp.today()
    return dt.strftime("%Y%m%d")


def _estimate_peer_pe_from_names(company_names: list[str], trade_date: str) -> dict:
    import peer_valuation

    pro = peer_valuation._pro_api()
    return peer_valuation.estimate_peer_pe_from_company_names(pro, company_names, trade_date)


def _show_prefill_summary(prefill: dict, sources: dict | None = None) -> None:
    if not prefill:
        return
    import ui_helpers

    rows = []
    for key, value in prefill.items():
        if isinstance(value, list):
            display = "、".join(map(str, value))
        else:
            display = value
        rows.append({
            "字段": key,
            "值": display,
            "来源": ui_helpers.prefill_source_label(key, (sources or {}).get(key)),
        })
    with st.expander("查看已回填字段与来源", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Result display ────────────────────────────────────────────────────────────
def _conf_html(conf: str) -> str:
    cls = {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}.get(conf, "")
    labels = {"high": "高", "medium": "中", "low": "低"}
    return f'<span class="{cls}">{labels.get(conf, conf)}</span>'


def show_result(res: dict) -> None:
    rate   = res["subscription_rate_display"]
    over   = res["oversubscription_ratio_pred"]
    board  = res["board"]
    conf   = res["confidence"]
    model  = res["model"]
    stage  = res["stage"]
    name   = res.get("security_name", "")
    code   = res.get("security_code", "")

    # Header
    title = f"{name}　{code}" if name and code else code
    st.markdown(f"### {title}　`{board}`")

    # Big rate card
    st.markdown(f"""
    <div class="card">
      <div class="sub-label">预计网下中签率</div>
      <div class="big-rate">{rate}</div>
      <div class="sub-label">预计超额认购倍数 {over:,.0f}×</div>
    </div>
    """, unsafe_allow_html=True)

    # Metadata row
    col1, col2, col3 = st.columns(3)
    col1.metric("预测阶段", stage)
    col2.metric("使用模型", model)
    col3.markdown(
        f"**置信度**<br>{_conf_html(conf)}",
        unsafe_allow_html=True,
    )

    # Actual vs predicted (if available)
    if "actual_subscription_rate_display" in res:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("预测中签率", res["subscription_rate_display"])
        c2.metric("实际中签率", res["actual_subscription_rate_display"])
        delta_pct = res["subscription_rate_pred_pct"] - res["actual_subscription_rate_pct"]
        c3.metric("误差 (预测−实际)", f"{delta_pct:+.5f}%",
                  delta=f"log误差 {res['prediction_error_log']:.3f}")

    # Interpretation hint
    if stage == OFFICIAL_STAGE:
        stage_note = "正式预测口径：询价开始前，不使用询价结果、回拨、申购或配售数据"
    else:
        stage_note = "研究对照口径：包含询价后信息，不作为领导确认后的正式预测口径"
    st.caption(f"网下中签率 = 1 ÷ 超额认购倍数 · {stage_note}")


def show_explanation(exp: dict) -> None:
    """Render a SHAP contribution waterfall as Chinese-safe HTML bars."""
    contribs = exp.get("contributions", [])
    if not contribs:
        return
    max_abs = max((abs(c["shap"]) for c in contribs), default=1.0) or 1.0
    base = exp["base_value"]
    pred = exp["predicted"]
    over = float(np.exp(pred))

    rows = []
    for c in contribs:
        s   = c["shap"]
        pct = abs(s) / max_abs * 100.0
        color = "#d9534f" if s >= 0 else "#3b82c4"   # red = pushes up, blue = pushes down
        sign  = "+" if s >= 0 else "−"
        val   = c["value"]
        if val is None:
            val_s = ""
        elif isinstance(val, float):
            val_s = f"（{val:,.3g}）"
        else:
            val_s = f"（{val}）"
        rows.append(
            f'<div style="display:flex;align-items:center;margin:3px 0;font-size:0.85rem;">'
            f'<div style="width:48%;text-align:right;padding-right:8px;color:#333;">'
            f'{c["label"]}<span style="color:#aaa;">{val_s}</span></div>'
            f'<div style="width:10%;text-align:right;padding-right:6px;color:{color};'
            f'font-weight:600;">{sign}{abs(s):.2f}</div>'
            f'<div style="width:42%;"><div style="height:13px;width:{pct:.0f}%;'
            f'background:{color};border-radius:3px;opacity:0.85;"></div></div>'
            f'</div>'
        )

    html = (
        f'<div style="margin:0.2em 0 0.6em;color:#555;font-size:0.85rem;">'
        f'基准值（训练均值, log）= {base:.2f}　→　预测值（log）= {pred:.2f}'
        f'（超额认购 {over:,.0f}×）</div>'
        + "".join(rows)
        + '<div style="margin-top:8px;color:#999;font-size:0.78rem;">'
          '🔴 正贡献＝推高超额认购倍数（即压低中签率）　·　🔵 负贡献＝相反</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _try_explain(explain_input, stage: str) -> None:
    """Best-effort SHAP explanation block; never breaks the main result."""
    try:
        exp = explain_prediction(explain_input, stage=stage)
        with st.expander("📊 预测解释：各特征贡献（SHAP）", expanded=True):
            show_explanation(exp)
    except Exception as e:
        st.caption(f"（解释暂不可用：{e}）")


def _render_no_label_note(res: dict) -> None:
    import pandas as pd
    as_of = res.get("data_as_of")
    for w in res.get("warnings", []):
        st.warning(w)
    st.caption(
        "⚠️ 本股暂无真实披露的网下中签率，**无法计算本股准确率**。\n\n"
        "下列为**模型整体回测水平**（OOS Spearman 0.62 / MAE 0.31，模型级，非本股）；"
        f"市场/参考数据截至 **{pd.Timestamp(as_of).date() if as_of is not None else '—'}**。"
    )
    pred = res.get("oversubscription_ratio_pred")
    board = res.get("board")
    if pred is not None and board:
        from predict import oversub_percentile
        pct = oversub_percentile(float(pred), board)
        if pct == pct:  # not NaN
            st.caption(f"该预测超额认购倍数处于同板块历史 **{pct*100:.0f}%** 分位。")


def run_and_show(code: str, stage: str) -> None:
    """Predict by code and render, with consistent error handling."""
    with st.spinner("预测中…"):
        try:
            res = predict_from_code(code, stage=stage, prefer_board_model=False)
            show_result(res)
            row = _fetch_row(res["security_code"])
            if not row.empty:
                _try_explain(row, stage)
        except ValueError as e:
            st.error(str(e))
        except FileNotFoundError as e:
            st.error(f"模型文件未找到：{e}")
        except Exception as e:
            st.error(f"预测出错：{e}")


def _looks_like_code(s: str) -> bool:
    """True if input is a numeric code (optionally with exchange suffix)."""
    return s.strip().split(".")[0].isdigit()


# ── Stage info ────────────────────────────────────────────────────────────────
STAGE_INFO = {
    "T6":     "**询价前正式模型**（推荐）— 核心业务口径。仅使用询价开始前已可获得的信息。",
    "T1":     "**询价后研究模型** — 使用询价结果，仅用于历史对照和信息增益分析。",
    "T1PLUS": "**回拨后研究模型** — 使用回拨后信息，仅用于上界参考。",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    stage = st.radio(
        "预测阶段",
        options=["T6", "T1", "T1PLUS"],
        index=0,
        format_func=lambda s: {"T6": "询价前正式（推荐）",
                               "T1": "询价后研究",
                               "T1PLUS": "回拨后研究"}[s],
    )
    st.markdown(STAGE_INFO[stage])
    st.divider()
    st.caption("正式模型：LightGBM · 特征：发行结构+申购规则+估值+历史市场热度+新增询价前因子\n\n"
               "OOS Spearman（询价前 T-6）：全局 0.619")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 A股IPO 网下中签率预测")
st.caption("基于LightGBM询价前正式模型 · 询价后/回拨后模型仅作研究对照")

tab_code, tab_manual, tab_recent = st.tabs(["股票代码查询", "手动输入特征", "近期IPO参考"])

# ── Tab 1: Code / name lookup ──────────────────────────────────────────────────
with tab_code:
    st.markdown("#### 输入股票代码或名称")
    st.caption("支持6位代码（688XXX / 300XXX / 00XXXX / 8XXXXX）、带后缀（688041.SH），"
               "或股票名称（如 禾迈股份）")

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        query_input = st.text_input(
            "股票代码或名称",
            placeholder="例：688041  /  300257  /  禾迈股份",
            label_visibility="collapsed",
        )
    with col_btn:
        predict_btn = st.button("预测", type="primary", use_container_width=True)

    if predict_btn:
        q = query_input.strip()
        st.session_state.pop("name_candidates", None)
        if not q:
            st.warning("请输入股票代码或名称")
        elif _looks_like_code(q):
            run_and_show(q, stage)
        else:
            cands = resolve_code_by_name(q)
            if not cands:
                st.error(f"未找到名称包含“{q}”的股票。新股请使用“手动输入特征”。")
            elif len(cands) == 1:
                run_and_show(cands[0]["security_code"], stage)
            else:
                st.session_state["name_candidates"] = cands

    # Disambiguation: multiple name matches pending
    cands = st.session_state.get("name_candidates")
    if cands:
        st.info(f"找到 {len(cands)} 只名称匹配的股票，请选择后预测：")
        labels = [f'{c["security_name"]}　{c["security_code"]}　{c["board"]}' for c in cands]
        sel = st.selectbox(
            "选择股票", options=list(range(len(cands))),
            format_func=lambda i: labels[i],
        )
        if st.button("确认预测", type="primary", key="confirm_name"):
            run_and_show(cands[sel]["security_code"], stage)

# ── Tab 2: Manual feature input ────────────────────────────────────────────────
with tab_manual:
    st.markdown("#### 手动输入新IPO特征")
    st.info("新股预测固定使用 T-6 询价前正式模型（无询价/回拨数据）。")

    try:
        _uw_list, _ind_list = _known_underwriters_and_industries()
    except Exception:
        _uw_list, _ind_list = [], []
    _uw_options = ["（未知/其他）"] + _uw_list
    _ind_options = [None] + _ind_list

    def _industry_label(code: str | None) -> str:
        if code is None:
            return "（未知/其他）"
        import reference_data
        return reference_data.sw_level1_industry_name(code)

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
                    st.session_state["pdf_prefill_sources"] = {k: "pdf" for k in res.fields}
                    for w in res.warnings:
                        st.warning(w)
                    st.success(f"已识别 {len(res.fields)} 个字段，已回填下方表单，请核对后再预测。")
                except Exception as e:
                    st.error(f"识别失败：{e}。请改为手动输入。")

    up2 = st.file_uploader("（可选）上传招股书 PDF 补充财务/估值字段（营收/CAGR/可比公司名单/拟募资）", type="pdf", key="prospectus_pdf")
    if up2 is not None and st.button("识别招股书字段", key="run_prospectus_extract"):
        import prospectus_extract, llm_client
        if not llm_client.is_configured():
            st.error("未配置 LLM（见 .env / Secrets：LLM_API_KEY），无法识别招股书。")
        else:
            with st.spinner("定位章节并识别招股书中…"):
                try:
                    res2 = prospectus_extract.extract_prospectus_fields(up2.read())
                    merged = dict(st.session_state.get("pdf_prefill", {}))
                    merged.update(res2.fields)
                    sources = dict(st.session_state.get("pdf_prefill_sources", {}))
                    sources.update({k: "prospectus" for k in res2.fields})
                    st.session_state["pdf_prefill"] = merged
                    st.session_state["pdf_prefill_sources"] = sources
                    for w in res2.warnings:
                        st.info(w)
                    st.success(f"招股书补充了 {len(res2.fields)} 个字段，请在下方核对。")
                except Exception as e:
                    st.error(f"招股书识别失败：{e}。请手动输入这些字段。")
    _pf = st.session_state.get("pdf_prefill", {})
    _src = st.session_state.get("pdf_prefill_sources", {})
    _show_prefill_summary(_pf, _src)

    sw_code_for_pe = _pf.get("sw_level1_industry_code")
    if sw_code_for_pe:
        try:
            import market_source
            import reference_data
            ref_pe = market_source.latest_sw_industry_pe(
                market_source.read_cached_sw_daily(),
                str(sw_code_for_pe),
                _to_tushare_trade_date(_pf.get("subscription_deadline_date")),
            )
            if ref_pe:
                st.caption(
                    f"Tushare 申万行业 PE 参考："
                    f"{reference_data.sw_level1_industry_name(sw_code_for_pe)} "
                    f"{ref_pe['pe']:.2f}（{ref_pe['trade_date']}）"
                )
                if st.button("回填行业 PE", key="fill_industry_pe"):
                    merged = dict(_pf)
                    merged["industry_pe_at_ipo"] = round(float(ref_pe["pe"]), 2)
                    st.session_state["pdf_prefill"] = merged
                    sources = dict(_src)
                    sources["industry_pe_at_ipo"] = "industry_pe"
                    st.session_state["pdf_prefill_sources"] = sources
                    _pf = merged
                    _src = sources
                    st.success(f"已回填行业 PE {float(ref_pe['pe']):.2f}")
        except Exception:
            pass

    peer_names = _pf.get("comparable_company_names") or []
    if isinstance(peer_names, str):
        peer_names = [x.strip() for x in peer_names.replace("，", ",").split(",") if x.strip()]
    if peer_names:
        st.info("招股书识别到可比公司：" + "、".join(map(str, peer_names)))
        if st.button("用 Tushare 计算可比公司 PE 并回填", key="run_peer_pe"):
            trade_date = _to_tushare_trade_date(_pf.get("subscription_deadline_date"))
            with st.spinner("拉取可比公司行情 PE 中…"):
                try:
                    stats = _estimate_peer_pe_from_names(list(peer_names), trade_date)
                    pe_mean = stats.get("peer_pe_ttm_mean")
                    if not pe_mean:
                        st.warning("未匹配到可用的可比公司 PE，请手动输入。")
                    else:
                        merged = dict(_pf)
                        merged["comparable_pe_avg_ex_nonrecurring"] = round(float(pe_mean), 2)
                        merged["peer_pe_ttm_median"] = stats.get("peer_pe_ttm_median")
                        merged["peer_pe_trade_date"] = stats.get("trade_date")
                        st.session_state["pdf_prefill"] = merged
                        sources = dict(_src)
                        sources["comparable_pe_avg_ex_nonrecurring"] = "peer_pe"
                        st.session_state["pdf_prefill_sources"] = sources
                        _pf = merged
                        _src = sources
                        st.success(
                            f"已回填可比公司 PE 均值 {float(pe_mean):.2f}，"
                            f"匹配 {stats.get('resolved_peer_count', 0)} 家："
                            + "、".join(stats.get("resolved_names", []))
                        )
                except Exception as e:
                    st.error(f"Tushare peer PE 拉取失败：{e}")

    with st.form("manual_form"):
        st.markdown("##### 基础信息")
        c1, c2 = st.columns(2)
        _boards = ["科创板", "创业板", "主板", "北交所"]
        board_sel = c1.selectbox("板块 *", _boards,
            index=_boards.index(_pf["board"]) if _pf.get("board") in _boards else 0)

        import datetime as _dt
        _dd = _pf.get("subscription_deadline_date")
        try:
            _dd = _dt.date.fromisoformat(_dd) if _dd else None
        except (TypeError, ValueError):
            _dd = None
        deadline_date = c2.date_input("申购截止日 *", value=_dd)

        c3, c4 = st.columns(2)
        uw_sel = c3.selectbox("主承销商", options=_uw_options,
            index=_uw_options.index(_pf["lead_underwriter"]) if _pf.get("lead_underwriter") in _uw_options else 0)
        ind_sel = c4.selectbox("申万一级行业", options=_ind_options,
            index=_ind_options.index(str(_pf["sw_level1_industry_code"])) if _pf.get("sw_level1_industry_code") is not None and str(_pf["sw_level1_industry_code"]) in _ind_options else 0,
            format_func=_industry_label)

        st.markdown("##### 发行结构")
        c5, c6 = st.columns(2)
        total_shares = c5.number_input("发行总股数（万股）", min_value=0.0,
            value=float(_pf.get("total_issue_shares_10k") or 0.0))
        offline_before = c6.number_input("网下发行量（回拨前，万股）", min_value=0.0,
            value=float(_pf.get("offline_issue_before_clawback_10k") or 0.0))

        c7, c8 = st.columns(2)
        online_before = c7.number_input("网上发行量（回拨前，万股）", min_value=0.0,
            value=float(_pf.get("online_issue_before_clawback_10k") or 0.0))
        _strat_alloc = _pf.get("strategic_allocation_10k")
        _total_pf = _pf.get("total_issue_shares_10k")
        if _strat_alloc and _total_pf and float(_total_pf) > 0:
            _strat_pct_val = round(float(_strat_alloc) / float(_total_pf) * 100, 2)
        else:
            _strat_pct_val = 0.0
        strategic_pct = c8.number_input("战略配售占比（%）", min_value=0.0, value=_strat_pct_val, step=0.1)

        st.markdown("##### 申购规则")
        c9, c10 = st.columns(2)
        sub_upper = c9.number_input("网下申购上限（万股）", min_value=0.0,
            value=float(_pf.get("subscription_upper_limit_10k") or 0.0))
        sub_lower = c10.number_input("网下申购下限（万股）", min_value=0.0,
            value=float(_pf.get("subscription_lower_limit_10k") or 0.0))

        c11, c12 = st.columns(2)
        sub_step = c11.number_input("网下申购步长（万股）", min_value=0.0,
            value=float(_pf.get("subscription_step_10k") or 0.0))
        mkt_threshold = c12.number_input("网下市值门槛（万元）", min_value=0.0,
            value=float(_pf.get("offline_market_value_threshold_10k_yuan") or 0.0))

        st.markdown("##### 财务估值")
        c13, c14 = st.columns(2)
        industry_pe = c13.number_input("行业PE", min_value=0.0,
            value=float(_pf.get("industry_pe_at_ipo") or 0.0), step=0.1)
        comparable_pe = c14.number_input("可比公司PE（招股书名单 + Tushare PE）", min_value=0.0,
            value=float(_pf.get("comparable_pe_avg_ex_nonrecurring") or 0.0), step=0.1)

        c15, c16 = st.columns(2)
        expected_raise = c15.number_input("预计募资额（亿元）", min_value=0.0,
            value=float(_pf.get("expected_fundraising_100m_yuan") or 0.0), step=0.1)
        revenue = c16.number_input("近一年营收（亿元）", min_value=0.0,
            value=float(_pf.get("latest_revenue_100m_yuan") or 0.0), step=0.1)
        revenue_cagr = st.number_input("3年营收CAGR（%）",
            value=float(_pf.get("revenue_cagr_3y_pct") or 0.0), step=0.1)

        submitted = st.form_submit_button("预测", type="primary")

    if submitted:
        if deadline_date is None:
            st.error("申购截止日为必填项，请选择日期后重试。")
        else:
            raw: dict = {
                "board":                                board_sel,
                "subscription_deadline_date":          str(deadline_date),
                "lead_underwriter":                    uw_sel if uw_sel != "（未知/其他）" else None,
                "sw_level1_industry_code":             ind_sel,
                "total_issue_shares_10k":              total_shares or None,
                "offline_issue_before_clawback_10k":   offline_before or None,
                "online_issue_before_clawback_10k":    online_before or None,
                "strategic_allocation_10k":            (
                    round(total_shares * strategic_pct / 100, 4) if total_shares and strategic_pct
                    else None
                ),
                "subscription_upper_limit_10k":        sub_upper or None,
                "subscription_lower_limit_10k":        sub_lower or None,
                "subscription_step_10k":               sub_step or None,
                "offline_market_value_threshold_10k_yuan": mkt_threshold or None,
                "industry_pe_at_ipo":                  industry_pe or None,
                "comparable_pe_avg_ex_nonrecurring":   comparable_pe or None,
                "expected_fundraising_100m_yuan":      expected_raise or None,
                "latest_revenue_100m_yuan":            revenue or None,
                "revenue_cagr_3y_pct":                 revenue_cagr if revenue_cagr != 0.0 else None,
            }
            # Drop None values
            raw = {k: v for k, v in raw.items() if v is not None}
            with st.spinner("组装因子并预测中…"):
                try:
                    from predict import predict_new_ipo
                    res = predict_new_ipo(raw, stage="T6")
                    show_result(res)
                    _render_no_label_note(res)
                    _try_explain(res.get("features", raw), "T6")
                except Exception as e:
                    st.error(f"预测出错：{e}")

# ── Tab 3: Recent IPO reference ────────────────────────────────────────────────
with tab_recent:
    st.markdown("#### 近期已上市IPO参考")
    df_ref = _load_recent(40)
    if df_ref.empty:
        st.info("数据库中暂无数据")
    else:
        df_ref = df_ref.rename(columns={
            "security_code": "代码",
            "security_name": "名称",
            "board":         "板块",
            "listing_date":  "上市日期",
            "offline_oversubscription_ratio": "网下超额认购倍数",
            "offline_allotment_ratio_pct":    "网下中签率(%)",
        })
        df_ref["网下中签率(%)"] = pd.to_numeric(df_ref["网下中签率(%)"], errors="coerce")
        df_ref["网下超额认购倍数"] = pd.to_numeric(df_ref["网下超额认购倍数"], errors="coerce").round(0)
        st.dataframe(
            df_ref,
            use_container_width=True,
            hide_index=True,
            column_config={
                "网下超额认购倍数": st.column_config.NumberColumn(format="localized"),
                "网下中签率(%)":   st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"共 {len(df_ref)} 条记录 · 按上市日期降序排列")
