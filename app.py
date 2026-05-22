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

from predict import (                       # noqa: E402
    predict_from_code,
    predict_from_dict,
    compute_t1_features,
    DB_PATH,
)

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
    st.caption(
        f"ℹ️ 网下中签率 = 1 ÷ 超额认购倍数  ·  "
        f"T-1 演示模型仅使用询价结果，不含任何网下申购数据"
    )


# ── Stage info ────────────────────────────────────────────────────────────────
STAGE_INFO = {
    "T1":     "**T-1 演示模型**（推荐）— 询价完成后，申购开始前。使用询价结果。",
    "T6":     "**T-6 早期模型** — 询价开始前。仅用招股书/行业PE等信息。精度较低。",
    "T1PLUS": "**T+1 回拨后模型** — 申购完成、回拨比例公告后。精度最高。",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    stage = st.radio(
        "预测阶段",
        options=["T1", "T6", "T1PLUS"],
        index=0,
        format_func=lambda s: {"T1": "T-1 演示（推荐）",
                               "T6": "T-6 早期",
                               "T1PLUS": "T+1 回拨后"}[s],
    )
    st.markdown(STAGE_INFO[stage])
    st.divider()
    st.caption("模型：LightGBM · 特征：询价结果+发行结构+市场热度\n\n"
               "OOS Spearman（T-1）：全局 0.951\n科创板 0.984 · 创业板 0.979")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 A股IPO 网下中签率预测")
st.caption("基于LightGBM三阶段模型 · 演示版本")

tab_code, tab_manual, tab_recent = st.tabs(["股票代码查询", "手动输入特征", "近期IPO参考"])

# ── Tab 1: Code lookup ────────────────────────────────────────────────────────
with tab_code:
    st.markdown("#### 输入股票代码")
    st.caption("支持6位代码（688XXX / 300XXX / 00XXXX / 8XXXXX）或带后缀（688041.SH）")

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        code_input = st.text_input(
            "股票代码",
            placeholder="例：688041  或  300257",
            label_visibility="collapsed",
        )
    with col_btn:
        predict_btn = st.button("预测", type="primary", use_container_width=True)

    if predict_btn:
        if not code_input.strip():
            st.warning("请输入股票代码")
        else:
            with st.spinner("预测中…"):
                try:
                    res = predict_from_code(
                        code_input.strip(),
                        stage=stage,
                        prefer_board_model=False,
                    )
                    show_result(res)
                except ValueError as e:
                    st.error(str(e))
                except FileNotFoundError as e:
                    st.error(f"模型文件未找到：{e}")
                except Exception as e:
                    st.error(f"预测出错：{e}")

# ── Tab 2: Manual feature input ────────────────────────────────────────────────
with tab_manual:
    st.markdown("#### 手动输入新IPO特征")
    st.caption("适用于尚未入库的新IPO。仅T-1演示模型需要询价结果字段。")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        board_sel = c1.selectbox("板块 *", ["科创板", "创业板", "主板", "北交所"])
        offer_price = c2.number_input("最终发行价（元）", min_value=0.0, value=0.0, step=0.01)

        c3, c4 = st.columns(2)
        inq_total = c3.number_input("询价申购总量（万股）", min_value=0.0, value=0.0)
        offline_before = c4.number_input("回拨前网下发行量（万股）", min_value=0.0, value=0.0)

        c5, c6 = st.columns(2)
        investors = c5.number_input("参与询价投资者数（家）", min_value=0, value=0, step=1)
        allot_accts = c6.number_input("配售对象数（个）", min_value=0, value=0, step=1)

        c7, c8 = st.columns(2)
        issue_amt = c7.number_input("募集资金总额（亿元）", min_value=0.0, value=0.0, step=0.1)
        total_shares = c8.number_input("发行总股数（万股）", min_value=0.0, value=0.0)

        c9, c10 = st.columns(2)
        mkt_heat = c9.number_input("近20只同板块IPO首日涨幅均值（%）", value=0.0, step=0.1,
                                    help="recent_ipo_first_day_return_ma20；留0则模型自行推算")
        quote_avg = c10.number_input("询价加权均价（元，可选）", min_value=0.0, value=0.0, step=0.01)

        submitted = st.form_submit_button("预测", type="primary")

    if submitted:
        raw: dict = {
            "board":                          board_sel,
            "offer_price_yuan":               offer_price or None,
            "inquiry_subscription_total_10k": inq_total or None,
            "offline_issue_before_clawback_10k": offline_before or None,
            "inquiry_investors_count":        investors or None,
            "inquiry_allotment_accounts":     allot_accts or None,
            "issue_amount_100m_yuan":         issue_amt or None,
            "total_issue_shares_10k":         total_shares or None,
            "recent_ipo_first_day_return_ma20": mkt_heat if mkt_heat != 0.0 else None,
            "quote_price_weighted_avg":       quote_avg or None,
        }
        # Remove None values; compute derived features
        raw = {k: v for k, v in raw.items() if v is not None}
        raw = compute_t1_features(raw)

        with st.spinner("预测中…"):
            try:
                res = predict_from_dict(raw, stage=stage, prefer_board_model=False)
                show_result(res)
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
        df_ref["网下超额认购倍数"] = pd.to_numeric(df_ref["网下超额认购倍数"], errors="coerce")
        st.dataframe(
            df_ref.style.format({
                "网下超额认购倍数": "{:,.0f}",
                "网下中签率(%)":   "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"共 {len(df_ref)} 条记录 · 按上市日期降序排列")
