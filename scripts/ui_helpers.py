"""Pure UI metadata helpers for Streamlit forms.

Keep these free of Streamlit imports so field grouping and source labels can be
unit-tested without launching the app.
"""
from __future__ import annotations


def manual_form_sections() -> dict[str, list[str]]:
    return {
        "基础信息": [
            "board",
            "subscription_deadline_date",
            "lead_underwriter",
            "sw_level1_industry_code",
        ],
        "发行结构": [
            "total_issue_shares_10k",
            "offline_issue_before_clawback_10k",
            "online_issue_before_clawback_10k",
            "strategic_allocation_10k",
        ],
        "申购规则": [
            "subscription_upper_limit_10k",
            "subscription_lower_limit_10k",
            "subscription_step_10k",
            "offline_market_value_threshold_10k_yuan",
        ],
        "财务估值": [
            "industry_pe_at_ipo",
            "comparable_pe_avg_ex_nonrecurring",
            "expected_fundraising_100m_yuan",
            "latest_revenue_100m_yuan",
            "revenue_cagr_3y_pct",
        ],
        "市场参考": [
            "peer_pe_ttm_mean",
            "peer_pe_ttm_median",
            "comparable_company_names",
        ],
    }


def prefill_source_label(field: str, source: str | None) -> str:
    labels = {
        "manual": "手动输入",
        "pdf": "询价公告PDF",
        "prospectus": "招股书",
        "industry_pe": "Tushare行业PE参考",
        "peer_pe": "Tushare peer PE参考",
        "industry_constituent_pe": "Tushare申万行业成分PE兜底",
        "history": "历史上下文自动生成",
    }
    return labels.get(str(source or ""), "")
