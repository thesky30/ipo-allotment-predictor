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


@dataclass
class AssemblyResult:
    features: dict[str, Any]
    data_as_of: pd.Timestamp
    warnings: list[str] = field(default_factory=list)


def _date_to_excel_serial(ts: pd.Timestamp) -> float:
    """Convert a Timestamp to Excel serial number (days since 1899-12-30)."""
    return float((ts - pd.Timestamp("1899-12-30")).days)


def assemble_t6(raw: dict[str, Any], history: reference_data.History | None = None) -> AssemblyResult:
    hist = history if history is not None else reference_data.load_history()
    warnings: list[str] = []

    sub_date = pd.to_datetime(raw.get("subscription_deadline_date"), errors="coerce")
    if pd.isna(sub_date):
        raise ValueError("subscription_deadline_date is required (申购截止日)")

    pred_date = pd.to_datetime(raw.get("prediction_date"), errors="coerce")
    if pd.isna(pred_date):
        pred_date = sub_date

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
    # add_features re-derives subscription_deadline_date from subscription_deadline_date_raw
    # (Excel serial), so we must supply the serial; the datetime column gets overwritten.
    new_sample["subscription_deadline_date_raw"] = _date_to_excel_serial(sub_date)
    # listing_date_raw must also be a valid numeric for add_features not to fail;
    # use subscription date as proxy (new IPO has no listing date yet).
    new_sample["listing_date_raw"] = _date_to_excel_serial(sub_date)
    for k in ("total_issue_shares_10k", "offline_issue_before_clawback_10k",
              "online_issue_before_clawback_10k", "strategic_allocation_10k",
              "subscription_upper_limit_10k", "subscription_lower_limit_10k",
              "subscription_step_10k", "industry_pe_at_ipo",
              "comparable_pe_avg_ex_nonrecurring", "offer_price_upper_yuan",
              "offer_price_lower_yuan"):
        if k in new_sample and raw.get(k) is not None:
            new_sample[k] = raw[k]
    new_sample_df = pd.DataFrame([new_sample]).astype(
        {c: t for c, t in hist.sample.dtypes.items() if c in new_sample}, errors="ignore")
    sample_aug = pd.concat([hist.sample, new_sample_df], ignore_index=True)
    built = add_features(sample_aug)
    new_built = built.iloc[-1]

    # 2) board rolling + underwriter/sw_l1 priors — reuse panel builders.
    new_panel = {c: np.nan for c in hist.panel.columns}
    new_panel["security_code"] = NEW_CODE
    new_panel["board"] = raw.get("board")
    # Subtract 1 nanosecond so the new row sorts strictly before same-date historical
    # stocks in prior_group_stats (which uses sort_values, not stable by default).
    # 1 ns does not change board-market lookup (trade_dates are day-resolution) and
    # does not affect merge_asof backward search semantics.
    new_panel["prediction_date"] = pred_date - pd.Timedelta(nanoseconds=1)
    new_panel["primary_underwriter"] = primary_underwriter(raw.get("lead_underwriter"))
    new_panel["sw_level1_industry_code"] = raw.get("sw_level1_industry_code")
    new_panel_df = pd.DataFrame([new_panel]).astype(
        {c: t for c, t in hist.panel.dtypes.items() if c in new_panel}, errors="ignore")
    panel_aug = pd.concat([hist.panel, new_panel_df], ignore_index=True)
    # Ensure prediction_date stays datetime after concat (mixed-type concat can
    # produce object dtype, which breaks merge_asof in add_board_market_factors).
    panel_aug["prediction_date"] = pd.to_datetime(panel_aug["prediction_date"], errors="coerce")
    # hist.panel already has board-rolling columns baked in from the DB; drop them
    # before re-running add_board_market_factors to avoid _x/_y column collisions
    # in merge_asof (which requires left frame NOT to have the merged columns).
    _board_rolling_cols = [
        "board_turnover_ma20", "board_turnover_pct_rank_1y",
        "board_turnover_ma20_over_ma60", "board_return_ma20",
    ]
    panel_aug = panel_aug.drop(
        columns=[c for c in _board_rolling_cols if c in panel_aug.columns]
    )
    # board_market.trade_date may be stored as str in SQLite; ensure it is
    # datetime64 so merge_asof can compare with prediction_date (datetime64).
    board_market = hist.board_market.copy()
    board_market["trade_date"] = pd.to_datetime(board_market["trade_date"], errors="coerce")
    panel_aug = add_board_market_factors(panel_aug, board_market)
    panel_aug = prior_group_stats(panel_aug, "primary_underwriter", "underwriter")
    panel_aug = prior_group_stats(panel_aug, "sw_level1_industry_code", "sw_l1")
    # add_board_market_factors and prior_group_stats sort by prediction_date, so
    # the new row is no longer last; locate it by security_code instead of iloc[-1].
    new_prior = panel_aug.loc[panel_aug["security_code"] == NEW_CODE].iloc[0]

    _uw_cnt = new_prior.get("underwriter_prior_ipo_count")
    if pd.isna(new_panel["primary_underwriter"]) or not (_uw_cnt and _uw_cnt > 0):
        warnings.append("该主承销商无历史样本，承销商先验按缺失处理。")
    _sw_cnt = new_prior.get("sw_l1_prior_ipo_count")
    if raw.get("sw_level1_industry_code") is None or not (_sw_cnt and _sw_cnt > 0):
        warnings.append("该申万一级行业无历史样本，行业先验按缺失处理。")

    # 3) Pass-through raw features add_features does not derive.
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

    def _to_none_if_na(v):
        if isinstance(v, str):
            return v
        try:
            return None if pd.isna(v) else v
        except (TypeError, ValueError):
            return v
    feats = {k: _to_none_if_na(v) for k, v in feats.items()}
    return AssemblyResult(features=feats, data_as_of=as_of, warnings=warnings)
