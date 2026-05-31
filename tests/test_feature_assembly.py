import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest

import reference_data
from feature_assembly import assemble_t6

# Context features verified to match training exactly under leave-one-out assembly.
# The target stock is removed from history before assembly so that:
#   - concurrent-peer counts (concurrent_ipo_count, same_board_concurrent_ipo_count,
#     concurrent_offline_issue_sum_10k) are not inflated by the original DB row.
#   - recent_ipo_first_day_return_ma20 rolling windows include exactly the same
#     peers as training without a spurious extra row shifting the window.

CONTEXT_FEATS = [
    "market_turnover_ma20", "market_turnover_pct_rank_1y",
    "market_turnover_ma20_over_ma60", "market_return_ma20",
    "same_board_break_rate_ma10",
    "board_turnover_ma20", "board_turnover_pct_rank_1y",
    "board_turnover_ma20_over_ma60", "board_return_ma20",
    "underwriter_prior_ipo_count", "underwriter_prior_log_oversub_mean",
    "underwriter_prior_first_day_return_mean", "underwriter_prior_break_rate",
    "sw_l1_prior_ipo_count", "sw_l1_prior_log_oversub_mean",
    "sw_l1_prior_first_day_return_mean", "sw_l1_prior_break_rate",
    # Batch-competition features: pass under leave-one-out.
    "concurrent_ipo_count",
    "same_board_concurrent_ipo_count",
    "concurrent_offline_issue_sum_10k",
    # Rolling heat feature: also passes under leave-one-out.
    "recent_ipo_first_day_return_ma20",
]


def _load_panel_metadata() -> pd.DataFrame:
    """Load prediction_date, primary_underwriter, sw_level1_industry_code from DB panel."""
    db_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "ipo_offline.db"
    with sqlite3.connect(db_path) as conn:
        panel = pd.read_sql(
            "SELECT security_code, prediction_date, primary_underwriter, "
            "sw_level1_industry_code FROM new_factor_panel",
            conn,
        )
    panel["prediction_date"] = pd.to_datetime(panel["prediction_date"], errors="coerce")
    return panel


def _raw_from_row(row: pd.Series) -> dict:
    return {
        "board": row["board"],
        "subscription_deadline_date": row["subscription_deadline_date"],
        "prediction_date": row.get("prediction_date"),
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
    df = modeling_data.dropna(subset=["subscription_deadline_date"]).reset_index(drop=True)
    # Enrich with panel metadata (prediction_date, primary_underwriter, sw_level1_industry_code)
    # needed to reproduce board and prior features. These columns are not included in
    # load_modeling_data() which only merges NEW_T6_FACTOR_COLS from new_factor_panel.
    panel_meta = _load_panel_metadata()
    df = df.merge(panel_meta, on="security_code", how="left")
    row = df.iloc[offset % len(df)]
    code = row["security_code"]

    # Leave-one-out: remove the target stock from history before assembly so that
    # batch-competition features (concurrent_ipo_count, same_board_concurrent_ipo_count,
    # concurrent_offline_issue_sum_10k) reproduce training values exactly. Without this,
    # the original DB row stays in history and is counted as a concurrent peer of the
    # newly-assembled row, inflating counts by 1.
    full = reference_data.load_history()
    reduced = reference_data.History(
        sample=full.sample[full.sample["security_code"] != code].reset_index(drop=True),
        panel=full.panel[full.panel["security_code"] != code].reset_index(drop=True),
        board_market=full.board_market,
    )

    result = assemble_t6(_raw_from_row(row), history=reduced)
    for f in CONTEXT_FEATS:
        if f not in df.columns or pd.isna(row[f]):
            continue
        got = result.features.get(f)
        assert got is not None, f"{f} missing"
        assert np.isclose(float(got), float(row[f]), rtol=1e-6, atol=1e-6), (
            f"{f}: assembled {got} != training {row[f]}"
        )


def test_log_fields_and_missing_keys():
    r = assemble_t6({
        "board": "创业板",
        "subscription_deadline_date": "2024-06-03",
        "expected_fundraising_100m_yuan": 10.0,
        "latest_revenue_100m_yuan": 5.0,
    })
    assert np.isclose(r.features["log_expected_fundraising"], np.log1p(10.0))
    assert np.isclose(r.features["log_latest_revenue"], np.log1p(5.0))
    # No underwriter / industry supplied → a warning fires for each.
    assert any("承销商" in w for w in r.warnings)
    assert any("行业" in w for w in r.warnings)
    # When missing underwriter, prior count is filled with 0.0
    assert r.features["underwriter_prior_ipo_count"] == 0.0


def test_requires_subscription_date():
    with pytest.raises(ValueError):
        assemble_t6({"board": "主板"})


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
    # assert on the REAL prediction key found in Step 0:
    assert "oversubscription_ratio_pred" in res
    assert res["data_as_of"] is not None
    assert isinstance(res["warnings"], list)


def test_oversub_percentile_in_unit_range():
    from predict import oversub_percentile
    p = oversub_percentile(predicted_oversub=2000.0, board="科创板")
    assert 0.0 <= p <= 1.0
