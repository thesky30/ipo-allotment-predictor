import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest

from feature_assembly import assemble_t6

# Context features that can be exactly reproduced by assemble_t6 for in-DB stocks.
#
# Excluded from CONTEXT_FEATS (documented below):
#
# "concurrent_ipo_count", "same_board_concurrent_ipo_count",
# "concurrent_offline_issue_sum_10k":
#   When an in-DB stock is re-assembled, the new row is appended at the same
#   subscription_deadline_date as the original DB row. _concurrent_ipo_count
#   then counts the original row as a peer of the new row, inflating the count
#   by 1. This is CORRECT semantics for a genuinely new IPO (it should count
#   all existing historical stocks as concurrent peers), but it makes exact
#   reproduction of training values impossible for in-DB stocks.
#
# "recent_ipo_first_day_return_ma20":
#   add_features sorts each board group by sort_date_for_heat using quicksort
#   (default, non-stable). When the new row and the original in-DB stock share
#   the same subscription_deadline_date, the sort order between them is
#   non-deterministic. The rolling window either matches or is shifted by one
#   position depending on how quicksort partitions the tie.

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
    result = assemble_t6(_raw_from_row(row))
    for f in CONTEXT_FEATS:
        if f not in df.columns or pd.isna(row[f]):
            continue
        got = result.features.get(f)
        assert got is not None, f"{f} missing"
        assert np.isclose(float(got), float(row[f]), rtol=1e-6, atol=1e-6), (
            f"{f}: assembled {got} != training {row[f]}"
        )
