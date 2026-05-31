import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import peer_valuation


def test_peer_pe_stats_filters_exclusions_and_non_positive_values():
    daily = pd.DataFrame([
        {"ts_code": "000001.SZ", "pe_ttm": 10.0, "pe": 11.0},
        {"ts_code": "000002.SZ", "pe_ttm": 20.0, "pe": 21.0},
        {"ts_code": "000003.SZ", "pe_ttm": -5.0, "pe": 30.0},
        {"ts_code": "000004.SZ", "pe_ttm": None, "pe": 40.0},
    ])
    members = pd.DataFrame([
        {"con_code": "000001.SZ"},
        {"con_code": "000002.SZ"},
        {"con_code": "000003.SZ"},
        {"con_code": "000004.SZ"},
    ])

    stats = peer_valuation.peer_pe_stats(
        daily, members, exclude_ts_codes={"000002.SZ"}, pe_col="pe_ttm"
    )

    assert stats["peer_count"] == 1
    assert stats["peer_pe_ttm_mean"] == 10.0
    assert stats["peer_pe_ttm_median"] == 10.0


def test_active_members_on_date_respects_entry_and_exit_dates():
    members = pd.DataFrame([
        {"con_code": "000001.SZ", "in_date": "20200101", "out_date": None},
        {"con_code": "000002.SZ", "in_date": "20270101", "out_date": None},
        {"con_code": "000003.SZ", "in_date": "20200101", "out_date": "20250101"},
    ])

    active = peer_valuation.active_members_on_date(members, "20260529")

    assert active["con_code"].tolist() == ["000001.SZ"]
