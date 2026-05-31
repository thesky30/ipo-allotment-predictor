import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import import_sw_pe_excel as imp


def test_build_sw_pe_frame_maps_known_sw_names_and_keeps_latest_date():
    raw = pd.DataFrame({
        "板块": ["SW电子", "SW通信", "SW综合", None, "数据来源：Wind"],
        "市盈率(TTM,整体法)\n[交易日期] 最新收盘日\n[剔除规则] 不调整": [105.46, 36.45, -73.5, None, None],
    })

    out = imp.build_sw_pe_frame(raw, "20260529")

    assert list(out["sw_level1_industry_name"]) == ["电子", "通信"]
    assert list(out["sw_level1_industry_code"]) == ["1000042193000000", "1000042215000000"]
    assert list(out["trade_date"].dt.strftime("%Y%m%d").unique()) == ["20260529"]
    assert out.loc[out["sw_level1_industry_name"] == "电子", "pe"].iloc[0] == 105.46


def test_merge_sw_pe_frame_replaces_same_date_and_industry():
    existing = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-05-28", "2026-05-29"]),
        "sw_level1_industry_code": ["1000042193000000", "1000042193000000"],
        "sw_level1_industry_name": ["电子", "电子"],
        "tushare_index_code": [None, None],
        "turnover_100m_yuan": [None, None],
        "return_pct": [None, None],
        "pe": [100.0, 101.0],
        "pb": [None, None],
    })
    new = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-05-29"]),
        "sw_level1_industry_code": ["1000042193000000"],
        "sw_level1_industry_name": ["电子"],
        "tushare_index_code": [None],
        "turnover_100m_yuan": [None],
        "return_pct": [None],
        "pe": [105.46],
        "pb": [None],
    })

    out = imp.merge_sw_pe_frame(existing, new)

    assert len(out) == 2
    assert out.loc[out["trade_date"].dt.strftime("%Y%m%d") == "20260529", "pe"].iloc[0] == 105.46
