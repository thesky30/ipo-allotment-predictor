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


def test_sw_level1_industry_name_mapping():
    from reference_data import sw_level1_industry_name

    assert sw_level1_industry_name("1000042211000000") == "机械设备"
    assert sw_level1_industry_name("1000042199000000") == "医药生物"
    assert sw_level1_industry_name("unknown") == "unknown"
