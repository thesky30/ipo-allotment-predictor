import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import market_source


class FakePro:
    def index_daily(self, ts_code, start_date, end_date, fields):
        rows = {
            "000001.SH": [
                {"trade_date": "20260529", "amount": 100000.0, "close": 3100, "pct_chg": 1.1},
            ],
            "399106.SZ": [
                {"trade_date": "20260529", "amount": 200000.0, "close": 2100, "pct_chg": 1.2},
            ],
            "000300.SH": [
                {"trade_date": "20260529", "amount": 0.0, "close": 4000, "pct_chg": 0.5},
            ],
        }
        return pd.DataFrame(rows[ts_code])

    def index_classify(self, **kwargs):
        return pd.DataFrame([
            {"index_code": "801890.SI", "industry_name": "机械设备"},
            {"index_code": "801150.SI", "industry_name": "医药生物"},
        ])

    def sw_daily(self, ts_code, start_date, end_date, fields):
        return pd.DataFrame([
            {"ts_code": ts_code, "trade_date": "20260529", "amount": 123456.0, "pct_change": 1.5, "pe": 22.3, "pb": 2.1},
        ])


def test_fetch_market_daily_normalizes_turnover_units():
    df = market_source.fetch_market_daily(FakePro(), "20260501", "20260529")

    assert list(df.columns) == [
        "trade_date", "sse_amount_100m", "szse_amount_100m",
        "total_turnover_100m", "csi300_close", "csi300_pct_chg",
    ]
    assert df.loc[0, "total_turnover_100m"] == 3.0
    assert str(df.loc[0, "trade_date"].date()) == "2026-05-29"


def test_fetch_sw_level1_mapping_joins_tushare_index_codes_to_wind_codes():
    df = market_source.fetch_sw_level1_mapping(FakePro())

    by_code = df.set_index("sw_level1_industry_code")
    assert by_code.loc["1000042211000000", "sw_level1_industry_name"] == "机械设备"
    assert by_code.loc["1000042211000000", "tushare_index_code"] == "801890.SI"


def test_fetch_sw_level1_daily_adds_wind_code_and_pe():
    mapping = pd.DataFrame([
        {
            "sw_level1_industry_code": "1000042211000000",
            "sw_level1_industry_name": "机械设备",
            "tushare_index_code": "801890.SI",
        }
    ])

    df = market_source.fetch_sw_level1_daily(FakePro(), mapping, "20260501", "20260529")

    assert df.loc[0, "sw_level1_industry_code"] == "1000042211000000"
    assert df.loc[0, "sw_level1_industry_name"] == "机械设备"
    assert df.loc[0, "pe"] == 22.3
    assert df.loc[0, "turnover_100m_yuan"] == 12.3456


def test_choose_sw_daily_output_keeps_cache_when_refresh_empty():
    cached = pd.DataFrame([
        {
            "trade_date": pd.Timestamp("2026-05-29"),
            "sw_level1_industry_code": "1000042211000000",
            "sw_level1_industry_name": "机械设备",
            "tushare_index_code": "801890.SI",
            "turnover_100m_yuan": 12.3,
            "return_pct": 1.2,
            "pe": 22.3,
            "pb": 2.1,
        }
    ])
    fresh = pd.DataFrame(columns=cached.columns)

    out, used_cache = market_source.choose_sw_daily_output(fresh, cached)

    assert used_cache is True
    assert out.equals(cached)


def test_latest_sw_industry_pe_uses_latest_available_before_trade_date():
    sw_daily = pd.DataFrame([
        {"trade_date": "2026-05-28", "sw_level1_industry_code": "1000042211000000", "pe": 21.0},
        {"trade_date": "2026-05-29", "sw_level1_industry_code": "1000042211000000", "pe": 22.0},
        {"trade_date": "2026-05-30", "sw_level1_industry_code": "1000042211000000", "pe": 23.0},
    ])

    out = market_source.latest_sw_industry_pe(sw_daily, "1000042211000000", "20260529")

    assert out["pe"] == 22.0
    assert out["trade_date"] == "20260529"


def test_fetch_latest_sw_industry_pe_fetches_selected_industry_only():
    class FakeSingleIndustryPro:
        def index_classify(self, **kwargs):
            return pd.DataFrame([
                {"index_code": "801770.SI", "industry_name": "通信"},
            ])

        def sw_daily(self, ts_code, start_date, end_date, fields):
            assert ts_code == "801770.SI"
            assert end_date == "20260529"
            return pd.DataFrame([
                {"ts_code": ts_code, "trade_date": "20260528", "amount": 10000, "pct_change": 1.0, "pe": 31.0, "pb": 2.0},
                {"ts_code": ts_code, "trade_date": "20260529", "amount": 10000, "pct_change": 1.0, "pe": 32.0, "pb": 2.0},
            ])

    out = market_source.fetch_latest_sw_industry_pe(
        FakeSingleIndustryPro(), "1000042215000000", "20260529"
    )

    assert out["sw_level1_industry_name"] == "通信"
    assert out["pe"] == 32.0
