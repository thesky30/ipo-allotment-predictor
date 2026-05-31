"""Read-only loaders for the historical reference frames used to assemble
T-6 features for a brand-new IPO. No network, no Streamlit."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"

SW_LEVEL1_INDUSTRY_NAME_BY_CODE = {
    "1000042189000000": "农林牧渔",
    "1000042190000000": "基础化工",
    "1000042191000000": "钢铁",
    "1000042192000000": "有色金属",
    "1000042193000000": "电子",
    "1000042194000000": "汽车",
    "1000042195000000": "家用电器",
    "1000042196000000": "食品饮料",
    "1000042197000000": "纺织服饰",
    "1000042198000000": "轻工制造",
    "1000042199000000": "医药生物",
    "1000042200000000": "公用事业",
    "1000042201000000": "交通运输",
    "1000042202000000": "房地产",
    "1000042203000000": "商贸零售",
    "1000042204000000": "社会服务",
    "1000042208000000": "建筑材料",
    "1000042209000000": "建筑装饰",
    "1000042210000000": "电力设备",
    "1000042211000000": "机械设备",
    "1000042212000000": "国防军工",
    "1000042213000000": "计算机",
    "1000042214000000": "传媒",
    "1000042215000000": "通信",
    "1000042216000000": "银行",
    "1000042217000000": "非银金融",
    "1000042218000000": "环保",
    "1000042219000000": "美容护理",
}


def sw_level1_industry_name(code: object) -> str:
    if code is None:
        return ""
    text = str(code)
    return SW_LEVEL1_INDUSTRY_NAME_BY_CODE.get(text, text)


@dataclass(frozen=True)
class History:
    sample: pd.DataFrame        # ipo_offline_sample (raw + market/concurrent/heat baked in)
    panel: pd.DataFrame         # new_factor_panel (board rolling + priors)
    board_market: pd.DataFrame  # board_market_daily


def load_history() -> History:
    with sqlite3.connect(DB_PATH) as conn:
        sample = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)
        panel = pd.read_sql("SELECT * FROM new_factor_panel", conn)
        board_market = pd.read_sql("SELECT * FROM board_market_daily", conn)
    for df, col in [(sample, "subscription_deadline_date"),
                    (sample, "listing_date"),
                    (panel, "prediction_date")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return History(sample=sample, panel=panel, board_market=board_market)


def data_as_of() -> pd.Timestamp:
    """Latest trading day available in market_daily (freshness boundary)."""
    from initial_data_analysis import load_market_daily
    md = load_market_daily()
    if md is None or md.empty:
        return pd.Timestamp.min
    return pd.Timestamp(md["trade_date"].max())
