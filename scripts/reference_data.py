"""Read-only loaders for the historical reference frames used to assemble
T-6 features for a brand-new IPO. No network, no Streamlit."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"


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
