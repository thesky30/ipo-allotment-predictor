"""One-time historical download of market-level liquidity / sentiment data via Tushare.

Pulls daily index data needed to derive the T-6 market features used in
`initial_data_analysis.add_features()`:

    market_turnover_ma20   ← 沪深两市日成交额（亿元），由 上证综指(000001.SH)
                             + 深证综指(399106.SZ) 的 amount 求和得到
    market_return_ma20     ← 沪深300(000300.SH) 收盘价，用于算近20交易日涨跌幅

Output
------
data/processed/market_daily.csv  (one row per trading day)
    trade_date, sse_amount_100m, szse_amount_100m,
    total_turnover_100m, csi300_close, csi300_pct_chg

Token
-----
Tushare Pro 需要 token。优先级：
    1. --token 命令行参数
    2. 环境变量 TUSHARE_TOKEN
    3. tushare 已保存的 token（曾经 ts.set_token 过）

用法
----
    python scripts/fetch_market_data.py --token <YOUR_TUSHARE_TOKEN>
    # 或先 set TUSHARE_TOKEN，再
    python scripts/fetch_market_data.py

Tushare 单位说明
----------------
index_daily 的 amount 单位是「千元」。1 亿元 = 1e5 千元，故 亿元 = amount / 1e5。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_PATH = DATA_DIR / "market_daily.csv"

# Cover from before the earliest subscription deadline (2019-06-27) so that
# the 20-trading-day look-back window is fully populated, through a margin
# past the latest listing date.
START_DATE = "20190101"
END_DATE = "20260601"

# Whole-market turnover proxies (composite indices cover ALL listed stocks,
# unlike 成份指数 which only cover constituents).
SSE_COMPOSITE = "000001.SH"   # 上证综指 → SSE total turnover (incl. 科创板)
SZSE_COMPOSITE = "399106.SZ"  # 深证综指 → SZSE total turnover (incl. 创业板)
CSI300 = "000300.SH"          # 沪深300 → market sentiment


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip()
    env = os.environ.get("TUSHARE_TOKEN")
    if env:
        return env.strip()
    return None  # fall back to ts saved token


def fetch_index_daily(pro, ts_code: str, fields: str) -> pd.DataFrame:
    """index_daily for one ts_code over the full window. Retries on transient errors."""
    last_err = None
    for attempt in range(3):
        try:
            df = pro.index_daily(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
                fields=fields,
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty response for {ts_code}")
            return df
        except Exception as e:  # noqa: BLE001 - surface tushare/network errors after retries
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {ts_code}: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch market-level daily data via Tushare")
    parser.add_argument("--token", default=None, help="Tushare Pro token (overrides env/saved)")
    parser.add_argument("--out", default=str(OUT_PATH), help="output CSV path")
    args = parser.parse_args()

    try:
        import tushare as ts
    except ImportError:
        print("ERROR: tushare not installed. Run: pip install tushare", file=sys.stderr)
        sys.exit(1)

    token = resolve_token(args.token)
    if token:
        ts.set_token(token)
    try:
        pro = ts.pro_api(token) if token else ts.pro_api()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot init Tushare Pro API: {e}\n"
              f"Provide a token via --token or TUSHARE_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching index_daily {START_DATE}–{END_DATE} ...")
    sse = fetch_index_daily(pro, SSE_COMPOSITE, "trade_date,amount")
    szse = fetch_index_daily(pro, SZSE_COMPOSITE, "trade_date,amount")
    csi = fetch_index_daily(pro, CSI300, "trade_date,close,pct_chg")

    # 千元 → 亿元
    sse = sse.rename(columns={"amount": "sse_amount_k"})
    szse = szse.rename(columns={"amount": "szse_amount_k"})
    sse["sse_amount_100m"] = sse["sse_amount_k"] / 1e5
    szse["szse_amount_100m"] = szse["szse_amount_k"] / 1e5
    csi = csi.rename(columns={"close": "csi300_close", "pct_chg": "csi300_pct_chg"})

    out = (
        sse[["trade_date", "sse_amount_100m"]]
        .merge(szse[["trade_date", "szse_amount_100m"]], on="trade_date", how="outer")
        .merge(csi[["trade_date", "csi300_close", "csi300_pct_chg"]], on="trade_date", how="outer")
    )
    out["total_turnover_100m"] = out["sse_amount_100m"].fillna(0) + out["szse_amount_100m"].fillna(0)
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    out = out.sort_values("trade_date").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} trading days → {out_path}")
    print(f"  date range: {out['trade_date'].min().date()} .. {out['trade_date'].max().date()}")
    print(f"  avg total turnover: {out['total_turnover_100m'].mean():.0f} 亿元/日")


if __name__ == "__main__":
    main()
