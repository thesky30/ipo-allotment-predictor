"""Convert a Wind daily market export → data/processed/market_daily.csv.

The IPO-level features `market_turnover_ma20` and `market_return_ma20`
(see scripts/initial_data_analysis.add_features) need a daily market series.
Tushare's index_daily requires paid credits, so we source it from Wind instead.

Expected Wind export (one sheet, daily / 按交易日, 2019-01-01 .. 2026-06-01):
    日期 + 成交额(amt) + 涨跌幅(pct_chg)   ← 推荐：万得全A 881001.WI
    （涨跌幅给收盘价 close 也可以，本脚本两种都认）

The reader is deliberately tolerant of real Wind quirks:
  - multi-row headers (e.g. a 单位/字段 row above a 日期/全部A股 row),
  - a "数据来源：Wind" footer and trailing all-NaN / zero-turnover rows,
  - dates stored as Excel serials (style strip drops the date cell format),
  - turnover in 元 / 万元 / 亿元 (auto-scaled to 亿元 by magnitude).

It scans the raw grid (no assumed header), locates the date column by Excel-serial
range, maps turnover / return / close columns by header keywords (searched across
all header rows) with a numeric-magnitude fallback, prints what it picked, and
writes a source-neutral CSV:
    trade_date, total_turnover_100m, mkt_close

If only a daily pct_chg series is available, a synthetic price index is
reconstructed (cumprod of 1+r) so the downstream close-ratio formula yields the
exact 20-trading-day compounded return.

Usage:
    python scripts/convert_wind_market.py --in "D:/wind导出数据/市场日度数据.xlsx"
    python scripts/convert_wind_market.py --in <file> --turnover-col 1 --return-col 2
        (column overrides accept a 0-based integer index)
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_PATH = DATA_DIR / "market_daily.csv"

TURNOVER_KEYS = ["成交额", "成交金额", "amt", "amount", "turnover"]
RETURN_KEYS = ["涨跌幅", "pct", "chg", "change", "return", "涨幅"]
CLOSE_KEYS = ["收盘", "close"]

SERIAL_LO, SERIAL_HI = 20000, 80000  # Excel serials: 1954 .. 2089


def _strip_xlsx_styles(src: Path, dst: Path) -> None:
    """Copy an xlsx while removing the malformed styles part (Wind quirk)."""
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "xl/styles.xml":
                continue
            zout.writestr(item, zin.read(item.filename))


def _read_grid(path: Path) -> pd.DataFrame:
    """Read the first sheet as a raw grid (no header assumptions)."""
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        with TemporaryDirectory() as td:
            safe = Path(td) / f"{path.stem}_nostyles.xlsx"
            try:
                _strip_xlsx_styles(path, safe)
                return pd.read_excel(safe, sheet_name=0, header=None)
            except Exception:
                return pd.read_excel(path, sheet_name=0, header=None)
    if path.suffix.lower() == ".xls":
        return pd.read_excel(path, sheet_name=0, header=None)
    return pd.read_csv(path, header=None)


def _serial_to_dt(num: pd.Series) -> pd.Series:
    return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")


def _date_score(s: pd.Series) -> float:
    """Fraction of cells that look like a trading date (Excel serial or real dt)."""
    num = pd.to_numeric(s, errors="coerce")
    serial_frac = num.between(SERIAL_LO, SERIAL_HI).mean()
    real_dt_frac = 0.0
    if not pd.api.types.is_numeric_dtype(s):
        real_dt_frac = pd.to_datetime(s, errors="coerce").notna().mean()
    return float(max(serial_frac, real_dt_frac))


def _parse_date_column(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.between(SERIAL_LO, SERIAL_HI).mean() >= 0.5:
        return _serial_to_dt(num.where(num.between(SERIAL_LO, SERIAL_HI)))
    return pd.to_datetime(s, errors="coerce")


def _resolve_override(val: str | None, ncols: int) -> int | None:
    if val is None:
        return None
    try:
        idx = int(val)
        return idx if 0 <= idx < ncols else None
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Wind market export to market_daily.csv")
    p.add_argument("--in", dest="inp", required=True, help="path to the Wind export (xlsx/xls/csv)")
    p.add_argument("--out", default=str(OUT_PATH), help="output CSV path")
    p.add_argument("--date-col", default=None, help="override date column (0-based index)")
    p.add_argument("--turnover-col", default=None, help="override turnover column (0-based index)")
    p.add_argument("--return-col", default=None, help="override return column (0-based index)")
    p.add_argument("--close-col", default=None, help="override close column (0-based index)")
    args = p.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    grid = _read_grid(src)
    grid = grid.dropna(axis=1, how="all")
    grid.columns = range(grid.shape[1])  # positional integer columns
    ncols = grid.shape[1]
    print(f"Grid shape: {grid.shape}")

    # ── 1. Date column: highest serial/datetime score ───────────────────────
    date_idx = _resolve_override(args.date_col, ncols)
    if date_idx is None:
        scores = {c: _date_score(grid[c]) for c in grid.columns}
        date_idx = max(scores, key=scores.get)
        if scores[date_idx] < 0.5:
            print(f"ERROR: no column looks like dates (scores={scores}); "
                  f"pass --date-col <idx>.", file=sys.stderr)
            sys.exit(1)
    date_series = _parse_date_column(grid[date_idx])

    # First real data row = first parseable date; rows above it are headers.
    valid = date_series.notna()
    if not valid.any():
        print("ERROR: date column parsed to all-NaT.", file=sys.stderr)
        sys.exit(1)
    data_start = int(np.argmax(valid.values))

    # ── 2. Header text per column (rows above data_start) ────────────────────
    header_txt: dict[int, str] = {}
    head_block = grid.iloc[:data_start]
    for c in grid.columns:
        parts = [str(v) for v in head_block[c].tolist() if isinstance(v, str)]
        header_txt[c] = " ".join(parts).lower()

    def match_by_header(keys: list[str]) -> int | None:
        for c in grid.columns:
            if c == date_idx:
                continue
            if any(k.lower() in header_txt[c] for k in keys):
                return c
        return None

    turn_idx = _resolve_override(args.turnover_col, ncols)
    ret_idx = _resolve_override(args.return_col, ncols)
    close_idx = _resolve_override(args.close_col, ncols)
    if turn_idx is None:
        turn_idx = match_by_header(TURNOVER_KEYS)
    if ret_idx is None:
        ret_idx = match_by_header(RETURN_KEYS)
    if close_idx is None:
        close_idx = match_by_header(CLOSE_KEYS)

    # Magnitude fallback among non-date numeric columns:
    #   turnover = large-magnitude column, return = small-magnitude column.
    if turn_idx is None or (ret_idx is None and close_idx is None):
        other = [c for c in grid.columns if c != date_idx]
        med = {}
        for c in other:
            v = pd.to_numeric(grid[c].iloc[data_start:], errors="coerce").abs()
            med[c] = v.median()
        ranked = sorted((c for c in other if pd.notna(med[c])), key=lambda c: med[c], reverse=True)
        if turn_idx is None and ranked:
            turn_idx = ranked[0]
        if ret_idx is None and close_idx is None and len(ranked) > 1:
            ret_idx = ranked[-1]

    if turn_idx is None:
        print("ERROR: could not find a turnover column; pass --turnover-col <idx>.",
              file=sys.stderr)
        sys.exit(1)
    if ret_idx is None and close_idx is None:
        print("ERROR: need a return (涨跌幅) or close (收盘价) column; "
              "pass --return-col / --close-col <idx>.", file=sys.stderr)
        sys.exit(1)

    print(f"  date col   -> #{date_idx}  '{header_txt.get(date_idx,'').strip()[:40]}'")
    print(f"  turnover   -> #{turn_idx}  '{header_txt.get(turn_idx,'').strip()[:40]}'")
    print(f"  return     -> #{ret_idx}" if ret_idx is not None else "  return     -> (none)")
    print(f"  close      -> #{close_idx}" if close_idx is not None else "  close      -> (none)")

    # ── 3. Assemble & clean ─────────────────────────────────────────────────
    out = pd.DataFrame({
        "trade_date": date_series,
        "_turn": pd.to_numeric(grid[turn_idx], errors="coerce"),
    })
    if close_idx is not None:
        out["_ret_src"] = pd.to_numeric(grid[close_idx], errors="coerce")
        is_close = True
    else:
        out["_ret_src"] = pd.to_numeric(grid[ret_idx], errors="coerce")
        is_close = False

    out = out[out["trade_date"].notna()].copy()
    out = out[out["_turn"] > 0].copy()  # drop footer / placeholder zero-turnover rows
    out = out.sort_values("trade_date").reset_index(drop=True)

    # Auto-scale turnover to 亿元 (1 亿 = 1e8 元).
    med_turn = out["_turn"].median()
    if med_turn > 1e11:
        scale, unit = 1e8, "元"
    elif med_turn > 1e7:
        scale, unit = 1e4, "万元"
    else:
        scale, unit = 1.0, "亿元(原值)"
    out["total_turnover_100m"] = out["_turn"] / scale
    print(f"  turnover median={med_turn:,.2f} → unit '{unit}', scaled to 亿元")

    # mkt_close: real close, or synthetic index from daily pct_chg.
    if is_close:
        out["mkt_close"] = out["_ret_src"]
        ret_basis = "real close"
    else:
        frac = out["_ret_src"].fillna(0) / 100.0  # pct → fraction
        out["mkt_close"] = (1.0 + frac).cumprod() * 1000.0
        ret_basis = "synthetic index from pct_chg"

    out = out[["trade_date", "total_turnover_100m", "mkt_close"]].dropna(
        subset=["total_turnover_100m"]
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out)} trading days → {args.out}")
    print(f"  date range : {out['trade_date'].min().date()} .. {out['trade_date'].max().date()}")
    print(f"  avg turnover: {out['total_turnover_100m'].mean():,.0f} 亿元/日")
    print(f"  return basis: {ret_basis}")


if __name__ == "__main__":
    main()
