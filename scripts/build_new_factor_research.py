"""Build candidate pre-inquiry factors from newly added data and evaluate IC.

This script prepares factors for the new prediction node: before initial
inquiry starts. It does not retrain production models.

Inputs from SQLite:
  - ipo_offline_sample
  - company_factor_data
  - board_market_daily

Outputs:
  - outputs/new_factor_research/factor_panel.csv
  - outputs/new_factor_research/factor_ic.csv
  - outputs/new_factor_research/factor_group_returns.csv
  - outputs/new_factor_research/new_factor_research_report.md

Note: sw_level1_market_daily is intentionally not merged yet because the
company factor file currently contains SW industry codes while market data
contains SW industry names. A code-name mapping is needed before attaching
industry market factors.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"
OUT_DIR = ROOT / "outputs" / "new_factor_research"
TARGET = "log_offline_oversubscription"


FACTOR_META = {
    "offline_market_value_threshold_10k_yuan": ("申购规则", "网下询价市值门槛，询价前可得"),
    "expected_fundraising_100m_yuan": ("发行规模", "首发预计募集资金，询价前候选"),
    "log_expected_fundraising": ("发行规模", "log1p(预计募资额)"),
    "industry_pe_at_ipo_factor": ("估值", "首发时所属行业市盈率，询价前候选"),
    "comparable_pe_avg_ex_nonrecurring_factor": ("估值", "可比上市公司PE均值，询价前候选"),
    "latest_revenue_100m_yuan": ("公司规模", "近一年营收额"),
    "log_latest_revenue": ("公司规模", "log1p(近一年营收额)"),
    "revenue_cagr_3y_pct": ("成长", "近三年营收复合增长率，覆盖率较低"),
    "issue_pb_factor": ("估值", "发行市净率，需确认是否依赖最终发行价"),
    "board_turnover_ma20": ("板块流动性", "同板块成交额20日均值，严格取询价开始日前一交易日"),
    "board_turnover_pct_rank_1y": ("板块流动性", "同板块成交额20日均值一年分位"),
    "board_turnover_ma20_over_ma60": ("板块流动性", "同板块成交额20/60日均线比-1"),
    "board_return_ma20": ("板块情绪", "同板块近20交易日涨跌幅"),
    "underwriter_prior_ipo_count": ("承销商声誉", "主承销商历史已发生IPO数量"),
    "underwriter_prior_log_oversub_mean": ("承销商声誉", "主承销商历史平均log网下超额认购"),
    "underwriter_prior_first_day_return_mean": ("承销商声誉", "主承销商历史平均首日涨幅"),
    "underwriter_prior_break_rate": ("承销商声誉", "主承销商历史破发率"),
    "sw_l1_prior_ipo_count": ("行业历史热度", "同申万一级代码历史IPO数量"),
    "sw_l1_prior_log_oversub_mean": ("行业历史热度", "同申万一级代码历史平均log网下超额认购"),
    "sw_l1_prior_first_day_return_mean": ("行业历史热度", "同申万一级代码历史平均首日涨幅"),
    "sw_l1_prior_break_rate": ("行业历史热度", "同申万一级代码历史破发率"),
}

FACTOR_LIST = list(FACTOR_META)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(DB_PATH) as conn:
        ipo = pd.read_sql("SELECT * FROM ipo_offline_sample WHERE security_code IS NOT NULL", conn)
        company = pd.read_sql("SELECT * FROM company_factor_data", conn)
        board_market = pd.read_sql("SELECT * FROM board_market_daily", conn)

    for col in ["listing_date", "inquiry_deadline_date", "subscription_deadline_date"]:
        ipo[col] = pd.to_datetime(ipo[col], errors="coerce")
    for col in ["inquiry_announcement_date", "inquiry_start_date"]:
        company[col] = pd.to_datetime(company[col], errors="coerce")
    board_market["trade_date"] = pd.to_datetime(board_market["trade_date"], errors="coerce")

    ipo[TARGET] = pd.to_numeric(ipo[TARGET], errors="coerce")
    ipo["first_day_return_pct"] = pd.to_numeric(ipo["first_day_return_pct"], errors="coerce")
    return ipo, company, board_market


def primary_underwriter(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    parts = re.split(r"[,，;；、]+", text)
    return parts[0].strip() or pd.NA


def add_board_market_factors(panel: pd.DataFrame, board_market: pd.DataFrame) -> pd.DataFrame:
    bm = board_market.copy()
    bm = bm.sort_values(["board", "trade_date"]).reset_index(drop=True)
    inactive = pd.to_numeric(bm["turnover_100m_yuan"], errors="coerce") <= 0
    bm.loc[inactive, ["turnover_100m_yuan", "return_pct"]] = np.nan
    bm["board_turnover_ma20"] = bm.groupby("board")["turnover_100m_yuan"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    bm["board_turnover_ma60"] = bm.groupby("board")["turnover_100m_yuan"].transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    bm["board_turnover_ma20_over_ma60"] = bm["board_turnover_ma20"] / bm["board_turnover_ma60"] - 1.0
    bm["board_turnover_pct_rank_1y"] = bm.groupby("board")["board_turnover_ma20"].transform(
        lambda s: s.rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    bm["board_return_ma20"] = bm.groupby("board")["return_pct"].transform(
        lambda s: s.rolling(20, min_periods=10).sum()
    )
    bm = bm[[
        "trade_date",
        "board",
        "board_turnover_ma20",
        "board_turnover_pct_rank_1y",
        "board_turnover_ma20_over_ma60",
        "board_return_ma20",
    ]].dropna(subset=["trade_date", "board"])

    frames = []
    for board, group in panel.groupby("board", dropna=False):
        missing_date = group[group["prediction_date"].isna()].copy()
        left = group[group["prediction_date"].notna()].sort_values("prediction_date")
        right = bm[bm["board"] == board].sort_values("trade_date")
        if left.empty:
            frames.append(group)
            continue
        if right.empty:
            frames.append(left)
            if not missing_date.empty:
                frames.append(missing_date)
            continue
        merged = pd.merge_asof(
            left,
            right,
            left_on="prediction_date",
            right_on="trade_date",
            direction="backward",
            allow_exact_matches=False,
        )
        if "board_y" in merged.columns:
            merged = merged.drop(columns=["board_y"]).rename(columns={"board_x": "board"})
        frames.append(merged)
        if not missing_date.empty:
            frames.append(missing_date)

    return pd.concat(frames, ignore_index=True)


def prior_group_stats(panel: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    out = panel.sort_values("prediction_date").copy()
    count_col = f"{prefix}_prior_ipo_count"
    log_col = f"{prefix}_prior_log_oversub_mean"
    ret_col = f"{prefix}_prior_first_day_return_mean"
    break_col = f"{prefix}_prior_break_rate"
    out[[count_col, log_col, ret_col, break_col]] = np.nan

    histories: dict[object, list[tuple[float, float]]] = {}
    for idx, row in out.iterrows():
        key = row.get(key_col)
        hist = histories.get(key, []) if pd.notna(key) else []
        if hist:
            arr = pd.DataFrame(hist, columns=[TARGET, "first_day_return_pct"])
            out.at[idx, count_col] = len(arr)
            out.at[idx, log_col] = arr[TARGET].mean()
            out.at[idx, ret_col] = arr["first_day_return_pct"].mean()
            out.at[idx, break_col] = (arr["first_day_return_pct"] < 0).mean()
        else:
            out.at[idx, count_col] = 0

        if pd.notna(key) and pd.notna(row.get(TARGET)):
            histories.setdefault(key, []).append((row[TARGET], row.get("first_day_return_pct", np.nan)))

    return out


def build_panel() -> pd.DataFrame:
    ipo, company, board_market = load_tables()
    panel = ipo.merge(company, on="security_code", how="left", validate="many_to_one")

    panel["prediction_date"] = (
        panel["inquiry_start_date"]
        .fillna(panel["inquiry_announcement_date"])
        .fillna(panel["inquiry_deadline_date"])
        .fillna(panel["subscription_deadline_date"])
        .fillna(panel["listing_date"])
    )
    panel["prediction_date_source"] = np.select(
        [
            panel["inquiry_start_date"].notna(),
            panel["inquiry_announcement_date"].notna(),
            panel["inquiry_deadline_date"].notna(),
            panel["subscription_deadline_date"].notna(),
        ],
        ["inquiry_start_date", "inquiry_announcement_date", "inquiry_deadline_date", "subscription_deadline_date"],
        default="listing_date",
    )

    panel["log_expected_fundraising"] = np.log1p(pd.to_numeric(panel["expected_fundraising_100m_yuan"], errors="coerce"))
    panel["log_latest_revenue"] = np.log1p(pd.to_numeric(panel["latest_revenue_100m_yuan"], errors="coerce"))
    panel["primary_underwriter"] = panel["lead_underwriter"].map(primary_underwriter)

    panel = add_board_market_factors(panel, board_market)
    panel = prior_group_stats(panel, "primary_underwriter", "underwriter")
    panel = prior_group_stats(panel, "sw_level1_industry_code", "sw_l1")

    cols = [
        "security_code",
        "security_name",
        "board",
        "listing_date",
        "prediction_date",
        "prediction_date_source",
        TARGET,
        "first_day_return_pct",
        "inquiry_announcement_date",
        "inquiry_start_date",
        "sw_level1_industry_code",
        "sw_level2_industry_code",
        "primary_underwriter",
        *FACTOR_LIST,
    ]
    existing = [c for c in cols if c in panel.columns]
    return panel[existing].sort_values(["prediction_date", "security_code"]).reset_index(drop=True)


def spearman_ic(x: pd.Series, y: pd.Series) -> tuple[int, float, str]:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 20:
        return len(d), np.nan, "too_few"
    if d["x"].nunique() < 3 or d["y"].nunique() < 3:
        return len(d), np.nan, "low_variance"
    xr = d["x"].rank(method="average")
    yr = d["y"].rank(method="average")
    return len(d), float(xr.corr(yr)), "ok"


def build_ic(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_LIST:
        if factor not in panel.columns:
            continue
        group, note = FACTOR_META[factor]
        n, ic, status = spearman_ic(panel[factor], panel[TARGET])
        rows.append({
            "scope": "all",
            "factor": factor,
            "factor_group": group,
            "n": n,
            "spearman_ic": ic,
            "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
            "status": status,
            "note": note,
        })
        for board, g in panel.groupby("board", dropna=False):
            n_b, ic_b, status_b = spearman_ic(g[factor], g[TARGET])
            rows.append({
                "scope": f"board:{board}",
                "factor": factor,
                "factor_group": group,
                "n": n_b,
                "spearman_ic": ic_b,
                "abs_ic": abs(ic_b) if pd.notna(ic_b) else np.nan,
                "status": status_b,
                "note": note,
            })
    return pd.DataFrame(rows).sort_values(["scope", "abs_ic"], ascending=[True, False])


def build_groups(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_LIST:
        d = panel[["board", factor, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 50 or d[factor].nunique() < 5:
            continue
        try:
            d["bucket"] = pd.qcut(d[factor].rank(method="first"), 5, labels=False) + 1
        except ValueError:
            continue
        for bucket, g in d.groupby("bucket"):
            rows.append({
                "factor": factor,
                "factor_group": FACTOR_META[factor][0],
                "bucket": int(bucket),
                "n": len(g),
                "factor_min": g[factor].min(),
                "factor_max": g[factor].max(),
                "factor_median": g[factor].median(),
                "target_mean": g[TARGET].mean(),
                "target_median": g[TARGET].median(),
            })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 3) -> str:
    if df.empty:
        return "_无数据_"
    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        else:
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(panel: pd.DataFrame, ic: pd.DataFrame, groups: pd.DataFrame) -> None:
    label_n = int(panel[TARGET].notna().sum())
    strict_date_n = int((panel["prediction_date_source"] == "inquiry_start_date").sum())
    top = (
        ic[(ic["scope"] == "all") & (ic["status"] == "ok")]
        .sort_values("abs_ic", ascending=False)
        [["factor", "factor_group", "n", "spearman_ic", "note"]]
        .head(15)
    )
    coverage = []
    for factor in FACTOR_LIST:
        if factor in panel.columns:
            n = int(panel[factor].notna().sum())
            coverage.append({
                "factor": factor,
                "factor_group": FACTOR_META[factor][0],
                "non_null": n,
                "coverage": n / len(panel),
            })
    cov = pd.DataFrame(coverage).sort_values("coverage", ascending=False)

    text = f"""# 新增询价前候选因子 IC 报告

## 样本

- 因子面板行数：{len(panel)}
- 有训练标签行数：{label_n}
- 使用 `inquiry_start_date` 作为预测日的行数：{strict_date_n}
- 目标变量：`log_offline_oversubscription`

## 全样本 IC Top 15

IC 为候选因子与未来 `log(网下超额认购倍数)` 的 Spearman 排序相关。
正 IC 表示因子越高，未来超额认购通常越高，中签率通常越低。

{md_table(top, max_rows=15)}

## 因子覆盖率

{md_table(cov, max_rows=30)}

## 重要限制

- 北交所缺板块行情，因此 `board_*` 因子在北交所为空。
- 申万一级行情表是行业名称，上市公司因子表是行业代码；缺少代码-名称映射，所以暂未生成行业行情滚动因子。
- `issue_pb_factor`、`industry_pe_at_ipo_factor` 等估值字段虽然已列入候选，但入模前仍需确认是否在询价开始前已经公开。
- 本报告是单因子筛选，不代表多因子模型最终权重；下一步应把通过筛选的因子加入滚动回测模型比较。
"""
    (OUT_DIR / "new_factor_research_report.md").write_text(text, encoding="utf-8")

    selected = top["factor"].head(5).tolist()
    if selected and not groups.empty:
        group_text = ["# Top 因子五分组表现\n"]
        for factor in selected:
            g = groups[groups["factor"] == factor][[
                "factor", "bucket", "n", "factor_median", "target_mean", "target_median"
            ]]
            group_text.append(f"\n## {factor}\n\n{md_table(g, max_rows=10)}\n")
        (OUT_DIR / "new_factor_group_summary.md").write_text("\n".join(group_text), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    ic = build_ic(panel)
    groups = build_groups(panel)

    panel.to_csv(OUT_DIR / "factor_panel.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(OUT_DIR / "factor_ic.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT_DIR / "factor_group_returns.csv", index=False, encoding="utf-8-sig")
    with sqlite3.connect(DB_PATH) as conn:
        panel.to_sql("new_factor_panel", conn, if_exists="replace", index=False)
        ic.to_sql("new_factor_ic", conn, if_exists="replace", index=False)
        groups.to_sql("new_factor_group_returns", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_new_factor_panel_code ON new_factor_panel(security_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_new_factor_panel_pred_date ON new_factor_panel(prediction_date)")
    write_report(panel, ic, groups)

    ok = ic[(ic["scope"] == "all") & (ic["status"] == "ok")].sort_values("abs_ic", ascending=False)
    print(f"factor panel rows: {len(panel)}")
    print(f"label rows: {int(panel[TARGET].notna().sum())}")
    print("top IC:")
    print(ok[["factor", "n", "spearman_ic"]].head(10).to_string(index=False))
    print(f"wrote: {OUT_DIR / 'new_factor_research_report.md'}")


if __name__ == "__main__":
    main()
