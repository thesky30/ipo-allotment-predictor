"""T-6 factor insight report for IPO offline subscription prediction.

This script is intentionally report-oriented: it does not retrain models or
mutate the source database. It reads the cleaned SQLite table plus the saved
T-6 LightGBM model, then writes leadership-friendly factor diagnostics to
outputs/factor_insights/.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DB_PATH = DATA_DIR / "ipo_offline.db"
MARKET_DAILY_PATH = DATA_DIR / "market_daily.csv"
MODEL_PATH = ROOT / "outputs" / "baseline_models" / "models" / "lgbm_t6.joblib"
OUT_DIR = ROOT / "outputs" / "factor_insights"
FIG_DIR = OUT_DIR / "figures"
TARGET = "log_offline_oversubscription"

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baseline_models import FEATURE_NODES, FEATS_T6  # noqa: E402
from model_classes import BoardMeanModel, _ColSelector  # noqa: E402,F401


PERIODS = [
    ("2019-2020 注册制早期", 2019, 2020),
    ("2021-2022 扩容与破发阶段", 2021, 2022),
    ("2023 全面注册制切换", 2023, 2023),
    ("2024-2026 当前阶段", 2024, 2026),
]


FACTOR_META: dict[str, dict[str, str]] = {
    "market_turnover_ma20": {
        "group": "市场流动性因子",
        "source": "market_daily.csv 向后看滚动20日均值",
        "window": "20 trading days",
        "backward_looking": "yes",
        "expected_direction": "流动性越高，资金越活跃，可能推高超额认购、压低中签率",
    },
    "market_turnover_pct_rank_1y": {
        "group": "市场流动性因子",
        "source": "market_daily.csv 成交额20日均值的一年滚动分位",
        "window": "252 trading days",
        "backward_looking": "yes",
        "expected_direction": "高分位流动性可能对应更拥挤的网下申购",
    },
    "market_turnover_ma20_over_ma60": {
        "group": "市场流动性因子",
        "source": "market_daily.csv 成交额20日均值/60日均值-1",
        "window": "20 vs 60 trading days",
        "backward_looking": "yes",
        "expected_direction": "短期成交额放大可能代表情绪升温和资金活跃",
    },
    "market_return_ma20": {
        "group": "市场情绪因子",
        "source": "market_daily.csv 市场指数20日涨跌幅",
        "window": "20 trading days",
        "backward_looking": "yes",
        "expected_direction": "市场上涨可能增强申购热度",
    },
    "recent_ipo_first_day_return_ma20": {
        "group": "市场情绪因子",
        "source": "同板块过去20只已上市IPO首日涨幅",
        "window": "20 prior IPOs",
        "backward_looking": "yes",
        "expected_direction": "新股赚钱效应越强，申购越拥挤",
    },
    "same_board_break_rate_ma10": {
        "group": "市场情绪因子",
        "source": "同板块过去10只已上市IPO破发比例",
        "window": "10 prior IPOs",
        "backward_looking": "yes",
        "expected_direction": "破发率越高，申购意愿可能下降",
    },
    "concurrent_ipo_count": {
        "group": "IPO供给拥挤因子",
        "source": "申购日历±7天全市场IPO数量",
        "window": "±7 calendar days",
        "backward_looking": "calendar_known",
        "expected_direction": "IPO越拥挤，可能分流资金；也可能代表热发行窗口",
    },
    "same_board_concurrent_ipo_count": {
        "group": "IPO供给拥挤因子",
        "source": "申购日历±7天同板块IPO数量",
        "window": "±7 calendar days",
        "backward_looking": "calendar_known",
        "expected_direction": "同板块供给拥挤可能分流同类资金",
    },
    "concurrent_offline_issue_sum_10k": {
        "group": "IPO供给拥挤因子",
        "source": "申购日历±7天其他IPO回拨前网下发行量合计",
        "window": "±7 calendar days",
        "backward_looking": "calendar_known",
        "expected_direction": "同窗口可申购供给越大，单只拥挤度可能缓和",
    },
    "total_issue_shares_10k": {
        "group": "发行供给因子",
        "source": "招股书/发行安排",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "发行规模越大，供给越多，中签率可能更高",
    },
    "offline_issue_before_clawback_10k": {
        "group": "发行供给因子",
        "source": "发行安排公告",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "网下初始发行量越大，中签率可能更高",
    },
    "offline_issue_before_share_pct": {
        "group": "发行供给因子",
        "source": "发行安排派生",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "网下初始占比越高，供给越充分",
    },
    "strategic_allocation_share_pct": {
        "group": "发行供给因子",
        "source": "发行安排派生",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "战略配售占比高可能压缩可网下分配供给",
    },
    "subscription_upper_limit_10k": {
        "group": "申购规则因子",
        "source": "询价及推介公告",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "申购上限影响机构可申购规模和拥挤度",
    },
    "subscription_lower_limit_10k": {
        "group": "申购规则因子",
        "source": "询价及推介公告",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "申购下限反映参与门槛",
    },
    "subscription_step_10k": {
        "group": "申购规则因子",
        "source": "询价及推介公告",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "申购步长影响报价/申购离散度",
    },
    "industry_pe_at_ipo": {
        "group": "估值因子",
        "source": "发行时行业市盈率",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "行业估值环境影响新股吸引力",
    },
    "comparable_pe_avg_ex_nonrecurring": {
        "group": "估值因子",
        "source": "招股书可比公司估值",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "可比估值影响询价前吸引力判断",
    },
    "offer_price_range_pct": {
        "group": "估值因子",
        "source": "询价公告价格区间派生",
        "window": "static",
        "backward_looking": "yes",
        "expected_direction": "询价区间宽度反映定价不确定性",
    },
}

FACTOR_LIST = list(FACTOR_META)


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start",
             weight: str = "400", color: str = "#333") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,Microsoft YaHei,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">'
        f"{escape(str(text))}</text>"
    )


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.write_text(
        "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            *body,
            "</svg>",
        ]),
        encoding="utf-8",
    )


def fmt_num(v: object, digits: int = 2) -> str:
    if pd.isna(v):
        return ""
    return f"{float(v):,.{digits}f}"


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


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)
    for c in ["listing_date", "subscription_deadline_date", "inquiry_deadline_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["listing_year"] = pd.to_numeric(df["listing_year"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return ensure_derived_factors(df)


def load_market_daily() -> pd.DataFrame | None:
    if not MARKET_DAILY_PATH.exists():
        return None
    md = pd.read_csv(MARKET_DAILY_PATH)
    md["trade_date"] = pd.to_datetime(md["trade_date"], errors="coerce")
    md = md.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    close_col = "mkt_close" if "mkt_close" in md.columns else "csi300_close"
    md["turnover_ma20"] = md["total_turnover_100m"].rolling(20, min_periods=10).mean()
    md["turnover_ma60"] = md["total_turnover_100m"].rolling(60, min_periods=30).mean()
    md["turnover_ma20_over_ma60"] = md["turnover_ma20"] / md["turnover_ma60"] - 1.0
    md["turnover_pct_rank_252"] = (
        md["turnover_ma20"]
        .rolling(252, min_periods=60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    md["mkt_ret20"] = (md[close_col] / md[close_col].shift(20) - 1.0) * 100
    return md


def attach_market_if_missing(df: pd.DataFrame, ref_date: pd.Series) -> pd.DataFrame:
    needed = ["market_turnover_pct_rank_1y", "market_turnover_ma20_over_ma60"]
    if all(c in df.columns and df[c].notna().any() for c in needed):
        return df
    md = load_market_daily()
    for c in ["market_turnover_ma20", "market_turnover_pct_rank_1y",
              "market_turnover_ma20_over_ma60", "market_return_ma20"]:
        if c not in df.columns:
            df[c] = np.nan
    if md is None:
        return df
    dates = md["trade_date"].values.astype("datetime64[ns]")
    idx = np.searchsorted(dates, ref_date.values.astype("datetime64[ns]"), side="left") - 1
    ok = ref_date.notna().values & (idx >= 0)
    safe_idx = np.where(ok, idx, 0)
    df.loc[ok, "market_turnover_ma20"] = md["turnover_ma20"].values[safe_idx][ok]
    df.loc[ok, "market_turnover_pct_rank_1y"] = md["turnover_pct_rank_252"].values[safe_idx][ok]
    df.loc[ok, "market_turnover_ma20_over_ma60"] = md["turnover_ma20_over_ma60"].values[safe_idx][ok]
    df.loc[ok, "market_return_ma20"] = md["mkt_ret20"].values[safe_idx][ok]
    return df


def ensure_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ref_date = out["subscription_deadline_date"].fillna(out["listing_date"])
    out = attach_market_if_missing(out, ref_date)

    if "same_board_concurrent_ipo_count" not in out.columns:
        out["same_board_concurrent_ipo_count"] = np.nan
    if "concurrent_offline_issue_sum_10k" not in out.columns:
        out["concurrent_offline_issue_sum_10k"] = np.nan

    need_same = out["same_board_concurrent_ipo_count"].isna().all()
    need_sum = out["concurrent_offline_issue_sum_10k"].isna().all()
    if need_same or need_sum:
        sub = out["subscription_deadline_date"].reset_index(drop=True)
        board = out["board"].reset_index(drop=True)
        offline = pd.to_numeric(out["offline_issue_before_clawback_10k"], errors="coerce").reset_index(drop=True)
        win = pd.Timedelta(days=7)
        for pos in range(len(out)):
            d = sub.iloc[pos]
            if pd.isna(d):
                continue
            mask = sub.notna() & (sub >= d - win) & (sub <= d + win)
            same_mask = mask & (board == board.iloc[pos])
            mask.iloc[pos] = False
            same_mask.iloc[pos] = False
            if need_same:
                out.iloc[pos, out.columns.get_loc("same_board_concurrent_ipo_count")] = int(same_mask.sum())
            if need_sum:
                out.iloc[pos, out.columns.get_loc("concurrent_offline_issue_sum_10k")] = offline[mask].sum(min_count=1)

    out["period"] = out["listing_year"].map(period_label)
    return out


def period_label(year: float) -> str:
    if pd.isna(year):
        return "未知时期"
    y = int(year)
    for label, start, end in PERIODS:
        if start <= y <= end:
            return label
    return "其他时期"


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run scripts/baseline_models.py first.")
    return joblib.load(MODEL_PATH)


def predict_log(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    features = bundle["features"]
    row = df.copy()
    for col in features:
        if col not in row.columns:
            row[col] = np.nan
    return np.asarray(bundle["model"].predict(row), dtype=float)


def shap_long(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = bundle["model"]
    features = bundle["features"]
    row = df.copy()
    for col in features:
        if col not in row.columns:
            row[col] = np.nan

    pipe = getattr(model, "_pipe", None)
    avail = [c for c in getattr(model, "_avail", features) if c in row.columns]
    pre = pipe.named_steps["pre"]
    reg = pipe.named_steps["reg"]
    X = pre.transform(row[avail])
    feat_names = list(pre.get_feature_names_out())
    contrib = np.asarray(reg.booster_.predict(X, pred_contrib=True))
    shap_vals = contrib[:, :-1]
    base_vals = contrib[:, -1]

    agg: dict[str, np.ndarray] = {}
    for i, name in enumerate(feat_names):
        if name.startswith("num__"):
            key = name[len("num__"):]
        elif name.startswith("cat__"):
            key = "board"
        else:
            key = name
        agg[key] = agg.get(key, np.zeros(len(row))) + shap_vals[:, i]

    meta_cols = ["security_code", "security_name", "board", "listing_year", "period", TARGET]
    wide = row[[c for c in meta_cols if c in row.columns]].copy()
    wide["base_value"] = base_vals
    wide["predicted_log"] = base_vals + np.sum(np.column_stack(list(agg.values())), axis=1)
    for key, vals in agg.items():
        wide[f"shap__{key}"] = vals

    parts = []
    for key, vals in agg.items():
        tmp = wide[[c for c in meta_cols if c in wide.columns]].copy()
        tmp["feature"] = key
        tmp["shap"] = vals
        tmp["abs_shap"] = np.abs(vals)
        parts.append(tmp)
    return wide, pd.concat(parts, ignore_index=True)


def summarize_profile(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["subscription_rate_pct_calc"] = 100.0 / pd.to_numeric(d["offline_oversubscription_ratio"], errors="coerce")
    d["break_flag"] = pd.to_numeric(d["first_day_return_pct"], errors="coerce") < 0
    agg = (
        d.groupby(group_cols, dropna=False)
        .agg(
            sample_count=("security_code", "count"),
            label_count=(TARGET, lambda s: int(s.notna().sum())),
            median_oversubscription=("offline_oversubscription_ratio", "median"),
            median_subscription_rate_pct=("subscription_rate_pct_calc", "median"),
            median_first_day_return_pct=("first_day_return_pct", "median"),
            break_rate=("break_flag", "mean"),
            median_total_issue_shares_10k=("total_issue_shares_10k", "median"),
            median_offline_issue_before_10k=("offline_issue_before_clawback_10k", "median"),
            median_offline_issue_before_share_pct=("offline_issue_before_share_pct", "median"),
            median_strategic_allocation_share_pct=("strategic_allocation_share_pct", "median"),
            median_subscription_upper_limit_10k=("subscription_upper_limit_10k", "median"),
            median_subscription_lower_limit_10k=("subscription_lower_limit_10k", "median"),
            median_market_turnover_ma20=("market_turnover_ma20", "median"),
            median_recent_ipo_return_ma20=("recent_ipo_first_day_return_ma20", "median"),
            median_concurrent_ipo_count=("concurrent_ipo_count", "median"),
            median_same_board_break_rate_ma10=("same_board_break_rate_ma10", "median"),
        )
        .reset_index()
    )
    agg["label_coverage"] = agg["label_count"] / agg["sample_count"]
    return agg


def factor_dictionary() -> pd.DataFrame:
    rows = []
    for f, meta in FACTOR_META.items():
        node = FEATURE_NODES.get(f, {})
        rows.append({
            "factor": f,
            "factor_group": meta["group"],
            "time_node": node.get("node", "T-6"),
            "source": meta["source"],
            "category": node.get("cat", meta["group"]),
            "window": meta["window"],
            "backward_looking": meta["backward_looking"],
            "expected_direction": meta["expected_direction"],
        })
    return pd.DataFrame(rows)


def spearman_ic(x: pd.Series, y: pd.Series) -> tuple[int, float, str]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 20 or pair.iloc[:, 0].nunique() < 2:
        return len(pair), np.nan, "insufficient_sample"
    return len(pair), float(ss.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0]), "ok"


def icir_for_groups(df: pd.DataFrame, factor: str, group_col: str) -> float:
    vals = []
    for _, g in df.groupby(group_col):
        n, ic, status = spearman_ic(g[factor], g[TARGET])
        if status == "ok" and pd.notna(ic):
            vals.append(ic)
    if len(vals) < 2:
        return np.nan
    std = float(np.std(vals, ddof=1))
    return float(np.mean(vals) / std) if std > 0 else np.nan


def build_factor_ic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("all", "全市场", None), ("board", "分板块", "board"),
              ("year", "分年份", "listing_year"), ("period", "分时期", "period")]
    for f in FACTOR_LIST:
        if f not in df.columns:
            continue
        for scope, label, col in scopes:
            if col is None:
                n, ic, status = spearman_ic(df[f], df[TARGET])
                rows.append({
                    "factor": f, "factor_group": FACTOR_META[f]["group"],
                    "scope": scope, "scope_value": label, "n": n,
                    "spearman_ic": ic, "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
                    "icir_by_year": icir_for_groups(df, f, "listing_year"),
                    "status": status,
                })
            else:
                for key, g in df.groupby(col, dropna=False):
                    n, ic, status = spearman_ic(g[f], g[TARGET])
                    rows.append({
                        "factor": f, "factor_group": FACTOR_META[f]["group"],
                        "scope": scope, "scope_value": key, "n": n,
                        "spearman_ic": ic, "abs_ic": abs(ic) if pd.notna(ic) else np.nan,
                        "icir_by_year": np.nan,
                        "status": status,
                    })
    return pd.DataFrame(rows).sort_values(["scope", "abs_ic"], ascending=[True, False])


def build_factor_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FACTOR_LIST:
        if f not in df.columns:
            continue
        pair = df[[f, TARGET, "offline_oversubscription_ratio", "offline_allotment_ratio_pct", "y_pred_t6"]].copy()
        pair[f] = pd.to_numeric(pair[f], errors="coerce")
        pair = pair.dropna(subset=[f, TARGET])
        if len(pair) < 50 or pair[f].nunique() < 5:
            continue
        try:
            pair["bucket"] = pd.qcut(pair[f], 5, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"], duplicates="drop")
        except ValueError:
            continue
        for bucket, g in pair.groupby("bucket", observed=True):
            rows.append({
                "factor": f,
                "factor_group": FACTOR_META[f]["group"],
                "bucket": str(bucket),
                "n": len(g),
                "factor_min": g[f].min(),
                "factor_max": g[f].max(),
                "factor_median": g[f].median(),
                "mean_log_oversubscription": g[TARGET].mean(),
                "median_oversubscription": g["offline_oversubscription_ratio"].median(),
                "median_actual_lottery_rate_pct": g["offline_allotment_ratio_pct"].median(),
                "median_predicted_lottery_rate_pct": (100.0 / np.exp(g["y_pred_t6"])).median(),
            })
    return pd.DataFrame(rows)


def build_contribution_tables(shap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        shap_df.groupby("feature", dropna=False)
        .agg(n=("shap", "size"), mean_shap=("shap", "mean"), mean_abs_shap=("abs_shap", "mean"))
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    by_board_year = (
        shap_df.groupby(["board", "listing_year", "feature"], dropna=False)
        .agg(n=("shap", "size"), mean_shap=("shap", "mean"), mean_abs_shap=("abs_shap", "mean"))
        .reset_index()
        .sort_values(["board", "listing_year", "mean_abs_shap"], ascending=[True, True, False])
    )
    return summary, by_board_year


def color_scale(value: float, vmin: float, vmax: float, reverse: bool = False) -> str:
    if pd.isna(value) or vmax <= vmin:
        return "#f2f2f2"
    t = (float(value) - vmin) / (vmax - vmin)
    t = 1 - t if reverse else t
    r1, g1, b1 = (237, 248, 251)
    r2, g2, b2 = (44, 127, 184)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_heatmap(df: pd.DataFrame, path: Path, row_col: str, col_col: str, val_col: str,
                title: str, fmt_digits: int = 1) -> None:
    rows = [x for x in df[row_col].dropna().unique()]
    cols = [x for x in df[col_col].dropna().unique()]
    cell_w, cell_h = 145, 46
    left, top = 150, 70
    width = left + cell_w * len(cols) + 40
    height = top + cell_h * len(rows) + 55
    vals = pd.to_numeric(df[val_col], errors="coerce")
    vmin, vmax = vals.min(), vals.max()
    body = [svg_text(width / 2, 30, title, 18, "middle", "700")]
    for j, c in enumerate(cols):
        body.append(svg_text(left + j * cell_w + cell_w / 2, top - 18, c, 12, "middle", "700"))
    for i, r in enumerate(rows):
        body.append(svg_text(left - 12, top + i * cell_h + 28, r, 12, "end", "700"))
        for j, c in enumerate(cols):
            sub = df[(df[row_col] == r) & (df[col_col] == c)]
            val = sub[val_col].iloc[0] if not sub.empty else np.nan
            x, y = left + j * cell_w, top + i * cell_h
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color_scale(val, vmin, vmax)}" stroke="#fff"/>')
            body.append(svg_text(x + cell_w / 2, y + 28, fmt_num(val, fmt_digits), 12, "middle", "700", "#111"))
    save_svg(path, width, height, body)


def svg_bar(df: pd.DataFrame, path: Path, label_col: str, val_col: str, title: str,
            max_rows: int = 12, color: str = "#2f7ebc") -> None:
    d = df.dropna(subset=[val_col]).head(max_rows).copy()
    width, height = 860, 70 + 34 * len(d)
    left, top, bar_w = 270, 52, 500
    vmax = max(float(d[val_col].max()), 1e-9)
    body = [svg_text(width / 2, 28, title, 18, "middle", "700")]
    for i, row in enumerate(d.itertuples(index=False)):
        label = getattr(row, label_col)
        val = float(getattr(row, val_col))
        y = top + i * 34
        w = bar_w * val / vmax
        body.append(svg_text(left - 12, y + 18, label, 12, "end"))
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" fill="{color}" rx="2"/>')
        body.append(svg_text(left + w + 8, y + 17, fmt_num(val, 3), 12))
    save_svg(path, width, height, body)


def svg_factor_profile(profile: pd.DataFrame, path: Path) -> None:
    factors = [
        "median_market_turnover_ma20",
        "median_recent_ipo_return_ma20",
        "median_concurrent_ipo_count",
        "median_same_board_break_rate_ma10",
        "median_offline_issue_before_share_pct",
        "median_subscription_upper_limit_10k",
    ]
    names = ["市场成交额", "新股赚钱效应", "IPO拥挤度", "同板块破发率", "网下发行占比", "申购上限"]
    long = profile[["board", *factors]].melt("board", var_name="factor", value_name="value")
    long["factor"] = long["factor"].map(dict(zip(factors, names)))
    svg_heatmap(long, path, "board", "factor", "value", "各板块核心询价前因子画像", 1)


def svg_market_bucket(groups: pd.DataFrame, path: Path) -> None:
    d = groups[groups["factor"] == "market_turnover_ma20"].copy()
    if d.empty:
        save_svg(path, 720, 120, [svg_text(360, 60, "market_turnover_ma20 分组样本不足", 16, "middle")])
        return
    width, height = 760, 360
    left, top, plot_w, plot_h = 90, 55, 580, 230
    d = d.sort_values("bucket")
    vals = d["median_actual_lottery_rate_pct"].astype(float)
    vmax = max(vals.max(), 1e-9)
    body = [svg_text(width / 2, 30, "市场成交额分组 vs 实际网下中签率", 18, "middle", "700")]
    bw = plot_w / len(d) * 0.62
    for i, row in enumerate(d.itertuples(index=False)):
        x = left + i * plot_w / len(d) + plot_w / len(d) * 0.18
        h = plot_h * float(row.median_actual_lottery_rate_pct) / vmax
        y = top + plot_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#54a24b"/>')
        body.append(svg_text(x + bw / 2, top + plot_h + 22, row.bucket, 12, "middle"))
        body.append(svg_text(x + bw / 2, y - 6, fmt_num(row.median_actual_lottery_rate_pct, 3), 11, "middle"))
    body.append(svg_text(25, top + 20, "中签率(%)", 12))
    save_svg(path, width, height, body)


def build_monthly_dual_axis_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["board"].isin(["主板", "创业板", "科创板"])].copy()
    d["date_for_plot"] = d["subscription_deadline_date"].fillna(d["listing_date"])
    d = d.dropna(subset=["date_for_plot"])
    d["month"] = d["date_for_plot"].dt.to_period("M").dt.to_timestamp()
    board_month = (
        d.groupby(["month", "board"], dropna=False)
        .agg(
            n=("offline_oversubscription_ratio", lambda s: int(s.notna().sum())),
            median_oversubscription=("offline_oversubscription_ratio", "median"),
        )
        .reset_index()
    )
    turnover = (
        d.groupby("month", dropna=False)
        .agg(market_turnover_ma20=("market_turnover_ma20", "median"))
        .reset_index()
    )
    out = board_month.merge(turnover, on="month", how="left")
    return out.sort_values(["month", "board"])


def svg_board_oversub_turnover_dual_axis(monthly: pd.DataFrame, path: Path) -> None:
    """Dual-axis monthly trend: board oversubscription vs market turnover."""
    d = monthly.copy()
    d = d[d["n"] >= 2].copy()
    if d.empty:
        save_svg(path, 900, 140, [svg_text(450, 70, "月度样本不足，无法绘图", 16, "middle")])
        return

    boards = ["主板", "创业板", "科创板"]
    colors = {"主板": "#D95F02", "创业板": "#1B9E77", "科创板": "#386CB0"}
    months = pd.Series(sorted(d["month"].dropna().unique()))
    tmin, tmax = months.min(), months.max()
    width, height = 1120, 590
    left, right, top, bottom = 104, 118, 74, 92
    plot_w, plot_h = width - left - right, height - top - bottom

    y1 = pd.to_numeric(d["median_oversubscription"], errors="coerce")
    y1 = y1[y1 > 0]
    y1_min = max(1.0, float(y1.min()) * 0.75 if len(y1) else 10.0)
    y1_max = float(y1.max()) * 1.20 if len(y1) else 10000.0
    y1_max = max(y1_max, y1_min * 10)
    log_min, log_max = np.log10(y1_min), np.log10(y1_max)

    turn = pd.to_numeric(d["market_turnover_ma20"], errors="coerce").dropna()
    y2_min = 0.0
    y2_max = float(turn.max()) * 1.10 if len(turn) else 1.0
    y2_max = max(y2_max, 1.0)

    def xscale(ts) -> float:
        total = max((pd.Timestamp(tmax) - pd.Timestamp(tmin)).days, 1)
        return left + (pd.Timestamp(ts) - pd.Timestamp(tmin)).days / total * plot_w

    def yscale_left(v) -> float:
        if pd.isna(v) or v <= 0:
            return np.nan
        t = (np.log10(float(v)) - log_min) / (log_max - log_min)
        t = min(max(t, 0.0), 1.0)
        return top + plot_h * (1 - t)

    def yscale_right(v) -> float:
        if pd.isna(v):
            return np.nan
        t = (float(v) - y2_min) / (y2_max - y2_min)
        t = min(max(t, 0.0), 1.0)
        return top + plot_h * (1 - t)

    body = [
        svg_text(width / 2, 30, "各板块网下超额认购倍数与市场成交额（月度）", 19, "middle", "700"),
        svg_text(width / 2, 52, "左轴：超额认购倍数中位数（log）｜右轴：市场近20日成交额中位数（亿元）", 12, "middle", "400", "#666"),
        svg_text(left, top - 16, "超额认购倍数（左轴，log）", 12, "start", "700", "#333"),
        svg_text(left + plot_w, top - 16, "成交额（右轴，亿元）", 12, "end", "700", "#666"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#666"/>',
    ]

    left_ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    for tick in left_ticks:
        if y1_min <= tick <= y1_max:
            y = yscale_left(tick)
            body.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
            body.append(svg_text(left - 9, y + 4, f"{tick:,}", 11, "end", color="#333"))
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eee"/>')

    for tick in np.linspace(0, y2_max, 5):
        y = yscale_right(tick)
        body.append(f'<line x1="{left+plot_w}" y1="{y:.1f}" x2="{left+plot_w+5}" y2="{y:.1f}" stroke="#666"/>')
        body.append(svg_text(left + plot_w + 9, y + 4, f"{tick:,.0f}", 11, "start", color="#666"))

    # Market turnover as dashed area/line on right axis.
    turn_line = (
        d[["month", "market_turnover_ma20"]]
        .dropna()
        .drop_duplicates("month")
        .sort_values("month")
    )
    pts = [(xscale(r.month), yscale_right(r.market_turnover_ma20)) for r in turn_line.itertuples(index=False)]
    pts = [(x, y) for x, y in pts if np.isfinite(y)]
    if len(pts) >= 2:
        pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        body.append(f'<polyline points="{pstr}" fill="none" stroke="#777" stroke-width="2.5" stroke-dasharray="6 4"/>')

    for board in boards:
        sub = d[d["board"] == board].sort_values("month")
        pts = [(xscale(r.month), yscale_left(r.median_oversubscription)) for r in sub.itertuples(index=False)]
        pts = [(x, y) for x, y in pts if np.isfinite(y)]
        if len(pts) < 2:
            continue
        pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        body.append(f'<polyline points="{pstr}" fill="none" stroke="{colors[board]}" stroke-width="3"/>')
        for x, y in pts:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colors[board]}"/>')

    # X ticks by year.
    years = sorted({pd.Timestamp(m).year for m in months})
    for year in years:
        ts = pd.Timestamp(year=year, month=1, day=1)
        if ts < pd.Timestamp(tmin) or ts > pd.Timestamp(tmax):
            continue
        x = xscale(ts)
        body.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#333"/>')
        body.append(svg_text(x, top + plot_h + 24, year, 11, "middle"))

    legend_x, legend_y = left + 12, top + 16
    for i, board in enumerate(boards):
        y = legend_y + i * 22
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="{colors[board]}" stroke-width="3"/>')
        body.append(svg_text(legend_x + 34, y + 4, board, 12))
    y = legend_y + len(boards) * 22
    body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="#777" stroke-width="2.5" stroke-dasharray="6 4"/>')
    body.append(svg_text(legend_x + 34, y + 4, "市场成交额", 12))

    save_svg(path, width, height, body)


def svg_shap_matrix(contrib: pd.DataFrame, path: Path, group_col: str, title: str) -> None:
    top = (
        contrib.groupby("feature")["mean_abs_shap"].mean()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    d = contrib[contrib["feature"].isin(top)].copy()
    d = d.groupby([group_col, "feature"], dropna=False)["mean_abs_shap"].mean().reset_index()
    svg_heatmap(d, path, group_col, "feature", "mean_abs_shap", title, 3)


def make_findings(
    board_profile: pd.DataFrame,
    period_profile: pd.DataFrame,
    factor_ic: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    groups: pd.DataFrame,
) -> list[str]:
    findings = []
    bp = board_profile.dropna(subset=["median_oversubscription"])
    if len(bp) >= 2:
        hi = bp.sort_values("median_oversubscription", ascending=False).iloc[0]
        lo = bp.sort_values("median_oversubscription", ascending=True).iloc[0]
        ratio = hi["median_oversubscription"] / lo["median_oversubscription"]
        findings.append(
            f"{hi['board']}网下超额认购倍数中位数最高（{hi['median_oversubscription']:.0f}倍），"
            f"约为{lo['board']}（{lo['median_oversubscription']:.0f}倍）的{ratio:.1f}倍，板块制度差异是第一层结构。"
        )
    if "market_turnover_ma20" in contrib_summary["feature"].values:
        rank = int(contrib_summary["feature"].tolist().index("market_turnover_ma20") + 1)
        val = contrib_summary.loc[contrib_summary["feature"] == "market_turnover_ma20", "mean_abs_shap"].iloc[0]
        findings.append(f"市场成交额 `market_turnover_ma20` 在T-6模型平均绝对SHAP中排名第{rank}，平均贡献约{val:.3f}（log空间），是关键外部环境因子。")
    all_ic = factor_ic[(factor_ic["scope"] == "all") & (factor_ic["status"] == "ok")].copy()
    if not all_ic.empty:
        top = all_ic.sort_values("abs_ic", ascending=False).iloc[0]
        findings.append(f"单因子IC最高的是 `{top['factor']}`（Spearman IC={top['spearman_ic']:.3f}），可作为询价前重点跟踪因子。")
    mt = groups[groups["factor"] == "market_turnover_ma20"].sort_values("bucket")
    if len(mt) >= 2:
        low, high = mt.iloc[0], mt.iloc[-1]
        findings.append(
            f"市场成交额从低分位到高分位时，实际中签率中位数由{low['median_actual_lottery_rate_pct']:.4f}%"
            f"变化到{high['median_actual_lottery_rate_pct']:.4f}%，可用于向领导解释流动性环境分层。"
        )
    pp = period_profile.dropna(subset=["median_market_turnover_ma20"])
    if len(pp) >= 2:
        hi = pp.sort_values("median_market_turnover_ma20", ascending=False).iloc[0]
        findings.append(f"{hi['period']}的市场成交额中位数最高，说明不同年份/时期的资金环境不可混为一谈。")
    findings.append("T-1/T+1仅作为信息增益上界；本报告所有核心因子均按T-6询价前口径输出。")
    return findings[:8]


def make_report(
    findings: list[str],
    board_profile: pd.DataFrame,
    period_profile: pd.DataFrame,
    factor_ic: pd.DataFrame,
    contrib_summary: pd.DataFrame,
) -> str:
    top_ic = (
        factor_ic[(factor_ic["scope"] == "all") & (factor_ic["status"] == "ok")]
        .sort_values("abs_ic", ascending=False)
        [["factor", "factor_group", "n", "spearman_ic", "icir_by_year"]]
        .head(12)
    )
    top_contrib = contrib_summary[["feature", "n", "mean_shap", "mean_abs_shap"]].head(12)
    bullets = "\n".join([f"- {x}" for x in findings])
    return f"""# 询价前因子洞察与板块时期特征报告

生成日期：2026-05-28

## 领导速览

{bullets}

## 板块画像

{md_table(board_profile, digits=2)}

## 时期画像

{md_table(period_profile, digits=2)}

## T-6 单因子 IC Top

IC 是因子与未来 `log(网下超额认购倍数)` 的 Spearman 排序相关。正值表示因子越高，未来超额认购通常越高、中签率通常越低。

{md_table(top_ic, digits=3)}

## T-6 模型解释贡献 Top

以下为 LightGBM TreeSHAP 平均绝对贡献，含义是“对模型预测结果的解释贡献”，不是严格因果检验。

{md_table(top_contrib, digits=3)}

## 图表索引

- `figures/board_period_heatmap.svg`：板块 × 时期的超额认购倍数热力图
- `figures/board_factor_profile.svg`：各板块核心询价前因子画像
- `figures/factor_ic_bar_t6.svg`：T-6 因子 IC 排名
- `figures/market_turnover_bucket.svg`：市场成交额分组 vs 实际中签率
- `figures/factor_shap_by_board.svg`：各板块 Top 因子 SHAP 热力图
- `figures/factor_shap_by_year.svg`：因子贡献随年份变化
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    bundle = load_model_bundle()
    df["y_pred_t6"] = predict_log(df, bundle)
    df["predicted_lottery_rate_pct_t6"] = 100.0 / np.exp(df["y_pred_t6"])

    factor_dictionary().to_csv(OUT_DIR / "factor_dictionary.csv", index=False, encoding="utf-8-sig")

    board_profile = summarize_profile(df, ["board"])
    year_profile = summarize_profile(df, ["listing_year"])
    period_profile = summarize_profile(df, ["period"])
    board_period_profile = summarize_profile(df, ["board", "period"])
    board_profile.to_csv(OUT_DIR / "board_profile.csv", index=False, encoding="utf-8-sig")
    year_profile.to_csv(OUT_DIR / "year_profile.csv", index=False, encoding="utf-8-sig")
    period_profile.to_csv(OUT_DIR / "period_profile.csv", index=False, encoding="utf-8-sig")
    board_period_profile.to_csv(OUT_DIR / "board_period_profile.csv", index=False, encoding="utf-8-sig")
    monthly_dual = build_monthly_dual_axis_data(df)
    monthly_dual.to_csv(OUT_DIR / "board_monthly_oversub_turnover.csv", index=False, encoding="utf-8-sig")

    factor_ic = build_factor_ic(df)
    factor_ic.to_csv(OUT_DIR / "factor_ic.csv", index=False, encoding="utf-8-sig")

    factor_groups = build_factor_groups(df)
    factor_groups.to_csv(OUT_DIR / "factor_group_returns.csv", index=False, encoding="utf-8-sig")

    shap_wide, shap_df = shap_long(df[df[TARGET].notna()].copy(), bundle)
    shap_wide.to_csv(OUT_DIR / "factor_shap_wide.csv", index=False, encoding="utf-8-sig")
    shap_df.to_csv(OUT_DIR / "factor_shap_long.csv", index=False, encoding="utf-8-sig")
    contrib_summary, contrib_by_board_year = build_contribution_tables(shap_df)
    contrib_summary.to_csv(OUT_DIR / "factor_contribution_summary.csv", index=False, encoding="utf-8-sig")
    contrib_by_board_year.to_csv(OUT_DIR / "factor_contribution_by_board_year.csv", index=False, encoding="utf-8-sig")

    svg_heatmap(
        board_period_profile,
        FIG_DIR / "board_period_heatmap.svg",
        "board",
        "period",
        "median_oversubscription",
        "板块 × 时期：网下超额认购倍数中位数",
        0,
    )
    svg_factor_profile(board_profile, FIG_DIR / "board_factor_profile.svg")
    top_ic = (
        factor_ic[(factor_ic["scope"] == "all") & (factor_ic["status"] == "ok")]
        .sort_values("abs_ic", ascending=False)
        .assign(label=lambda x: x["factor"])
    )
    svg_bar(top_ic, FIG_DIR / "factor_ic_bar_t6.svg", "label", "abs_ic", "T-6 因子 |IC| 排名", 12)
    svg_market_bucket(factor_groups, FIG_DIR / "market_turnover_bucket.svg")
    svg_board_oversub_turnover_dual_axis(
        monthly_dual,
        FIG_DIR / "board_oversub_turnover_dual_axis.svg",
    )
    by_board = (
        shap_df.groupby(["board", "feature"], dropna=False)
        .agg(mean_abs_shap=("abs_shap", "mean"))
        .reset_index()
    )
    svg_shap_matrix(by_board, FIG_DIR / "factor_shap_by_board.svg", "board", "各板块T-6因子SHAP贡献")
    by_year = (
        shap_df.groupby(["listing_year", "feature"], dropna=False)
        .agg(mean_abs_shap=("abs_shap", "mean"))
        .reset_index()
    )
    svg_shap_matrix(by_year, FIG_DIR / "factor_shap_by_year.svg", "listing_year", "各年份T-6因子SHAP贡献")

    findings = make_findings(board_profile, period_profile, factor_ic, contrib_summary, factor_groups)
    (OUT_DIR / "factor_insights_report.md").write_text(
        make_report(findings, board_profile, period_profile, factor_ic, contrib_summary),
        encoding="utf-8",
    )
    manifest = {
        "generated": "2026-05-28",
        "official_stage": "T6",
        "outputs": [
            "factor_insights_report.md",
            "board_period_profile.csv",
            "factor_ic.csv",
            "factor_group_returns.csv",
            "factor_contribution_by_board_year.csv",
            "board_monthly_oversub_turnover.csv",
            "figures/*.svg",
        ],
        "factors": FACTOR_LIST,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
