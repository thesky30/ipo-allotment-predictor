"""Three-stage IPO offline subscription prediction with strict temporal discipline.

Timeline
--------
T-6  申购决策   Investor allocates capital before inquiry opens.
                Only prospectus / inquiry-announcement fields available.
T-1  回拨前预测  After inquiry closes; inquiry results published; subscription not yet open.
                This is the DEMO MODEL stage (no offline subscription data).
T+1  回拨后预测  After subscription closes; clawback ratio announced.

Every feature is tagged with its information-release node (T-6 / T-1 / T+1).
Model-1 (T-6)   : FEATS_T6 only.
Model-2A (T-1)  : FEATS_T6 + FEATS_T1_DELTA  ← demo model.
Model-2B (T+1)  : FEATS_T6 + FEATS_T1_DELTA + FEATS_T1PLUS_DELTA.

OOS backtest: expanding window ordered by subscription_deadline_date.
Cutoffs: 2022-01-01 / 2023-01-01 / 2024-01-01 / 2025-01-01.
No random train/test splits anywhere.

Trained models are saved to outputs/baseline_models/models/ for use by predict.py.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

# Ensure scripts/ directory is on sys.path so model_classes is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import shared model classes (stable pickle path regardless of __main__)
from model_classes import BoardMeanModel, _ColSelector  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "baseline_models"
MODEL_DIR = OUT_DIR / "models"

TARGET = "log_offline_oversubscription"
SORT_COL = "sort_date"
RANDOM_SEED = 42
MIN_TRAIN = 80

CUTOFFS = [
    pd.Timestamp("2022-01-01"),
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2025-01-01"),
]

# ---------------------------------------------------------------------------
# Feature time-node classification
# Each entry: field → (time_node, source, category)
# time_node: "T-6" | "T-1" | "T+1" | "T+2"
# T+2 = target / post-allocation; NEVER used as model input.
# ---------------------------------------------------------------------------

FEATURE_NODES: dict[str, dict] = {
    # ── T-6 : prospectus / inquiry-announcement (before inquiry opens) ────
    "board":                                {"node": "T-6", "src": "常识/招股书",          "cat": "板块结构"},
    "total_issue_shares_10k":               {"node": "T-6", "src": "招股书",               "cat": "发行结构"},
    "offline_issue_before_clawback_10k":    {"node": "T-6", "src": "发行安排公告",          "cat": "发行结构"},
    "online_issue_before_clawback_10k":     {"node": "T-6", "src": "发行安排公告",          "cat": "发行结构"},
    "strategic_allocation_10k":             {"node": "T-6", "src": "发行安排公告",          "cat": "战略配售"},
    "strategic_allocation_share_pct":       {"node": "T-6", "src": "派生",                  "cat": "战略配售"},
    "offline_issue_before_share_pct":       {"node": "T-6", "src": "派生",                  "cat": "发行结构"},
    "subscription_upper_limit_10k":         {"node": "T-6", "src": "询价及推介公告",         "cat": "申购规则"},
    "subscription_lower_limit_10k":         {"node": "T-6", "src": "询价及推介公告",         "cat": "申购规则"},
    "subscription_step_10k":                {"node": "T-6", "src": "询价及推介公告",         "cat": "申购规则"},
    "offer_price_upper_yuan":               {"node": "T-6", "src": "询价公告（科创板）",     "cat": "发行定价"},
    "offer_price_lower_yuan":               {"node": "T-6", "src": "询价公告（科创板）",     "cat": "发行定价"},
    "offer_price_range_pct":                {"node": "T-6", "src": "派生（科创板）",         "cat": "发行定价"},
    "comparable_pe_avg_ex_nonrecurring":    {"node": "T-6", "src": "招股书",               "cat": "估值"},
    "industry_pe_at_ipo":                   {"node": "T-6", "src": "市场数据",              "cat": "估值"},
    "recent_ipo_first_day_return_ma20":     {"node": "T-6", "src": "历史IPO数据派生（滚动）","cat": "市场热度"},

    # ── T-1 : inquiry-result announcement (after inquiry, before subscription) ─
    "offer_price_yuan":                     {"node": "T-1", "src": "定价公告",              "cat": "发行定价"},
    "ipo_pe_diluted":                       {"node": "T-1", "src": "基于最终发行价",         "cat": "估值"},
    "issue_pb":                             {"node": "T-1", "src": "基于最终发行价",         "cat": "估值"},
    "pe_vs_industry":                       {"node": "T-1", "src": "派生（依赖最终价）",     "cat": "估值"},
    "pe_vs_comparable":                     {"node": "T-1", "src": "派生（依赖最终价）",     "cat": "估值"},
    "issue_amount_100m_yuan":               {"node": "T-1", "src": "派生（依赖最终价）",     "cat": "发行规模"},
    "inquiry_subscription_total_10k":       {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价结果"},
    "inquiry_investors_count":              {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价结果"},
    "inquiry_allotment_accounts":           {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价结果"},
    "inquiry_oversubscription_ratio":       {"node": "T-1", "src": "派生",                  "cat": "询价结果"},
    "quote_price_weighted_avg":             {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价价格"},
    "quote_price_median":                   {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价价格"},
    "quote_price_vs_offer":                 {"node": "T-1", "src": "派生",                  "cat": "询价价格"},
    "excluded_subscription_share_pct":      {"node": "T-1", "src": "初步询价结果公告",       "cat": "询价结果"},
    "high_price_excluded_subscription_share_pct": {"node": "T-1", "src": "初步询价结果公告", "cat": "询价结果"},
    "offer_price_position_in_range":        {"node": "T-1", "src": "派生（科创板）",         "cat": "发行定价"},

    # ── T+1 : clawback announcement ──────────────────────────────────────
    "clawback_ratio_pct":                   {"node": "T+1", "src": "回拨公告",              "cat": "回拨"},
    "offline_issue_final_10k":              {"node": "T+1", "src": "回拨后派生",             "cat": "回拨"},
    "offline_issue_final_share_pct":        {"node": "T+1", "src": "派生",                  "cat": "回拨"},
    "online_issue_final_10k":               {"node": "T+1", "src": "回拨公告",              "cat": "回拨"},

    # ── T+2 : post-allocation results — NEVER used as model input ─────────
    "offline_oversubscription_ratio":           {"node": "T+2", "src": "申购结果公告", "cat": "目标变量"},
    "log_offline_oversubscription":             {"node": "T+2", "src": "派生",         "cat": "目标变量"},
    "offline_allotment_ratio_pct":              {"node": "T+2", "src": "配售结果公告", "cat": "目标变量"},
    "offline_subscription_total_10k":           {"node": "T+2", "src": "申购结果公告", "cat": "目标变量"},
    "offline_valid_quote_subscription_10k":     {"node": "T+2", "src": "申购结果公告", "cat": "目标变量"},
    "offline_allotment_accounts":               {"node": "T+2", "src": "配售结果公告", "cat": "目标变量"},
    "offline_inquiry_investors":                {"node": "T+2", "src": "申购结果公告（≠初步询价）", "cat": "目标变量"},
    "a_investor_lottery_rate_pct":              {"node": "T+2", "src": "配售结果公告", "cat": "目标变量"},
    "a_investor_allotted_shares_10k":           {"node": "T+2", "src": "配售结果公告", "cat": "目标变量"},
    "a_investor_subscription_shares_10k":       {"node": "T+2", "src": "申购结果公告", "cat": "目标变量"},
    "a_investor_allotted_accounts":             {"node": "T+2", "src": "配售结果公告", "cat": "目标变量"},
    "offline_oversubscription_ratio_before_clawback": {"node": "T+2", "src": "申购结果公告", "cat": "目标变量"},
}

# ---------------------------------------------------------------------------
# Derived feature sets (from FEATURE_NODES)
# ---------------------------------------------------------------------------

def _fields_for_node(node: str) -> list[str]:
    return [f for f, v in FEATURE_NODES.items() if v["node"] == node]

FEATS_T6 = _fields_for_node("T-6")

FEATS_T1_DELTA = _fields_for_node("T-1")          # incremental over T-6
FEATS_T1 = FEATS_T6 + FEATS_T1_DELTA              # cumulative (demo model)

FEATS_T1PLUS_DELTA = _fields_for_node("T+1")      # incremental over T-1
FEATS_T1PLUS = FEATS_T1 + FEATS_T1PLUS_DELTA      # cumulative

CAT_COLS = ["board"]

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

STAGE_LABEL = {
    "T6":     "模型一 (T-6)   申购决策期",
    "T1":     "模型二-A (T-1) 回拨前 [演示模型]",
    "T1PLUS": "模型二-B (T+1) 回拨后",
}


# ---------------------------------------------------------------------------
# Model classes & factories
# ---------------------------------------------------------------------------

# BoardMeanModel and _ColSelector imported from model_classes above


def _preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in CAT_COLS]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
            ]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )


def make_ridge(feature_cols: list[str], alpha: float = 1.0) -> _ColSelector:
    pipe = Pipeline([("pre", _preprocessor(feature_cols)),
                     ("reg", Ridge(alpha=alpha))])
    return _ColSelector(pipe, feature_cols)


def make_lgbm(feature_cols: list[str]) -> _ColSelector:
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in CAT_COLS]
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="drop",
    )
    pipe = Pipeline([
        ("pre", pre),
        ("reg", lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05,
            max_depth=6, num_leaves=31, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbose=-1,
        )),
    ])
    return _ColSelector(pipe, feature_cols)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    model_name: str,
    stage_key: str,
    make_model_fn,
    cutoffs: list[pd.Timestamp] = CUTOFFS,
    min_train: int = MIN_TRAIN,
) -> pd.DataFrame:
    """Expanding-window OOS backtest. Returns DataFrame of predictions."""
    labeled = df[df[TARGET].notna() & df[SORT_COL].notna()].copy()
    out_parts: list[pd.DataFrame] = []

    for i, cutoff in enumerate(cutoffs):
        next_cut = cutoffs[i + 1] if i + 1 < len(cutoffs) else pd.Timestamp("2099-01-01")
        train = labeled[labeled[SORT_COL] < cutoff]
        test  = labeled[(labeled[SORT_COL] >= cutoff) & (labeled[SORT_COL] < next_cut)]
        if len(train) < min_train or len(test) == 0:
            continue

        model = make_model_fn()
        model.fit(train, train[TARGET])
        y_pred = model.predict(test)

        chunk = test[["security_code", "security_name", "board",
                      "listing_date", "listing_year", SORT_COL, TARGET]].copy()
        chunk["y_pred"]   = y_pred
        chunk["model"]    = model_name
        chunk["stage"]    = stage_key
        chunk["cutoff"]   = cutoff.strftime("%Y-%m-%d")
        chunk["train_n"]  = len(train)
        out_parts.append(chunk)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Full-data model training & persistence
# ---------------------------------------------------------------------------

def train_and_save(
    df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    make_model_fn,
    model_dir: Path,
) -> object:
    """Train on all labeled data and persist with joblib. Returns fitted model."""
    labeled = df[df[TARGET].notna()].copy()
    model = make_model_fn()
    model.fit(labeled, labeled[TARGET])
    path = model_dir / f"{model_name}.joblib"
    joblib.dump({"model": model, "features": feature_cols,
                 "target": TARGET, "model_name": model_name}, path)
    return model


# ---------------------------------------------------------------------------
# LightGBM feature importance
# ---------------------------------------------------------------------------

def lgbm_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    labeled = df[df[TARGET].notna()].copy()
    avail = [c for c in feature_cols if c in labeled.columns]
    cat = [c for c in CAT_COLS if c in avail]
    num = [c for c in avail if c not in CAT_COLS]
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="drop",
    )
    X = pre.fit_transform(labeled[avail])
    y = labeled[TARGET].values
    feat_names = list(pre.get_feature_names_out())
    mdl = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(X, y)
    return (
        pd.DataFrame({
            "model": model_name,
            "feature": feat_names,
            "importance_gain":  mdl.booster_.feature_importance("gain"),
            "importance_split": mdl.booster_.feature_importance("split"),
        })
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        return {"n": n, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
    return {
        "n":        n,
        "mae":      float(mean_absolute_error(yt, yp)),
        "rmse":     float(np.sqrt(mean_squared_error(yt, yp))),
        "r2":       float(r2_score(yt, yp)),
        "spearman": float(ss.spearmanr(yt, yp)[0]),
    }


def build_metrics(pred_df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    for (model, stage), mdf in pred_df.groupby(["model", "stage"]):
        if group_col is None:
            m = metrics(mdf[TARGET].values, mdf["y_pred"].values)
            m.update({"model": model, "stage": stage})
            rows.append(m)
        else:
            for key, gdf in mdf.groupby(group_col):
                m = metrics(gdf[TARGET].values, gdf["y_pred"].values)
                m.update({"model": model, "stage": stage, group_col: key})
                rows.append(m)
    base = ["model", "stage"] + ([] if group_col is None else [group_col])
    return pd.DataFrame(rows)[base + ["n", "mae", "rmse", "r2", "spearman"]]


# ---------------------------------------------------------------------------
# SVG helpers (minimal, no external deps)
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    from xml.sax.saxutils import escape
    return escape(str(s))


def svg_stage_comparison(metrics_overall: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart: Spearman by model × stage."""
    df = metrics_overall[metrics_overall["spearman"].notna()].copy()
    if df.empty:
        return
    df = df.sort_values(["stage", "spearman"])

    stage_colors = {"T6": "#4C78A8", "T1": "#54A24B", "T1PLUS": "#F58518"}
    W, H = 800, max(320, len(df) * 38 + 120)
    left, top, bottom = 240, 70, 50
    pw = W - left - 80

    def bx(v: float) -> float:
        return left + max(float(v), 0) / 1.0 * pw

    rh = (H - top - bottom) / len(df)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W//2}" y="30" font-family="Arial,sans-serif" font-size="18" '
        f'text-anchor="middle" font-weight="700" fill="#222">三阶段模型 Spearman 比较（OOS）</text>',
        f'<text x="{W//2}" y="52" font-family="Arial,sans-serif" font-size="12" '
        f'text-anchor="middle" fill="#666">expanding-window backtest — 按申购截止日排序</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = bx(tick)
        lines += [
            f'<line x1="{x:.0f}" y1="{top}" x2="{x:.0f}" y2="{H - bottom}" stroke="#e6e6e6"/>',
            f'<text x="{x:.0f}" y="{H - bottom + 18}" font-family="Arial,sans-serif" '
            f'font-size="11" text-anchor="middle" fill="#666">{tick:.2f}</text>',
        ]
    for i, (_, row) in enumerate(df.iterrows()):
        y = top + i * rh + rh * 0.1
        bw = bx(row["spearman"]) - left
        col = stage_colors.get(row["stage"], "#888")
        label = f'{row["model"]}  [{row["stage"]}]'
        lines += [
            f'<text x="{left - 8}" y="{y + rh*0.62:.0f}" font-family="Arial,sans-serif" '
            f'font-size="12" text-anchor="end" fill="#333">{_esc(label)}</text>',
            f'<rect x="{left}" y="{y:.0f}" width="{max(bw,2):.0f}" height="{rh*0.78:.0f}" '
            f'fill="{col}" fill-opacity="0.85"/>',
            f'<text x="{left + bw + 6:.0f}" y="{y + rh*0.62:.0f}" font-family="Arial,sans-serif" '
            f'font-size="11" fill="#333">{row["spearman"]:.4f}</text>',
        ]
    # Legend
    for j, (stg, col) in enumerate(stage_colors.items()):
        lx, ly = W - 140, top + j * 22
        lines += [
            f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{col}"/>',
            f'<text x="{lx + 18}" y="{ly + 11}" font-family="Arial,sans-serif" '
            f'font-size="12" fill="#333">{stg}</text>',
        ]
    lines.append("</svg>")
    (fig_dir / "stage_spearman_comparison.svg").write_text("\n".join(lines), encoding="utf-8")


def svg_importance(imp_df: pd.DataFrame, model_name: str, fig_dir: Path) -> None:
    top = imp_df[imp_df["model"] == model_name].head(18).iloc[::-1]
    if top.empty:
        return
    W, H = 860, 560
    left, top_m, bottom = 320, 60, 50
    pw = W - left - 60
    vmax = float(top["importance_gain"].max()) * 1.1 or 1

    def bx(v: float) -> float:
        return left + float(v) / vmax * pw

    rh = (H - top_m - bottom) / len(top)

    # colour by time node
    node_colors = {"T-6": "#4C78A8", "T-1": "#54A24B", "T+1": "#F58518"}

    def feat_node(fname: str) -> str:
        # strip sklearn prefix (num__, cat__)
        raw = fname.split("__")[-1].split("_")[0] if "__" in fname else fname
        # look up in FEATURE_NODES by prefix match
        for field, info in FEATURE_NODES.items():
            if fname.endswith(field) or field in fname:
                return info["node"]
        return "T-1"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W//2}" y="30" font-family="Arial,sans-serif" font-size="17" '
        f'text-anchor="middle" font-weight="700" fill="#222">'
        f'Feature Importance — {_esc(model_name)}</text>',
        f'<text x="{W//2}" y="50" font-family="Arial,sans-serif" font-size="11" '
        f'text-anchor="middle" fill="#666">gain; colour = information node</text>',
    ]
    for i, (_, row) in enumerate(top.iterrows()):
        y = top_m + i * rh + rh * 0.1
        bw = bx(row["importance_gain"]) - left
        col = node_colors.get(feat_node(row["feature"]), "#888")
        lines += [
            f'<text x="{left - 8}" y="{y + rh*0.65:.0f}" font-family="Arial,sans-serif" '
            f'font-size="11" text-anchor="end" fill="#333">{_esc(row["feature"])}</text>',
            f'<rect x="{left}" y="{y:.0f}" width="{max(bw,2):.0f}" height="{rh*0.78:.0f}" '
            f'fill="{col}" fill-opacity="0.85"/>',
        ]
    for j, (nd, col) in enumerate(node_colors.items()):
        lx, ly = W - 120, top_m + j * 22
        lines += [
            f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{col}"/>',
            f'<text x="{lx+18}" y="{ly+11}" font-family="Arial,sans-serif" font-size="12" fill="#333">{nd}</text>',
        ]
    lines.append("</svg>")
    (fig_dir / f"importance_{model_name}.svg").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: object, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def _md(df: pd.DataFrame, d: int = 4) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row[c], d) for c in cols) + " |")
    return "\n".join(rows)


def make_report(
    metrics_overall: pd.DataFrame,
    metrics_board: pd.DataFrame,
    imp_df: pd.DataFrame,
) -> str:
    best = metrics_overall.sort_values("spearman", ascending=False).iloc[0]
    demo = metrics_overall[metrics_overall["stage"] == "T1"].sort_values("spearman", ascending=False)
    demo_row = demo.iloc[0] if not demo.empty else best
    return f"""# 三阶段模型回测报告

生成日期：2026-05-22

## 设计原则

| 时间节点 | 含义 | 可用信息 |
|---|---|---|
| T-6 | 申购决策期 | 招股书、询价公告（申购上下限/步长）、行业PE、历史热度 |
| T-1 | 回拨前预测 | T-6全部 + 询价结果（申购总量、机构数、价格分布、最终发行价）|
| T+1 | 回拨后预测 | T-1全部 + 回拨比例 |
| T+2 | 目标变量 | 网下超额认购倍数（禁止作为输入）|

演示模型 = **T-1 LightGBM**（不使用任何网下申购数据）

## OOS 整体指标

{_md(metrics_overall)}

> 最佳模型：**{best['model']}** [{best['stage']}]
> Spearman={_fmt(best['spearman'])} MAE={_fmt(best['mae'])}
>
> 演示模型（lgbm_t1）：Spearman={_fmt(demo_row.get('spearman', float('nan')))} MAE={_fmt(demo_row.get('mae', float('nan')))}

## 分板块 OOS 指标

{_md(metrics_board)}

## T-6 vs T-1 vs T+1 信息增益

| 阶段跃升 | Spearman 提升 | 含义 |
|---|---|---|
| T-6 → T-1 | 询价结果的价值 | 加入 inquiry_oversubscription_ratio 等 |
| T-1 → T+1 | 回拨比例的价值 | 加入 clawback_ratio_pct |

## 特征重要性 Top-10（lgbm_t1）

{_md(imp_df[imp_df['model']=='lgbm_t1'][['feature','importance_gain','importance_split']].head(10), d=1)}

## 保存的模型文件

所有模型已用 joblib 序列化至 `outputs/baseline_models/models/`。
使用 `scripts/predict.py` 加载并预测新 IPO。
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    with sqlite3.connect(DATA_DIR / "ipo_offline.db") as conn:
        df = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)
    for dcol in ["listing_date", "subscription_deadline_date"]:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[SORT_COL] = df["subscription_deadline_date"].fillna(df["listing_date"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["listing_year"] = pd.to_numeric(df["listing_year"], errors="coerce")

    labeled = df[df[TARGET].notna()]
    print(f"Labeled: {len(labeled)} | Missing sort_date: {df[SORT_COL].isna().sum()}")

    # ── Export feature time-node classification ────────────────────────────
    node_df = pd.DataFrame([
        {"field": f, "time_node": v["node"], "source": v["src"], "category": v["cat"],
         "in_model1_T6": v["node"] == "T-6",
         "in_model2A_T1": v["node"] in ("T-6", "T-1"),
         "in_model2B_T1PLUS": v["node"] in ("T-6", "T-1", "T+1"),
         "is_target": v["node"] == "T+2"}
        for f, v in FEATURE_NODES.items()
    ])
    node_df.to_csv(OUT_DIR / "feature_time_nodes.csv", index=False, encoding="utf-8-sig")
    print(f"Feature classification: {len(node_df)} fields tagged")

    # ── Define models ─────────────────────────────────────────────────────
    model_specs = [
        # stage T-6
        ("board_mean_t6", "T6",     FEATS_T6,     lambda: BoardMeanModel()),
        ("lgbm_t6",       "T6",     FEATS_T6,     lambda: make_lgbm(FEATS_T6)),
        # stage T-1  (demo)
        ("ridge_t1",      "T1",     FEATS_T1,     lambda: make_ridge(FEATS_T1)),
        ("lgbm_t1",       "T1",     FEATS_T1,     lambda: make_lgbm(FEATS_T1)),
        # stage T+1
        ("lgbm_t1plus",   "T1PLUS", FEATS_T1PLUS, lambda: make_lgbm(FEATS_T1PLUS)),
    ]

    # ── Backtest ──────────────────────────────────────────────────────────
    all_preds: list[pd.DataFrame] = []
    for name, stage, feats, factory in model_specs:
        print(f"  [{stage}] {name} ...", end=" ", flush=True)
        pred = run_backtest(df, name, stage, factory)
        if pred.empty:
            print("skipped (no valid folds)")
            continue
        all_preds.append(pred)
        m = metrics(pred[TARGET].values, pred["y_pred"].values)
        print(f"n={m['n']}  MAE={m['mae']:.4f}  Spearman={m['spearman']:.4f}")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    # ── Metrics tables ────────────────────────────────────────────────────
    metrics_overall = build_metrics(pred_df)
    metrics_board   = build_metrics(pred_df, "board")
    metrics_year    = build_metrics(pred_df, "listing_year")
    metrics_overall.to_csv(OUT_DIR / "metrics_overall.csv",   index=False, encoding="utf-8-sig")
    metrics_board.to_csv(  OUT_DIR / "metrics_by_board.csv",  index=False, encoding="utf-8-sig")
    metrics_year.to_csv(   OUT_DIR / "metrics_by_year.csv",   index=False, encoding="utf-8-sig")

    # ── Train full models & save ──────────────────────────────────────────
    print("\nTraining full-data models and saving...")
    imp_parts: list[pd.DataFrame] = []
    for name, stage, feats, factory in model_specs:
        print(f"  Saving {name} ...", end=" ", flush=True)
        train_and_save(df, name, feats, factory, MODEL_DIR)
        if "lgbm" in name:
            imp_parts.append(lgbm_importance(df, feats, name))
            print("done (+ importance)")
        else:
            print("done")

    imp_df = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # ── Figures ───────────────────────────────────────────────────────────
    svg_stage_comparison(metrics_overall, fig_dir)
    for name in ["lgbm_t6", "lgbm_t1", "lgbm_t1plus"]:
        svg_importance(imp_df, name, fig_dir)

    # ── Report ────────────────────────────────────────────────────────────
    report = make_report(metrics_overall, metrics_board, imp_df)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "generated": "2026-05-22",
        "version": "three-stage",
        "models_saved": [s[0] for s in model_specs],
        "oos_unique_ipos": int(pred_df["security_code"].nunique()),
        "feature_nodes_csv": str(OUT_DIR / "feature_time_nodes.csv"),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
