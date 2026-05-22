"""Baseline model training and time-series backtesting for IPO offline
oversubscription prediction.

Target  : log_offline_oversubscription  (= log(网下超额认购倍数))
Backtest: expanding-window CV ordered by subscription_deadline_date.
Cutoffs : 2022-01-01 / 2023-01-01 / 2024-01-01 / 2025-01-01.
          Each fold trains on all data before the cutoff and tests on the
          following 12 months.  Minimum 80 training samples per fold.

Models
------
board_mean          Board-level mean from training set (no-model baseline).
board_recent_mean   Rolling mean of last 20 same-board IPOs before test date.
ridge_basic         Ridge regression, pre-subscription fields only (no inquiry).
ridge_inquiry       Ridge regression, full inquiry-time feature set.
lgbm_inquiry        LightGBM, full inquiry-time feature set.

Feature discipline
------------------
FEATS_BASIC uses only fields available from main Wind export files
(valuation, size, strategic allocation, exclusion ratios).
FEATS_INQUIRY adds the inquiry-time supplement fields (subscription volume,
investor counts, price statistics, market-heat rolling average).
Both sets exclude any post-subscription or label fields.

Outputs  →  outputs/baseline_models/
-------------------------------------
predictions.csv         OOS predictions for all folds, all models.
metrics_overall.csv     Aggregate metrics per model.
metrics_by_board.csv    Metrics broken down by board.
metrics_by_year.csv     Metrics broken down by listing year.
feature_importance.csv  LightGBM gain/split importance (full-data fit).
report.md               Human-readable summary.
figures/                SVG scatter and bar charts.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "baseline_models"

TARGET = "log_offline_oversubscription"
SORT_COL = "sort_date"   # subscription_deadline_date, falling back to listing_date
RANDOM_SEED = 42
MIN_TRAIN = 80

CUTOFFS = [
    pd.Timestamp("2022-01-01"),
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2025-01-01"),
]

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

FEATS_BASIC = [
    "board",
    "offer_price_yuan",
    "issue_amount_100m_yuan",
    "ipo_pe_diluted",
    "issue_pb",
    "pe_vs_industry",
    "pe_vs_comparable",
    "strategic_allocation_share_pct",
    "excluded_subscription_share_pct",
    "high_price_excluded_subscription_share_pct",
    "offline_issue_before_share_pct",
]

FEATS_INQUIRY = FEATS_BASIC + [
    "inquiry_subscription_total_10k",
    "inquiry_investors_count",
    "inquiry_allotment_accounts",
    "inquiry_oversubscription_ratio",   # inquiry_total / offline_issue_before_clawback
    "subscription_upper_limit_10k",
    "subscription_lower_limit_10k",
    "subscription_step_10k",
    "quote_price_weighted_avg",
    "quote_price_median",
    "quote_price_vs_offer",             # quote_price_weighted_avg / offer_price
    "offer_price_upper_yuan",           # 科创板 only; NaN elsewhere
    "offer_price_lower_yuan",           # 科创板 only; NaN elsewhere
    "offer_price_range_pct",            # 科创板 only
    "offer_price_position_in_range",    # 科创板 only
    "recent_ipo_first_day_return_ma20", # rolling market heat (past 20 IPOs, same board)
]

CAT_COLS = ["board"]


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class BoardMeanModel:
    """Predict the training-set board mean (simplest no-model baseline)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoardMeanModel":
        self._global = float(y.mean())
        self._means = y.groupby(X["board"].values).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._means.get(b, self._global) for b in X["board"]])


class BoardRecentMeanModel:
    """Predict mean of last N same-board IPOs before the test sample's date."""

    def __init__(self, n: int = 20) -> None:
        self.n = n

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoardRecentMeanModel":
        self._train = X[[SORT_COL, "board"]].copy()
        self._train["_y"] = y.values
        self._global = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X[[SORT_COL, "board"]].iterrows():
            prior = self._train[
                (self._train["board"] == row["board"]) &
                (self._train[SORT_COL] < row[SORT_COL])
            ]
            recent = prior.nlargest(self.n, SORT_COL)
            preds.append(recent["_y"].mean() if len(recent) >= 3 else self._global)
        return np.array(preds)


class _SelectWrapper:
    """Thin wrapper so sklearn pipelines receive only their declared feature columns."""

    def __init__(self, pipeline: Pipeline, feature_cols: list[str]) -> None:
        self._pipe = pipeline
        self._cols = feature_cols

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_SelectWrapper":
        self._pipe.fit(X[self._cols], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipe.predict(X[self._cols])


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in CAT_COLS]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )


def make_ridge(feature_cols: list[str], alpha: float = 1.0) -> _SelectWrapper:
    pipe = Pipeline([
        ("pre", _make_preprocessor(feature_cols)),
        ("reg", Ridge(alpha=alpha)),
    ])
    return _SelectWrapper(pipe, feature_cols)


def make_lgbm(feature_cols: list[str]) -> _SelectWrapper:
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in CAT_COLS]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )
    pipe = Pipeline([
        ("pre", pre),
        ("reg", lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, num_leaves=31, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbose=-1,
        )),
    ])
    return _SelectWrapper(pipe, feature_cols)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    model_name: str,
    make_model_fn,
    cutoffs: list[pd.Timestamp] = CUTOFFS,
    min_train: int = MIN_TRAIN,
) -> pd.DataFrame:
    """Expanding-window backtest; returns DataFrame of OOS predictions."""
    labeled = df[df[TARGET].notna()].copy()
    results: list[pd.DataFrame] = []

    for i, cutoff in enumerate(cutoffs):
        next_cutoff = (
            cutoffs[i + 1] if i + 1 < len(cutoffs) else pd.Timestamp("2099-01-01")
        )
        train = labeled[labeled[SORT_COL] < cutoff]
        test = labeled[
            (labeled[SORT_COL] >= cutoff) & (labeled[SORT_COL] < next_cutoff)
        ]
        if len(train) < min_train or len(test) == 0:
            continue

        model = make_model_fn()
        model.fit(train, train[TARGET])
        y_pred = model.predict(test)

        out = test[
            ["security_code", "security_name", "board", "listing_date", "listing_year",
             SORT_COL, TARGET]
        ].copy()
        out["y_pred"] = y_pred
        out["model"] = model_name
        out["cutoff"] = cutoff.strftime("%Y-%m-%d")
        out["train_n"] = len(train)
        results.append(out)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        return {"n": n, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
    return {
        "n": n,
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "spearman": float(ss.spearmanr(yt, yp)[0]),
    }


def build_metrics_table(pred_df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    rows = []
    for model, mdf in pred_df.groupby("model"):
        if group_col is None:
            m = compute_metrics(mdf[TARGET].values, mdf["y_pred"].values)
            m["model"] = model
            rows.append(m)
        else:
            for key, gdf in mdf.groupby(group_col):
                m = compute_metrics(gdf[TARGET].values, gdf["y_pred"].values)
                m["model"] = model
                m[group_col] = key
                rows.append(m)
    col_order = (
        (["model"] if group_col is None else ["model", group_col]) +
        ["n", "mae", "rmse", "r2", "spearman"]
    )
    return pd.DataFrame(rows)[col_order].sort_values(
        ["model"] if group_col is None else ["model", group_col]
    )


# ---------------------------------------------------------------------------
# LightGBM feature importance (full-data fit)
# ---------------------------------------------------------------------------

def lgbm_full_fit_importance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    labeled = df[df[TARGET].notna()].copy()
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in CAT_COLS]

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )
    X = pre.fit_transform(labeled[feature_cols])
    y = labeled[TARGET].values

    feat_names = list(pre.get_feature_names_out())

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    return (
        pd.DataFrame({
            "feature": feat_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        })
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _svg_header(w: int, h: int, title: str, subtitle: str = "") -> list[str]:
    from xml.sax.saxutils import escape
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2:.0f}" y="30" font-family="Arial,sans-serif" font-size="18" '
        f'text-anchor="middle" font-weight="700" fill="#222">{escape(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'<text x="{w/2:.0f}" y="50" font-family="Arial,sans-serif" font-size="12" '
            f'text-anchor="middle" fill="#666">{escape(subtitle)}</text>'
        )
    return lines


def svg_scatter(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    """Actual vs predicted scatter for each model, all on one SVG."""
    models = pred_df["model"].unique()
    boards = ["科创板", "创业板", "主板", "北交所"]
    colors = {"科创板": "#4C78A8", "创业板": "#F58518", "主板": "#54A24B", "北交所": "#B279A2"}

    n_models = len(models)
    cell_w, cell_h = 420, 380
    cols = min(n_models, 3)
    rows_g = (n_models + cols - 1) // cols
    W = cols * cell_w + 60
    H = rows_g * cell_h + 100

    body = _svg_header(W, H, "Actual vs Predicted — OOS backtest",
                       "log(offline oversubscription); coloured by board")

    for idx, model in enumerate(models):
        col_i = idx % cols
        row_i = idx // cols
        ox = col_i * cell_w + 50
        oy = row_i * cell_h + 70

        sub = pred_df[pred_df["model"] == model].dropna(subset=[TARGET, "y_pred"])
        if sub.empty:
            continue

        vmin = min(sub[TARGET].min(), sub["y_pred"].min()) - 0.2
        vmax = max(sub[TARGET].max(), sub["y_pred"].max()) + 0.2
        plot_w = cell_w - 80
        plot_h = cell_h - 80

        def sx(v: float) -> float:
            return ox + 40 + (float(v) - vmin) / (vmax - vmin) * plot_w

        def sy(v: float) -> float:
            return oy + 10 + plot_h - (float(v) - vmin) / (vmax - vmin) * plot_h

        # axes
        body += [
            f'<line x1="{ox+40:.0f}" y1="{oy+10:.0f}" x2="{ox+40:.0f}" y2="{oy+10+plot_h:.0f}" stroke="#999"/>',
            f'<line x1="{ox+40:.0f}" y1="{oy+10+plot_h:.0f}" x2="{ox+40+plot_w:.0f}" y2="{oy+10+plot_h:.0f}" stroke="#999"/>',
        ]
        # diagonal reference
        diag_x1, diag_y1 = sx(vmin), sy(vmin)
        diag_x2, diag_y2 = sx(vmax), sy(vmax)
        body.append(f'<line x1="{diag_x1:.1f}" y1="{diag_y1:.1f}" x2="{diag_x2:.1f}" y2="{diag_y2:.1f}" stroke="#ccc" stroke-dasharray="4,3"/>')

        # points
        for _, row in sub.iterrows():
            c = colors.get(row["board"], "#888")
            body.append(f'<circle cx="{sx(row[TARGET]):.1f}" cy="{sy(row["y_pred"]):.1f}" r="3" fill="{c}" fill-opacity="0.7"/>')

        m = compute_metrics(sub[TARGET].values, sub["y_pred"].values)
        label = f'{model}  MAE={m["mae"]:.3f}  Spearman={m["spearman"]:.3f}  n={m["n"]}'
        body.append(
            f'<text x="{ox + cell_w/2:.0f}" y="{oy - 4}" font-family="Arial,sans-serif" '
            f'font-size="12" text-anchor="middle" fill="#333">{label}</text>'
        )

    # Legend
    lx = W - 140
    ly = H - 90
    for i, board in enumerate(boards):
        body += [
            f'<rect x="{lx}" y="{ly + i*20}" width="12" height="12" fill="{colors[board]}"/>',
            f'<text x="{lx+18}" y="{ly + i*20 + 11}" font-family="Arial,sans-serif" font-size="12" fill="#333">{board}</text>',
        ]

    body.append("</svg>")
    (fig_dir / "actual_vs_predicted.svg").write_text("\n".join(body), encoding="utf-8")


def svg_metric_bar(metrics_df: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    """Horizontal bar chart comparing models on a single metric."""
    from xml.sax.saxutils import escape
    df = metrics_df[metrics_df[metric].notna()].copy()
    if df.empty:
        return
    df = df.sort_values(metric, ascending=(metric not in {"r2", "spearman"}))

    W, H = 720, max(280, len(df) * 36 + 100)
    left, right, top, bottom = 220, 60, 70, 50
    pw = W - left - right
    vmin = 0.0
    vmax = float(df[metric].max()) * 1.15

    def bx(v: float) -> float:
        return left + (float(v) - vmin) / max(vmax - vmin, 1e-9) * pw

    row_h = (H - top - bottom) / len(df)
    body = _svg_header(W, H, f"Model comparison — {metric.upper()}", "OOS backtest, all boards")

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tv = vmin + tick * (vmax - vmin)
        tx = bx(tv)
        body += [
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + H - top - bottom}" stroke="#e6e6e6"/>',
            f'<text x="{tx:.1f}" y="{top + H - top - bottom + 18}" font-family="Arial,sans-serif" font-size="11" text-anchor="middle" fill="#666">{tv:.2f}</text>',
        ]

    for i, (_, row) in enumerate(df.iterrows()):
        y = top + i * row_h + row_h * 0.15
        bw = bx(row[metric]) - left
        body += [
            f'<text x="{left - 8}" y="{y + row_h*0.55:.1f}" font-family="Arial,sans-serif" font-size="12" text-anchor="end" fill="#333">{escape(str(row["model"]))}</text>',
            f'<rect x="{left}" y="{y:.1f}" width="{bw:.1f}" height="{row_h*0.7:.1f}" fill="#4C78A8" fill-opacity="0.8"/>',
            f'<text x="{left + bw + 6:.1f}" y="{y + row_h*0.55:.1f}" font-family="Arial,sans-serif" font-size="11" fill="#333">{row[metric]:.4f}</text>',
        ]

    body.append("</svg>")
    (fig_dir / f"metric_{metric}.svg").write_text("\n".join(body), encoding="utf-8")


def svg_feat_importance(imp_df: pd.DataFrame, fig_dir: Path) -> None:
    """Horizontal bar chart of top-20 LightGBM feature importances (gain)."""
    from xml.sax.saxutils import escape
    top = imp_df.head(20).iloc[::-1]
    W, H = 820, 560
    left, right, top_m, bottom = 300, 50, 60, 50
    pw = W - left - right
    vmax = float(top["importance_gain"].max()) * 1.1

    def bx(v: float) -> float:
        return left + float(v) / max(vmax, 1) * pw

    row_h = (H - top_m - bottom) / len(top)
    body = _svg_header(W, H, "LightGBM Feature Importance (gain)",
                       "lgbm_inquiry, trained on all labeled data")
    for i, (_, row) in enumerate(top.iterrows()):
        y = top_m + i * row_h + row_h * 0.1
        bw = bx(row["importance_gain"]) - left
        body += [
            f'<text x="{left-8}" y="{y+row_h*0.6:.1f}" font-family="Arial,sans-serif" font-size="11" text-anchor="end" fill="#333">{escape(str(row["feature"]))}</text>',
            f'<rect x="{left}" y="{y:.1f}" width="{bw:.1f}" height="{row_h*0.8:.1f}" fill="#F58518" fill-opacity="0.85"/>',
        ]
    body.append("</svg>")
    (fig_dir / "feature_importance.svg").write_text("\n".join(body), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(v: object, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    header = "| " + " | ".join(cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def make_report(
    metrics_overall: pd.DataFrame,
    metrics_board: pd.DataFrame,
    metrics_year: pd.DataFrame,
    imp_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> str:
    best = metrics_overall.sort_values("spearman", ascending=False).iloc[0]
    total_oos = len(pred_df["security_code"].unique())
    return f"""# 基准模型回测报告

生成日期：2026-05-22

## 回测设计

- 目标变量：`log_offline_oversubscription` = log(网下超额认购倍数)
- 排序依据：`subscription_deadline_date`（申购截止日），缺失时回退到 `listing_date`
- 回测方式：扩展窗口（expanding window），每折用截止日之前全部数据训练
- 截止日期：2022-01-01 / 2023-01-01 / 2024-01-01 / 2025-01-01
- OOS 样本总数：{total_oos} 只不同 IPO

## 模型

| 模型 | 特征集 | 说明 |
|---|---|---|
| board_mean | — | 训练集中同板块均值（无模型基线）|
| board_recent_mean | — | 同板块最近 20 条样本滚动均值 |
| ridge_basic | FEATS_BASIC | Ridge，无询价字段 |
| ridge_inquiry | FEATS_INQUIRY | Ridge，含询价字段 |
| lgbm_inquiry | FEATS_INQUIRY | LightGBM，含询价字段 |

## 整体 OOS 指标

{md_table(metrics_overall)}

> 最佳模型（Spearman）：**{best['model']}**，Spearman={fmt(best['spearman'])}，MAE={fmt(best['mae'])}

## 分板块 OOS 指标

{md_table(metrics_board)}

## 分年度 OOS 指标

{md_table(metrics_year)}

## LightGBM 特征重要性（Top 15，全量训练）

{md_table(imp_df.head(15)[['feature', 'importance_gain', 'importance_split']], digits=1)}

## 解读要点

- **Spearman 排名相关**是最重要指标，直接反映"哪只新股更值得优先申购"的排序能力。
- `inquiry_oversubscription_ratio`（初步询价超额认购倍数）预计是最强预测因子；如果
  特征重要性显示其他字段更重要，需要重新审视数据质量。
- 北交所由于分布极端（中位超额认购仅 21 倍），单独看其误差具有参考价值。
- Ridge 与 LightGBM 的差距可以量化询价字段对线性 vs 非线性关系的贡献。
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    with sqlite3.connect(DATA_DIR / "ipo_offline.db") as conn:
        df = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)

    # Parse dates
    for dcol in ["listing_date", "subscription_deadline_date"]:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Sort date: prefer subscription_deadline_date, fall back to listing_date
    df[SORT_COL] = df["subscription_deadline_date"].fillna(df["listing_date"])

    # Ensure numeric target
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["listing_year"] = pd.to_numeric(df["listing_year"], errors="coerce")

    labeled = df[df[TARGET].notna()]
    print(f"Labeled samples: {len(labeled)}")
    print(df[SORT_COL].isna().sum(), "samples missing sort_date (excluded from backtest)")

    # --- Define models ---
    models = [
        ("board_mean",        lambda: BoardMeanModel()),
        ("board_recent_mean", lambda: BoardRecentMeanModel(n=20)),
        ("ridge_basic",       lambda: make_ridge(FEATS_BASIC)),
        ("ridge_inquiry",     lambda: make_ridge(FEATS_INQUIRY)),
        ("lgbm_inquiry",      lambda: make_lgbm(FEATS_INQUIRY)),
    ]

    # --- Run backtests ---
    all_preds: list[pd.DataFrame] = []
    for name, factory in models:
        print(f"  Backtesting {name} ...", end=" ", flush=True)
        pred = run_backtest(df, model_name=name, make_model_fn=factory)
        if pred.empty:
            print("no folds produced — skipped")
            continue
        all_preds.append(pred)
        m = compute_metrics(pred[TARGET].values, pred["y_pred"].values)
        print(f"n={m['n']}  MAE={m['mae']:.4f}  Spearman={m['spearman']:.4f}")

    if not all_preds:
        print("No predictions generated. Check cutoffs and data coverage.")
        return

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    # --- Metrics ---
    metrics_overall = build_metrics_table(pred_df, group_col=None)
    metrics_board   = build_metrics_table(pred_df, group_col="board")
    metrics_year    = build_metrics_table(pred_df, group_col="listing_year")

    metrics_overall.to_csv(OUT_DIR / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    metrics_board.to_csv(OUT_DIR / "metrics_by_board.csv", index=False, encoding="utf-8-sig")
    metrics_year.to_csv(OUT_DIR / "metrics_by_year.csv", index=False, encoding="utf-8-sig")

    # --- Feature importance (full data, lgbm_inquiry) ---
    print("  Computing LightGBM feature importance on full dataset ...", end=" ", flush=True)
    imp_df = lgbm_full_fit_importance(df, FEATS_INQUIRY)
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print("done")

    # --- Figures ---
    svg_scatter(pred_df, fig_dir)
    svg_metric_bar(metrics_overall, "mae", fig_dir)
    svg_metric_bar(metrics_overall, "spearman", fig_dir)
    svg_feat_importance(imp_df, fig_dir)

    # --- Report ---
    report = make_report(metrics_overall, metrics_board, metrics_year, imp_df, pred_df)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "generated": "2026-05-22",
        "oos_predictions": len(pred_df),
        "unique_ipos": int(pred_df["security_code"].nunique()),
        "models": [name for name, _ in models],
        "outputs": {
            "predictions": str(OUT_DIR / "predictions.csv"),
            "metrics_overall": str(OUT_DIR / "metrics_overall.csv"),
            "metrics_by_board": str(OUT_DIR / "metrics_by_board.csv"),
            "metrics_by_year":  str(OUT_DIR / "metrics_by_year.csv"),
            "feature_importance": str(OUT_DIR / "feature_importance.csv"),
            "report": str(OUT_DIR / "report.md"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
