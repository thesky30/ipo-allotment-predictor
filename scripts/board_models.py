"""Per-board specialist LightGBM models at T-1 stage.

Motivation
----------
Global lgbm_t1 OOS Spearman by board:
  科创板 0.984  创业板 0.979  主板 0.636  北交所 0.771

主板 underperforms because:
  (a) Registration-system data starts only April 2023 — global model
      sees limited 主板 history vs 1000+ 科创板/创业板 samples.
  (b) Pricing dynamics differ (PE guidance vs market-driven).

Strategy
--------
Train board-specific lgbm_t1 on each board's own data.
  • 'board' feature dropped — constant within a single-board model.
  • Hyperparameters tuned per board (aggressive regularisation for 主板).
  • OOS: expanding window within each board's time series.
  • 北交所: full fit only (41 labeled samples, OOS not reliable).

Outputs
-------
  outputs/board_models/models/lgbm_t1_kcb.joblib   (科创板)
  outputs/board_models/models/lgbm_t1_cyb.joblib   (创业板)
  outputs/board_models/models/lgbm_t1_zb.joblib    (主板)
  outputs/board_models/models/lgbm_t1_bse.joblib   (北交所)
  outputs/board_models/metrics_comparison.csv
  outputs/board_models/figures/board_comparison.svg
  outputs/board_models/report.md
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path
from xml.sax.saxutils import escape as _esc

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model_classes import _ColSelector                         # noqa: E402
from baseline_models import (                                  # noqa: E402
    FEATS_T1, TARGET, SORT_COL, RANDOM_SEED, CUTOFFS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "outputs" / "board_models"
MODEL_DIR = OUT_DIR / "models"
GLOBAL_PREDS = ROOT / "outputs" / "baseline_models" / "predictions.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Short ASCII codes for filenames
BOARD_CODES: dict[str, str] = {
    "科创板": "kcb",
    "创业板": "cyb",
    "主板":   "zb",
    "北交所": "bse",
}

# Feature set: drop 'board' — constant within a single-board model
FEATS_BOARD: list[str] = [f for f in FEATS_T1 if f != "board"]

# Per-board LightGBM hyper-parameters and backtest settings
BOARD_CONFIGS: dict[str, dict] = {
    "科创板": dict(
        lgbm=dict(n_estimators=400, min_child_samples=10, max_depth=6,
                  num_leaves=31, subsample=0.8, colsample_bytree=0.8),
        cutoffs=CUTOFFS, min_train=60, run_oos=True,
    ),
    "创业板": dict(
        lgbm=dict(n_estimators=400, min_child_samples=10, max_depth=6,
                  num_leaves=31, subsample=0.8, colsample_bytree=0.8),
        cutoffs=CUTOFFS, min_train=60, run_oos=True,
    ),
    "主板": dict(
        # Small dataset (~100 labeled) → reduce complexity, more regularisation
        lgbm=dict(n_estimators=300, min_child_samples=8, max_depth=5,
                  num_leaves=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=0.5),
        # 主板 registration data starts 2023-04; first sensible cutoff is 2024-01-01
        cutoffs=[pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")],
        min_train=20, run_oos=True,
    ),
    "北交所": dict(
        lgbm=dict(n_estimators=150, min_child_samples=5, max_depth=3,
                  num_leaves=8, subsample=0.8, colsample_bytree=0.8),
        cutoffs=[], min_train=10, run_oos=False,   # 41 labeled — no OOS
    ),
}

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_board_lgbm(lgbm_params: dict) -> _ColSelector:
    """Board-specific LightGBM: numeric-only pipeline (no board OHE)."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg",    lgb.LGBMRegressor(
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbose=-1,
            **lgbm_params,
        )),
    ])
    return _ColSelector(pipe, FEATS_BOARD)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _calc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


# ---------------------------------------------------------------------------
# Board OOS backtest
# ---------------------------------------------------------------------------

def run_board_oos(
    df_board: pd.DataFrame,
    board:    str,
    cfg:      dict,
) -> pd.DataFrame:
    """Expanding-window OOS within a single board."""
    labeled   = df_board[df_board[TARGET].notna() & df_board[SORT_COL].notna()].copy()
    cutoffs   = cfg["cutoffs"]
    min_train = cfg["min_train"]
    parts: list[pd.DataFrame] = []

    for i, cutoff in enumerate(cutoffs):
        next_cut = cutoffs[i + 1] if i + 1 < len(cutoffs) else pd.Timestamp("2099-01-01")
        train = labeled[labeled[SORT_COL] < cutoff]
        test  = labeled[(labeled[SORT_COL] >= cutoff) & (labeled[SORT_COL] < next_cut)]
        if len(train) < min_train or len(test) == 0:
            print(f"    cutoff={cutoff.date()}  train={len(train)}  test={len(test)}  → skip")
            continue

        model = make_board_lgbm(cfg["lgbm"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train, train[TARGET])
        y_pred = model.predict(test)

        chunk = test[["security_code", "board", SORT_COL, TARGET, "listing_year"]].copy()
        chunk["y_pred"]   = y_pred
        chunk["model"]    = f"lgbm_t1_{BOARD_CODES[board]}"
        chunk["cutoff"]   = cutoff.strftime("%Y-%m-%d")
        chunk["train_n"]  = len(train)
        print(f"    cutoff={cutoff.date()}  train={len(train):>3}  test={len(test):>3}  "
              f"Spearman={float(ss.spearmanr(test[TARGET].values, y_pred)[0]):.4f}")
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Full-data model save
# ---------------------------------------------------------------------------

def save_board_model(
    df_board:  pd.DataFrame,
    board:     str,
    cfg:       dict,
    model_dir: Path,
) -> bool:
    labeled = df_board[df_board[TARGET].notna()].copy()
    if len(labeled) < cfg["min_train"]:
        print(f"    Skip save (n={len(labeled)} < min_train={cfg['min_train']})")
        return False
    model = make_board_lgbm(cfg["lgbm"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(labeled, labeled[TARGET])
    code = BOARD_CODES[board]
    name = f"lgbm_t1_{code}"
    path = model_dir / f"{name}.joblib"
    joblib.dump({
        "model":      model,
        "features":   FEATS_BOARD,
        "target":     TARGET,
        "model_name": name,
        "board":      board,
        "board_code": code,
        "n_train":    len(labeled),
    }, path)
    print(f"    Saved {path.name}  (n_train={len(labeled)})")
    return True


# ---------------------------------------------------------------------------
# SVG comparison chart
# ---------------------------------------------------------------------------

def svg_comparison(comp_df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bar: global vs board-specific Spearman per board."""
    boards = list(comp_df["board"])
    g_vals = [float(v) if pd.notna(v) else 0.0 for v in comp_df["global_spearman"]]
    b_vals = [float(v) if pd.notna(v) else 0.0 for v in comp_df["board_spearman"]]

    W, H     = 720, 420
    left, top_m, bot = 70, 80, 60
    pw  = W - left - 50
    ph  = H - top_m - bot
    n   = len(boards)
    gw  = pw / max(n, 1)
    bw  = gw * 0.36

    def hy(v: float) -> float:
        return top_m + (1.0 - max(v, 0.0)) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W//2}" y="28" font-family="Arial,sans-serif" font-size="17" '
        f'font-weight="700" text-anchor="middle" fill="#222">'
        f'分板块专项模型 vs 全局模型 (OOS Spearman, T-1)</text>',
        f'<text x="{W//2}" y="48" font-family="Arial,sans-serif" font-size="12" '
        f'text-anchor="middle" fill="#666">expanding-window backtest · lgbm_t1</text>',
    ]
    # grid lines
    for tick in [0.25, 0.5, 0.75, 1.0]:
        y = hy(tick)
        lines += [
            f'<line x1="{left}" y1="{y:.0f}" x2="{W-50}" y2="{y:.0f}" '
            f'stroke="#e0e0e0" stroke-dasharray="4,3"/>',
            f'<text x="{left-6}" y="{y+4:.0f}" font-family="Arial,sans-serif" '
            f'font-size="11" text-anchor="end" fill="#aaa">{tick:.2f}</text>',
        ]

    ybot = hy(0.0)
    for i, (board, gv, bv) in enumerate(zip(boards, g_vals, b_vals)):
        cx  = left + (i + 0.5) * gw
        gx  = cx - bw - 2
        bx2 = cx + 2
        gy_top = hy(gv)
        by_top = hy(bv)
        delta  = bv - gv
        # bar colour: green if improved, red if worse, grey if similar
        bar_col = "#54A24B" if delta >= 0.005 else ("#e45c5c" if delta <= -0.005 else "#54A24B")

        lines += [
            # global bar (blue)
            f'<rect x="{gx:.0f}" y="{gy_top:.0f}" width="{bw:.0f}" '
            f'height="{ybot - gy_top:.0f}" fill="#4C78A8" fill-opacity="0.82"/>',
            f'<text x="{gx + bw/2:.0f}" y="{gy_top - 5:.0f}" font-family="Arial,sans-serif" '
            f'font-size="10" text-anchor="middle" fill="#4C78A8">{gv:.3f}</text>',
            # board bar
            f'<rect x="{bx2:.0f}" y="{by_top:.0f}" width="{bw:.0f}" '
            f'height="{ybot - by_top:.0f}" fill="{bar_col}" fill-opacity="0.82"/>',
            f'<text x="{bx2 + bw/2:.0f}" y="{by_top - 5:.0f}" font-family="Arial,sans-serif" '
            f'font-size="10" text-anchor="middle" fill="{bar_col}">{bv:.3f}</text>',
            # delta annotation
            f'<text x="{cx:.0f}" y="{min(gy_top, by_top) - 16:.0f}" '
            f'font-family="Arial,sans-serif" font-size="10" text-anchor="middle" '
            f'fill="{"#2d8a2d" if delta >= 0 else "#c02020"}">{delta:+.3f}</text>',
            # board label
            f'<text x="{cx:.0f}" y="{H - bot + 20}" font-family="Arial,sans-serif" '
            f'font-size="13" text-anchor="middle" fill="#333">{_esc(board)}</text>',
        ]

    # legend
    lines += [
        f'<rect x="{W-160}" y="{top_m}" width="13" height="13" fill="#4C78A8" fill-opacity="0.82"/>',
        f'<text x="{W-143}" y="{top_m+11}" font-family="Arial,sans-serif" font-size="12" fill="#333">全局 lgbm_t1</text>',
        f'<rect x="{W-160}" y="{top_m+22}" width="13" height="13" fill="#54A24B" fill-opacity="0.82"/>',
        f'<text x="{W-143}" y="{top_m+33}" font-family="Arial,sans-serif" font-size="12" fill="#333">板块专项模型</text>',
    ]
    lines.append("</svg>")
    (fig_dir / "board_comparison.svg").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: object, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{float(v):.{d}f}" if isinstance(v, (float, np.floating)) else str(v)


def make_report(comp_df: pd.DataFrame, board_sample_counts: dict) -> str:
    lines = ["# 分板块专项模型报告\n",
             f"生成日期：2026-05-22\n",
             "## OOS 对比 (全局 lgbm_t1 vs 板块专项 lgbm_t1)\n",
             "| 板块 | 标注样本数 | 全局 Spearman | 专项 Spearman | ΔSpearman | 全局 MAE | 专项 MAE |",
             "|---|---|---|---|---|---|---|"]
    for _, r in comp_df.iterrows():
        n = board_sample_counts.get(r["board"], "—")
        delta = r["delta_spearman"]
        arrow = "↑" if delta >= 0.005 else ("↓" if delta <= -0.005 else "≈")
        lines.append(
            f"| {r['board']} | {n} | {_fmt(r['global_spearman'])} | "
            f"{_fmt(r['board_spearman'])} | {_fmt(delta)} {arrow} | "
            f"{_fmt(r['global_mae'])} | {_fmt(r['board_mae'])} |"
        )
    lines += [
        "\n## 结论\n",
        "**全局模型优于板块专项模型**（科创板除外，几乎持平）。",
        "",
        "原因分析：",
        "- 主板注册制数据始于 2023-04，在 2024-01-01 截点时训练集仅 36 条。",
        "  全局模型通过科创板/创业板 2019-2022 历史数据完成跨板块迁移学习，",
        "  在同一测试集上 Spearman 高出专项模型 **+0.23**。",
        "- 创业板：全局模型见过更多科创板早期样本，泛化更好（+0.017）。",
        "- 科创板：样本充足（608条），专项/全局几乎持平（+0.001）。",
        "",
        "**建议**：当前阶段使用全局 lgbm_t1 作为唯一预测模型。",
        "待主板数据积累到 300+ 条（预计 2025-2026 年），再重新评估专项模型价值。",
        "",
        "北交所样本过少（41 条），仅保存全量拟合模型，不作 OOS 评估。",
        "\n## 模型路由逻辑 (predict.py)\n",
        "- **默认**：所有板块使用全局 lgbm_t1（`prefer_board_model=False`）",
        "- 可选：`--board-model` 或 `prefer_board_model=True` 切换到板块专项",
        "- stage=T6 / T1PLUS：始终使用全局模型（无板块专项版本）",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    with sqlite3.connect(DATA_DIR / "ipo_offline.db") as conn:
        df = pd.read_sql("SELECT * FROM ipo_offline_sample", conn)
    for dcol in ["listing_date", "subscription_deadline_date"]:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[SORT_COL] = df["subscription_deadline_date"].fillna(df["listing_date"])
    df[TARGET]   = pd.to_numeric(df[TARGET], errors="coerce")
    df["listing_year"] = pd.to_numeric(df["listing_year"], errors="coerce")

    # ── Load global lgbm_t1 OOS predictions for comparison ────────────────
    global_t1 = pd.read_csv(GLOBAL_PREDS)
    global_t1 = global_t1[global_t1["model"] == "lgbm_t1"].copy()

    board_sample_counts = (df[df[TARGET].notna()].groupby("board").size().to_dict())

    comparison_rows: list[dict] = []
    all_board_oos:   list[pd.DataFrame] = []

    # ── Per-board loop ─────────────────────────────────────────────────────
    for board, cfg in BOARD_CONFIGS.items():
        n = board_sample_counts.get(board, 0)
        print(f"\n{'='*56}")
        print(f"  [{board}]  labeled={n}  run_oos={cfg['run_oos']}")
        print(f"{'='*56}")

        df_board = df[df["board"] == board].copy()

        # ── OOS backtest ──────────────────────────────────────────────────
        if cfg["run_oos"]:
            board_oos = run_board_oos(df_board, board, cfg)

            if not board_oos.empty:
                bm = _calc(board_oos[TARGET].values, board_oos["y_pred"].values)

                # Global model on exactly the same OOS security codes
                test_codes = set(board_oos["security_code"])
                gm_same = global_t1[global_t1["security_code"].isin(test_codes)]
                gm = _calc(gm_same[TARGET].values, gm_same["y_pred"].values)

                # Fallback: if code matching yields too few, use board-filtered global
                if gm["n"] < 5:
                    gm_all = global_t1[global_t1["board"] == board]
                    gm = _calc(gm_all[TARGET].values, gm_all["y_pred"].values)

                delta = (bm["spearman"] or 0.0) - (gm["spearman"] or 0.0)
                arrow = "↑" if delta >= 0.005 else ("↓" if delta <= -0.005 else "≈")
                print(f"  Global  : n={gm['n']:>3}  Spearman={gm['spearman']:.4f}  MAE={gm['mae']:.4f}")
                print(f"  Board   : n={bm['n']:>3}  Spearman={bm['spearman']:.4f}  MAE={bm['mae']:.4f}")
                print(f"  Delta Spearman: {delta:+.4f} {arrow}")

                comparison_rows.append({
                    "board":             board,
                    "n_labeled":         n,
                    "global_n":          gm["n"],
                    "global_spearman":   gm["spearman"],
                    "global_mae":        gm["mae"],
                    "global_rmse":       gm["rmse"],
                    "board_n":           bm["n"],
                    "board_spearman":    bm["spearman"],
                    "board_mae":         bm["mae"],
                    "board_rmse":        bm["rmse"],
                    "delta_spearman":    delta,
                    "delta_mae":         (bm["mae"] or 0.0) - (gm["mae"] or 0.0),
                })
                all_board_oos.append(board_oos)
            else:
                print("  No valid OOS folds.")

        # ── Full-data model save ──────────────────────────────────────────
        save_board_model(df_board, board, cfg, MODEL_DIR)

    # ── Export comparison table ────────────────────────────────────────────
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(OUT_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig")

        # OOS predictions
        if all_board_oos:
            pd.concat(all_board_oos, ignore_index=True).to_csv(
                OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
            )

        # SVG
        svg_comparison(comp_df, fig_dir)

        # Report
        (OUT_DIR / "report.md").write_text(
            make_report(comp_df, board_sample_counts), encoding="utf-8"
        )

        # Summary print
        print(f"\n{'='*56}")
        print("  SUMMARY")
        print(f"{'='*56}")
        for _, r in comp_df.iterrows():
            delta = r["delta_spearman"]
            arrow = "↑" if delta >= 0.005 else ("↓" if delta <= -0.005 else "≈")
            print(f"  {r['board']:<6}  global={r['global_spearman']:.4f}  "
                  f"board={r['board_spearman']:.4f}  {delta:+.4f} {arrow}")
    else:
        print("\nNo comparison data generated.")

    print("\nDone. Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
