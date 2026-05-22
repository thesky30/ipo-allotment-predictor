"""IPO offline subscription rate predictor — demo interface.

Usage
-----
CLI (look up from database):
    python predict.py --code 688XXX
    python predict.py --code 688XXX --stage T1
    python predict.py --code 688XXX --stage T6

CLI (supply raw features for a new IPO not yet in DB):
    python predict.py --features features.json
    python predict.py --features features.json --stage T1

Python API:
    from scripts.predict import predict_from_code, predict_from_dict

    result = predict_from_code("688041")
    result = predict_from_dict({"board": "科创板", "inquiry_oversubscription_ratio": 18.5, ...})

Output
------
{
  "security_code": "...",
  "security_name": "...",
  "stage": "T1",
  "model": "lgbm_t1",
  "log_oversubscription_pred": 7.82,
  "oversubscription_ratio_pred": 2491,
  "subscription_rate_pred_pct": 0.04013,   # 1 / oversubscription * 100
  "subscription_rate_display": "0.040%",
  "confidence": "high"                     # heuristic based on board
}

Stages
------
T6    : Before inquiry opens. Uses only prospectus-level data (申购决策期).
T1    : After inquiry closes, before subscription opens. The DEMO model.
        Uses inquiry results (询价结果) including oversubscription ratio.
T1PLUS: After clawback announcement. Most accurate but requires post-sub data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parents[1]
DATA_DIR       = ROOT / "data" / "processed"
MODEL_DIR      = ROOT / "outputs" / "baseline_models" / "models"
BOARD_MODEL_DIR = ROOT / "outputs" / "board_models" / "models"
DB_PATH        = DATA_DIR / "ipo_offline.db"

# ---------------------------------------------------------------------------
# Import model classes so joblib can unpickle them.
# They live in model_classes.py (stable module path regardless of __main__).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from model_classes import _ColSelector, BoardMeanModel  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Stage → model name mapping
# ---------------------------------------------------------------------------
STAGE_MODEL: dict[str, str] = {
    "T6":     "lgbm_t6",
    "T1":     "lgbm_t1",
    "T1PLUS": "lgbm_t1plus",
}
DEFAULT_STAGE = "T1"

# Board → short code for board-specific model filenames
BOARD_CODES: dict[str, str] = {
    "科创板": "kcb",
    "创业板": "cyb",
    "主板":   "zb",
    "北交所": "bse",
}

# Heuristic confidence labels (based on historical Spearman per board)
_CONFIDENCE: dict[str, dict[str, str]] = {
    "T1": {
        "科创板": "high",
        "创业板": "high",
        "北交所": "medium",
        "主板":   "medium",
    },
    "T6": {
        "科创板": "medium",
        "创业板": "low",
        "北交所": "low",
        "主板":   "low",
    },
    "T1PLUS": {
        "科创板": "high",
        "创业板": "high",
        "北交所": "high",
        "主板":   "medium",
    },
}

# ---------------------------------------------------------------------------
# Model loading (cached in module-level dicts)
# ---------------------------------------------------------------------------
_model_cache: dict[str, dict] = {}
_board_model_cache: dict[str, dict] = {}


def _load_model(stage: str) -> dict:
    """Load global joblib model bundle. Cached after first call."""
    if stage in _model_cache:
        return _model_cache[stage]
    model_name = STAGE_MODEL.get(stage.upper())
    if model_name is None:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGE_MODEL)}")
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run scripts/baseline_models.py first to generate model files."
        )
    bundle = joblib.load(path)
    _model_cache[stage] = bundle
    return bundle


def _load_board_model(board: str, stage: str) -> dict | None:
    """Load a board-specific model bundle if available; return None otherwise.

    Board-specific models exist only for stage=T1.
    Falls back gracefully (returns None) if the file doesn't exist.
    """
    if stage.upper() != "T1":
        return None
    code = BOARD_CODES.get(board)
    if code is None:
        return None
    cache_key = f"{stage}_{code}"
    if cache_key in _board_model_cache:
        return _board_model_cache[cache_key]
    path = BOARD_MODEL_DIR / f"lgbm_t1_{code}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    _board_model_cache[cache_key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def _predict_row(
    row:    pd.DataFrame,
    stage:  str,
    bundle: dict | None = None,
) -> dict[str, Any]:
    """Given a single-row DataFrame with all available fields, run the model.

    Parameters
    ----------
    bundle : If provided, use this pre-loaded model bundle instead of loading
             the global model.  Used by board-specific routing.
    """
    stage = stage.upper()
    if bundle is None:
        bundle = _load_model(stage)

    model      = bundle["model"]
    model_name = bundle["model_name"]
    features   = bundle["features"]

    # Build input DataFrame — fill missing columns with NaN
    row = row.copy()
    for col in features:
        if col not in row.columns:
            row[col] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_pred = float(model.predict(row)[0])

    # Convert: log(oversubscription) → oversubscription → subscription rate
    oversubscription = float(np.exp(log_pred))
    sub_rate_pct     = 100.0 / oversubscription
    sub_rate_display = f"{sub_rate_pct:.4f}%"

    board      = str(row["board"].iloc[0]) if "board" in row.columns else "未知"
    confidence = _CONFIDENCE.get(stage, {}).get(board, "unknown")

    return {
        "stage":                      stage,
        "model":                      model_name,
        "log_oversubscription_pred":  round(log_pred, 4),
        "oversubscription_ratio_pred": round(oversubscription, 1),
        "subscription_rate_pred_pct": round(sub_rate_pct, 6),
        "subscription_rate_display":  sub_rate_display,
        "board":                      board,
        "confidence":                 confidence,
    }


# ---------------------------------------------------------------------------
# Public API — predict from database (historical or recently loaded IPOs)
# ---------------------------------------------------------------------------

def predict_from_code(
    code:                 str,
    stage:                str = DEFAULT_STAGE,
    db_path:              Path | str = DB_PATH,
    prefer_board_model:   bool = False,
) -> dict[str, Any]:
    """Look up an IPO by security_code in the SQLite DB and predict.

    Parameters
    ----------
    code               : Security code, e.g. "688041" or "688041.SH"
    stage              : "T1" (default, demo model) | "T6" | "T1PLUS"
    prefer_board_model : If True (default), use the board-specific model
                         for stage=T1 when available.  Set False to force
                         the global lgbm_t1.

    Returns
    -------
    dict with prediction results + metadata.
    Includes "board_specific_model": True/False to show which model was used.
    """
    code = code.strip()
    # Strip exchange suffix for matching (DB stores "688041.SH"; user may pass "688041")
    code_bare = code.split(".")[0]

    with sqlite3.connect(str(db_path)) as conn:
        row = pd.read_sql(
            "SELECT * FROM ipo_offline_sample "
            "WHERE security_code = ? OR security_code LIKE ? LIMIT 1",
            conn, params=(code, code_bare + ".%")
        )

    if row.empty:
        raise ValueError(
            f"Security code '{code}' not found in database.\n"
            f"DB: {db_path}\n"
            f"For a new IPO, supply features via predict_from_dict() or --features JSON."
        )

    # ── stage-availability guard ────────────────────────────────────────────
    stage = stage.upper()
    if stage in ("T1", "T1PLUS"):
        inq = row.get("inquiry_oversubscription_ratio", pd.Series([None]))
        if pd.isna(inq.iloc[0]):
            warnings.warn(
                f"inquiry_oversubscription_ratio is missing for {code}. "
                f"T-1 model will be less accurate. Consider using stage='T6'.",
                UserWarning, stacklevel=2
            )
    if stage == "T1PLUS":
        cb = row.get("clawback_ratio_pct", pd.Series([None]))
        if pd.isna(cb.iloc[0]):
            warnings.warn(
                f"clawback_ratio_pct is missing for {code}. "
                f"T+1 model will impute. Consider using stage='T1'.",
                UserWarning, stacklevel=2
            )

    # ── Board-specific model routing (T1 only) ──────────────────────────────
    board_bundle = None
    board_val    = str(row["board"].iloc[0]) if "board" in row.columns else ""
    if prefer_board_model:
        board_bundle = _load_board_model(board_val, stage)

    result = _predict_row(row, stage, bundle=board_bundle)
    result["board_specific_model"] = (board_bundle is not None)
    result["security_code"] = str(row["security_code"].iloc[0]) if "security_code" in row.columns else code
    result["security_name"] = str(row["security_name"].iloc[0]) if "security_name" in row.columns else ""

    # Show actual result if labeled (for validation)
    if "log_offline_oversubscription" in row.columns and pd.notna(row["log_offline_oversubscription"].iloc[0]):
        actual_log   = float(row["log_offline_oversubscription"].iloc[0])
        actual_over  = float(np.exp(actual_log))
        actual_rate  = round(100.0 / actual_over, 6)
        result["actual_log_oversubscription"]       = round(actual_log, 4)
        result["actual_oversubscription_ratio"]     = round(actual_over, 1)
        result["actual_subscription_rate_pct"]      = actual_rate
        result["actual_subscription_rate_display"]  = f"{actual_rate:.4f}%"
        result["prediction_error_log"]              = round(abs(result["log_oversubscription_pred"] - actual_log), 4)

    return result


# ---------------------------------------------------------------------------
# Public API — predict from a raw feature dict (new IPO not in DB)
# ---------------------------------------------------------------------------

def predict_from_dict(
    features:             dict[str, Any],
    stage:                str = DEFAULT_STAGE,
    prefer_board_model:   bool = False,
) -> dict[str, Any]:
    """Predict from a feature dictionary.

    Parameters
    ----------
    features : dict mapping feature name → value.
               Keys can be a subset; missing keys are imputed by the model.
               Key T-1 fields for best accuracy:
                 board, inquiry_oversubscription_ratio, inquiry_investors_count,
                 inquiry_allotment_accounts, offer_price_yuan, issue_amount_100m_yuan,
                 total_issue_shares_10k, strategic_allocation_share_pct,
                 subscription_upper_limit_10k, recent_ipo_first_day_return_ma20
    stage              : "T1" (default) | "T6" | "T1PLUS"
    prefer_board_model : Use board-specific model for T1 if available.

    Returns
    -------
    dict with prediction results
    """
    row = pd.DataFrame([features])
    board_bundle = None
    if prefer_board_model:
        board_bundle = _load_board_model(features.get("board", ""), stage)

    result = _predict_row(row, stage, bundle=board_bundle)
    result["board_specific_model"] = (board_bundle is not None)
    result["security_code"] = features.get("security_code", "unknown")
    result["security_name"] = features.get("security_name", "")
    return result


# ---------------------------------------------------------------------------
# Convenience: compute derived T-1 features from raw inputs
# ---------------------------------------------------------------------------

def compute_t1_features(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive T-1 model features from raw inquiry-result data.

    Raw inputs expected (numeric or None/omit if unknown):
        board                           : str  (板块)
        offer_price_yuan                : float (最终发行价)
        offer_price_upper_yuan          : float (询价上限, 科创板)
        offer_price_lower_yuan          : float (询价下限, 科创板)
        quote_price_weighted_avg        : float (询价均价)
        quote_price_median              : float (询价中位数)
        inquiry_subscription_total_10k  : float (询价申购总量 万股)
        offline_issue_before_clawback_10k: float (回拨前网下发行量 万股)
        inquiry_investors_count         : int
        inquiry_allotment_accounts      : int
        comparable_pe_avg_ex_nonrecurring: float
        ipo_pe_diluted                  : float

    Returns the raw dict with derived fields added.
    """
    out = dict(raw)

    # inquiry_oversubscription_ratio
    total = raw.get("inquiry_subscription_total_10k")
    offline = raw.get("offline_issue_before_clawback_10k")
    if total is not None and offline and offline > 0:
        out["inquiry_oversubscription_ratio"] = float(total) / float(offline)

    # quote_price_vs_offer
    qwa = raw.get("quote_price_weighted_avg") or raw.get("quote_price_median")
    op  = raw.get("offer_price_yuan")
    if qwa and op and op > 0:
        out["quote_price_vs_offer"] = float(qwa) / float(op)

    # offer_price_position_in_range  (科创板)
    upper = raw.get("offer_price_upper_yuan")
    lower = raw.get("offer_price_lower_yuan")
    if op and upper and lower and (upper - lower) > 0:
        out["offer_price_position_in_range"] = (float(op) - float(lower)) / (float(upper) - float(lower))
        out["offer_price_range_pct"] = (float(upper) - float(lower)) / float(lower)

    # pe_vs_comparable
    comp_pe = raw.get("comparable_pe_avg_ex_nonrecurring")
    ipo_pe  = raw.get("ipo_pe_diluted")
    if comp_pe and comp_pe > 0 and ipo_pe:
        out["pe_vs_comparable"] = float(ipo_pe) / float(comp_pe)

    return out


# ---------------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------------

def print_result(result: dict[str, Any]) -> None:
    code  = result.get("security_code", "")
    name  = result.get("security_name", "")
    board = result.get("board", "")
    stage = result.get("stage", "")
    model = result.get("model", "")
    conf  = result.get("confidence", "")

    sep = "─" * 56
    board_flag = " [板块专项]" if result.get("board_specific_model") else " [全局]"
    print(f"\n{sep}")
    print(f"  {name}  ({code})  [{board}]")
    print(f"  模型：{model}{board_flag}  阶段：{stage}  置信度：{conf}")
    print(sep)
    print(f"  预测网下超额认购倍数：{result['oversubscription_ratio_pred']:>10,.0f}x")
    print(f"  预测网下中签率：       {result['subscription_rate_display']:>12}")
    if "actual_subscription_rate_display" in result:
        print(f"  实际网下中签率：       {result['actual_subscription_rate_display']:>12}")
        print(f"  预测误差 (log)：       {result['prediction_error_log']:>12.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IPO offline subscription rate predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --code 688041
  python predict.py --code 300257 --stage T6
  python predict.py --features my_ipo.json --stage T1
  python predict.py --code 688041 --json        # output raw JSON
        """,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--code",     metavar="CODE",
                     help="security code, e.g. 688041 or 688041.SH")
    grp.add_argument("--features", metavar="JSON_FILE",
                     help="path to a JSON file with feature values")
    p.add_argument("--stage", default=DEFAULT_STAGE,
                   choices=["T6", "T1", "T1PLUS"],
                   help=f"prediction stage (default: {DEFAULT_STAGE})")
    p.add_argument("--json", action="store_true",
                   help="output raw JSON instead of formatted text")
    p.add_argument("--board-model", dest="use_board", action="store_true",
                   help="use board-specific model (default: global model)")
    p.add_argument("--db", default=str(DB_PATH), metavar="DB_PATH",
                   help="path to SQLite database (default: data/processed/ipo_offline.db)")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    prefer_board = args.use_board

    try:
        if args.code:
            result = predict_from_code(
                args.code, stage=args.stage, db_path=args.db,
                prefer_board_model=prefer_board,
            )
        else:
            feat_path = Path(args.features)
            if not feat_path.exists():
                print(f"ERROR: features file not found: {feat_path}", file=sys.stderr)
                sys.exit(1)
            raw = json.loads(feat_path.read_text(encoding="utf-8"))
            raw = compute_t1_features(raw)
            result = predict_from_dict(raw, stage=args.stage,
                                       prefer_board_model=prefer_board)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)

    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
