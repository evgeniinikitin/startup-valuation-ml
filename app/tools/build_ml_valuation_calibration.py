from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modular_rag.ml_valuation import _load_bundle  # noqa: E402


SOURCE_CSV = Path(os.getenv("VALUATION_DATA_CSV", "data/valuation_enriched.csv"))
OUTPUT_JSON = ROOT / "models" / "ml_valuation_calibration_v1.json"
HOLDOUT_CSV = ROOT.parent / "ml_benchmark_holdout_enriched_v6.csv"

TARGET_COL = "Pre-Money Valuation"
CURRENCY_COL = "Pre-Money Valuation Currency"
PREDICTION_BINS = [-math.inf, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, math.inf]
MIN_SEGMENT_N = 25
MAX_ABS_CORRECTION_LOG10 = 0.75
CALIBRATION_SHARE = 0.75
SPLIT_SEED = 20260429


def _predict_raw(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    feature_cols = list(bundle.get("feature_cols") or [])
    cat_cols = list(bundle.get("cat_cols") or [])
    num_cols = list(bundle.get("num_cols") or [])
    text_col = str(bundle.get("text_col") or "model_text_all_derived")

    frame = frame[feature_cols].copy()
    for col in cat_cols + [text_col]:
        if col in frame.columns:
            frame[col] = frame[col].fillna("Unknown").astype(str)
    for col in num_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    model_name = str(bundle.get("best_model_name") or "")
    if model_name.startswith("CatBoost"):
        model = bundle.get("catboost_model") or bundle.get("best_model")
        cols = list(bundle.get("catboost_cols") or feature_cols)
        return np.asarray(model.predict(frame[cols]), dtype=float)

    model = bundle.get("best_model")
    preprocess = bundle.get("preprocess")
    if preprocess is not None:
        return np.asarray(model.predict(preprocess.transform(frame)), dtype=float)
    return np.asarray(model.predict(frame), dtype=float)


def _bucket(pred_log10: float) -> str:
    idx = int(np.digitize([pred_log10], PREDICTION_BINS)[0] - 1)
    lo = PREDICTION_BINS[idx]
    hi = PREDICTION_BINS[idx + 1]
    left = "-inf" if not math.isfinite(lo) else f"{lo:.1f}"
    right = "inf" if not math.isfinite(hi) else f"{hi:.1f}"
    return f"{left}_{right}"


def _summarize(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    correction = float(np.median(arr))
    correction = max(-MAX_ABS_CORRECTION_LOG10, min(MAX_ABS_CORRECTION_LOG10, correction))
    return {
        "n": int(len(arr)),
        "correction_log10": correction,
        "q10_residual_log10": float(np.quantile(arr, 0.10)),
        "q90_residual_log10": float(np.quantile(arr, 0.90)),
        "mae_log10_after_correction": float(np.median(np.abs(arr - correction))),
    }


def _group_stats(df: pd.DataFrame, cols: list[str], min_n: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for keys, group in df.groupby(cols, dropna=False):
        if len(group) < min_n:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = "||".join(str(k) for k in keys)
        out[key] = _summarize(group["residual_log10"])
    return out


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)

    bundle = _load_bundle()
    feature_cols = list(bundle.get("feature_cols") or [])
    needed_cols = list(dict.fromkeys(feature_cols + ["Organization Name", "Funding Type", TARGET_COL, CURRENCY_COL]))

    df = pd.read_csv(SOURCE_CSV, usecols=lambda c: c in needed_cols, low_memory=False)
    df["target_usd"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    work = df[
        df[CURRENCY_COL].astype(str).eq("USD")
        & df["target_usd"].notna()
        & (df["target_usd"] > 0)
    ].copy()
    work = work.dropna(subset=["Funding Type"])
    work = work[work["target_usd"].between(1_000_000, 50_000_000_000, inclusive="both")].copy()

    raw_pred = _predict_raw(bundle, work)
    work["raw_prediction_log10"] = raw_pred
    work["target_log10"] = np.log10(work["target_usd"])
    work["residual_log10"] = work["target_log10"] - work["raw_prediction_log10"]
    work = work[np.isfinite(work["residual_log10"])].copy()
    work["prediction_bucket"] = work["raw_prediction_log10"].map(_bucket)

    org_key = work["Organization Name"].fillna("").astype(str).str.lower().str.strip()
    org_hash = pd.util.hash_pandas_object(org_key, index=False).to_numpy(dtype=np.uint64)
    split_threshold = int(CALIBRATION_SHARE * 10_000)
    work["split"] = np.where((org_hash + SPLIT_SEED) % 10_000 < split_threshold, "calibration", "holdout")
    calibration_work = work[work["split"].eq("calibration")].copy()
    holdout_work = work[work["split"].eq("holdout")].copy()
    if calibration_work.empty or holdout_work.empty:
        raise RuntimeError("Calibration split produced an empty partition")

    calibration = {
        "version": "ml_valuation_calibration_v1",
        "source": str(SOURCE_CSV),
        "target_col": TARGET_COL,
        "currency": "USD",
        "split": {
            "type": "organization_hash",
            "seed": SPLIT_SEED,
            "calibration_share": CALIBRATION_SHARE,
            "calibration_rows": int(len(calibration_work)),
            "holdout_rows": int(len(holdout_work)),
        },
        "n_rows": int(len(calibration_work)),
        "prediction_bins": PREDICTION_BINS,
        "min_segment_n": MIN_SEGMENT_N,
        "max_abs_correction_log10": MAX_ABS_CORRECTION_LOG10,
        "global": _summarize(calibration_work["residual_log10"]),
        "by_stage": _group_stats(calibration_work, ["Funding Type"], MIN_SEGMENT_N),
        "by_stage_prediction_bucket": _group_stats(calibration_work, ["Funding Type", "prediction_bucket"], MIN_SEGMENT_N),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_cols = [col for col in needed_cols if col in holdout_work.columns]
    holdout_work[holdout_cols].to_csv(HOLDOUT_CSV, index=False)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {HOLDOUT_CSV}")
    print(f"rows={calibration['n_rows']}")
    print(f"holdout_rows={len(holdout_work)}")
    print(f"global={calibration['global']}")
    print(f"stage segments={len(calibration['by_stage'])}")
    print(f"stage+bucket segments={len(calibration['by_stage_prediction_bucket'])}")


if __name__ == "__main__":
    main()
