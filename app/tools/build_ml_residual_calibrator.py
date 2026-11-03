from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modular_rag.ml_valuation import _load_bundle  # noqa: E402


SOURCE_CSV = Path(os.getenv("VALUATION_DATA_CSV", "data/valuation_enriched.csv"))
OUTPUT_PATH = ROOT / "models" / "ml_residual_calibrator_v1.pkl"

TARGET_COL = "Pre-Money Valuation"
CURRENCY_COL = "Pre-Money Valuation Currency"
CALIBRATION_SHARE = 0.75
SPLIT_SEED = 20260429
PREDICTION_BINS = [-np.inf, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, np.inf]
MIN_SEGMENT_N = 25


def _prepare_model_frame(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(bundle.get("feature_cols") or [])
    cat_cols = list(bundle.get("cat_cols") or [])
    num_cols = list(bundle.get("num_cols") or [])
    text_col = str(bundle.get("text_col") or "model_text_all_derived")

    frame = df[feature_cols].copy()
    for col in cat_cols + [text_col]:
        if col in frame.columns:
            frame[col] = frame[col].fillna("Unknown").astype(str)
    for col in num_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _raw_predict(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    feature_cols = list(bundle.get("feature_cols") or [])
    model_name = str(bundle.get("best_model_name") or "")
    if model_name.startswith("CatBoost"):
        model = bundle.get("catboost_model") or bundle.get("best_model")
        cols = list(bundle.get("catboost_cols") or feature_cols)
        return np.asarray(model.predict(frame[cols]), dtype=float)
    model = bundle.get("best_model")
    preprocess = bundle.get("preprocess")
    return np.asarray(model.predict(preprocess.transform(frame)), dtype=float) if preprocess is not None else np.asarray(model.predict(frame), dtype=float)


def _bucket(pred_log10: float) -> str:
    idx = int(np.digitize([pred_log10], PREDICTION_BINS)[0] - 1)
    lo = PREDICTION_BINS[idx]
    hi = PREDICTION_BINS[idx + 1]
    left = "-inf" if not np.isfinite(lo) else f"{lo:.1f}"
    right = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
    return f"{left}_{right}"


def _summarize_error(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(len(arr)),
        "q10_error_log10": float(np.quantile(arr, 0.10)),
        "q90_error_log10": float(np.quantile(arr, 0.90)),
        "mae_log10": float(mean_absolute_error(np.zeros_like(arr), arr)),
    }


def _group_error_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for keys, group in df.groupby(cols, dropna=False):
        if len(group) < MIN_SEGMENT_N:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        out["||".join(str(k) for k in keys)] = _summarize_error(group["calibrated_error_log10"])
    return out


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)

    bundle = _load_bundle()
    feature_cols = list(bundle.get("feature_cols") or [])
    cat_cols = list(bundle.get("cat_cols") or [])
    text_col = str(bundle.get("text_col") or "model_text_all_derived")
    raw_cols = list(bundle.get("catboost_cols") or feature_cols)
    residual_feature_cols = [col for col in raw_cols if col != text_col] + ["raw_prediction_log10"]

    needed_cols = list(dict.fromkeys(feature_cols + ["Organization Name", "Funding Type", TARGET_COL, CURRENCY_COL]))
    df = pd.read_csv(SOURCE_CSV, usecols=lambda c: c in needed_cols, low_memory=False)
    df["target_usd"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    work = df[
        df[CURRENCY_COL].astype(str).eq("USD")
        & df["target_usd"].between(1_000_000, 50_000_000_000, inclusive="both")
    ].dropna(subset=["Funding Type"]).copy()

    model_frame = _prepare_model_frame(bundle, work)
    raw_pred = _raw_predict(bundle, model_frame)
    work["raw_prediction_log10"] = raw_pred
    work["target_log10"] = np.log10(work["target_usd"])
    work["residual_log10"] = work["target_log10"] - work["raw_prediction_log10"]
    work["prediction_bucket"] = work["raw_prediction_log10"].map(_bucket)

    org_key = work["Organization Name"].fillna("").astype(str).str.lower().str.strip()
    org_hash = pd.util.hash_pandas_object(org_key, index=False).to_numpy(dtype=np.uint64)
    train_mask = (org_hash + SPLIT_SEED) % 10_000 < int(CALIBRATION_SHARE * 10_000)
    test_mask = ~train_mask

    residual_frame = pd.concat(
        [
            model_frame[[col for col in residual_feature_cols if col != "raw_prediction_log10"]].copy(),
            pd.Series(raw_pred, index=model_frame.index, name="raw_prediction_log10"),
        ],
        axis=1,
    ).copy()

    residual_cat_features: list[int] = []
    for col in residual_frame.columns:
        if col in cat_cols:
            residual_frame[col] = residual_frame[col].fillna("Unknown").astype(str)
            residual_cat_features.append(residual_frame.columns.get_loc(col))
        else:
            residual_frame[col] = pd.to_numeric(residual_frame[col], errors="coerce").fillna(0.0)

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1000,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=50,
        random_seed=SPLIT_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        residual_frame.loc[train_mask, residual_feature_cols],
        work.loc[train_mask, "residual_log10"],
        cat_features=residual_cat_features,
    )

    work["residual_prediction_log10"] = model.predict(residual_frame[residual_feature_cols])
    work["calibrated_prediction_log10"] = work["raw_prediction_log10"] + work["residual_prediction_log10"]
    work["calibrated_error_log10"] = work["target_log10"] - work["calibrated_prediction_log10"]

    holdout = work.loc[test_mask].copy()
    y = holdout["target_log10"].to_numpy()
    p = holdout["calibrated_prediction_log10"].to_numpy()
    mae = float(mean_absolute_error(y, p))

    train_work = work.loc[train_mask].copy()
    artifact = {
        "version": "ml_residual_calibrator_v1",
        "source": str(SOURCE_CSV),
        "split": {
            "type": "organization_hash",
            "seed": SPLIT_SEED,
            "calibration_share": CALIBRATION_SHARE,
            "calibration_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
        },
        "model": model,
        "residual_feature_cols": residual_feature_cols,
        "residual_cat_cols": [col for col in residual_feature_cols if col in cat_cols],
        "residual_cat_feature_indices": residual_cat_features,
        "text_col_excluded": text_col,
        "prediction_bins": [None if not np.isfinite(x) else float(x) for x in PREDICTION_BINS],
        "global_error": _summarize_error(train_work["calibrated_error_log10"]),
        "by_stage_error": _group_error_stats(train_work, ["Funding Type"]),
        "by_stage_prediction_bucket_error": _group_error_stats(train_work, ["Funding Type", "prediction_bucket"]),
        "holdout_metrics": {
            "rows": int(len(holdout)),
            "r2": float(r2_score(y, p)),
            "mae_log10": mae,
            "error_x": float(10**mae),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(artifact["holdout_metrics"], indent=2))
    print(f"global_error={artifact['global_error']}")
    print(f"stage error segments={len(artifact['by_stage_error'])}")
    print(f"stage+bucket error segments={len(artifact['by_stage_prediction_bucket_error'])}")


if __name__ == "__main__":
    main()
