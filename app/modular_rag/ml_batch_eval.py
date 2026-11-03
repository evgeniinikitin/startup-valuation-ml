from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from modular_rag.ml_valuation import _load_bundle, predict_ml_from_feature_row


TARGET_CANDIDATES = [
    "Pre-Money Valuation USD (historical)",
    "Pre-Money Valuation USD",
    "Pre-Money Valuation",
    "valuation_usd",
    "target_usd",
]


def _find_target_col(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in df.columns:
        return requested
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Target column not found. Expected one of: {TARGET_CANDIDATES}")


def _target_mask(df: pd.DataFrame, target_col: str) -> pd.Series:
    target = pd.to_numeric(df[target_col], errors="coerce")
    mask = target.notna() & (target > 0)
    if target_col == "Pre-Money Valuation" and "Pre-Money Valuation Currency" in df.columns:
        mask &= df["Pre-Money Valuation Currency"].eq("USD")
    return mask


def _stage_metrics(out: pd.DataFrame) -> list[dict[str, Any]]:
    if "Funding Type" not in out.columns:
        return []
    rows: list[dict[str, Any]] = []
    for stage, group in out.groupby("Funding Type"):
        if len(group) < 10:
            continue
        y = group["target_log10"].to_numpy()
        p = group["prediction_log10"].to_numpy()
        mae = mean_absolute_error(y, p)
        rows.append(
            {
                "stage": stage,
                "n": int(len(group)),
                "r2": float(r2_score(y, p)),
                "mae_log10": float(mae),
                "error_x": float(10**mae),
            }
        )
    return sorted(rows, key=lambda x: x["r2"], reverse=True)


def evaluate_ml_csv(
    df: pd.DataFrame,
    target_col: str | None = None,
    max_rows: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    bundle = _load_bundle()
    feature_cols = set(bundle.get("feature_cols") or [])
    direct_feature_cols = sorted(feature_cols.intersection(df.columns))
    feature_coverage = (len(direct_feature_cols) / len(feature_cols)) if feature_cols else 0.0

    target_col = _find_target_col(df, target_col)
    mask = _target_mask(df, target_col)
    work = df.loc[mask].copy()
    if max_rows and max_rows > 0 and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=42).copy()
    if work.empty:
        raise ValueError("No rows with valid positive USD target were found")

    preds: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, row in work.iterrows():
        try:
            pred = predict_ml_from_feature_row(row.to_dict())
            preds.append(
                {
                    "__idx": idx,
                    "raw_prediction_log10": pred.get("raw_prediction_log10"),
                    "prediction_log10": pred["prediction_log10"],
                    "prediction_mid_usd": pred["valuation_mid_usd"],
                    "prediction_min_usd": pred["valuation_min_usd"],
                    "prediction_max_usd": pred["valuation_max_usd"],
                    "calibration_applied": pred.get("calibration_applied"),
                    "calibration_method": pred.get("calibration_method"),
                    "calibration_segment": pred.get("calibration_segment"),
                    "calibration_segment_n": pred.get("calibration_segment_n"),
                    "calibration_correction_log10": pred.get("calibration_correction_log10"),
                    "calibration_multiplier": pred.get("calibration_multiplier"),
                    "residual_prediction_log10": pred.get("residual_prediction_log10"),
                    "prediction_error": "",
                }
            )
        except Exception as e:
            preds.append(
                {
                    "__idx": idx,
                    "prediction_log10": math.nan,
                    "prediction_mid_usd": math.nan,
                    "prediction_min_usd": math.nan,
                    "prediction_max_usd": math.nan,
                    "prediction_error": str(e),
                }
            )
            errors.append(str(e))

    pred_df = pd.DataFrame(preds).set_index("__idx")
    overlapping_prediction_cols = [col for col in pred_df.columns if col in work.columns]
    if overlapping_prediction_cols:
        work = work.drop(columns=overlapping_prediction_cols)
    out = work.join(pred_df, how="left")
    out["target_usd_eval"] = pd.to_numeric(out[target_col], errors="coerce")
    out["target_log10"] = np.log10(out["target_usd_eval"])
    ok = out["prediction_log10"].notna()
    out.loc[ok, "prediction_error"] = (
        out.loc[ok, "prediction_mid_usd"] / out.loc[ok, "target_usd_eval"]
    )
    out.loc[ok, "prediction_abs_error_x"] = np.maximum(
        out.loc[ok, "prediction_error"],
        1.0 / out.loc[ok, "prediction_error"],
    )
    out.loc[ok, "target_in_prediction_range"] = (
        (out.loc[ok, "target_usd_eval"] >= out.loc[ok, "prediction_min_usd"])
        & (out.loc[ok, "target_usd_eval"] <= out.loc[ok, "prediction_max_usd"])
    )
    if not ok.any():
        first_error = errors[0] if errors else "unknown prediction error"
        raise RuntimeError(f"All predictions failed. First error: {first_error}")

    y = out.loc[ok, "target_log10"].to_numpy()
    p = out.loc[ok, "prediction_log10"].to_numpy()
    mae = mean_absolute_error(y, p)
    metrics = {
        "target_col": target_col,
        "model_feature_cols": int(len(feature_cols)),
        "csv_direct_feature_cols": int(len(direct_feature_cols)),
        "csv_direct_feature_coverage": float(feature_coverage),
        "csv_direct_feature_examples": direct_feature_cols[:20],
        "rows_input": int(len(df)),
        "rows_valid_target": int(len(work)),
        "rows_predicted": int(ok.sum()),
        "rows_failed": int((~ok).sum()),
        "r2": float(r2_score(y, p)),
        "mae_log10": float(mae),
        "error_x": float(10**mae),
        "median_abs_pct_error": float(np.median(np.abs(out.loc[ok, "prediction_mid_usd"] / out.loc[ok, "target_usd_eval"] - 1.0))),
        "median_error_x": float(out.loc[ok, "prediction_abs_error_x"].median()),
        "range_hit_rate": float(out.loc[ok, "target_in_prediction_range"].mean()),
        "range_hits": int(out.loc[ok, "target_in_prediction_range"].sum()),
        "stage_metrics": _stage_metrics(out.loc[ok].copy()),
        "first_errors": errors[:5],
    }
    return metrics, out
