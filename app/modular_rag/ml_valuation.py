from __future__ import annotations

import math
import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "startup_valuation_model_v6_historical_priors_no_money.pkl"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "ml_valuation_calibration_v1.json"
RESIDUAL_CALIBRATOR_PATH = Path(__file__).resolve().parents[1] / "models" / "ml_residual_calibrator_v1.pkl"
RANGE_PADDING_LOG10 = 0.05


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9а-яё\s,+#.&/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _prediction_bucket(pred_log10: float, bins: list[float]) -> str:
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if pred_log10 >= lo and pred_log10 < hi:
            left = "-inf" if not math.isfinite(lo) else f"{lo:.1f}"
            right = "inf" if not math.isfinite(hi) else f"{hi:.1f}"
            return f"{left}_{right}"
    return "unknown"


def _normalise_bins(raw_bins: list[Any]) -> list[float]:
    out: list[float] = []
    for value in raw_bins:
        if value is None:
            if not out:
                out.append(-math.inf)
            else:
                out.append(math.inf)
        else:
            out.append(float(value))
    return out


def _split_names(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [p.strip() for p in re.split(r"[,;|/]+", text) if p.strip()]


def _normalize_stage(stage: Any) -> str:
    text = str(stage or "").strip()
    if not text or text.lower() == "nan":
        return "Venture - Series Unknown"
    low = text.lower()
    aliases = {
        "pre seed": "Pre-Seed",
        "pre-seed": "Pre-Seed",
        "seed": "Seed",
        "angel": "Angel",
        "series a": "Series A",
        "series b": "Series B",
        "series c": "Series C",
        "series d": "Series D",
        "series e": "Series E",
        "series f": "Series F",
        "series g": "Series G",
    }
    if low in aliases:
        return aliases[low]
    if low == "funding round":
        return "Venture - Series Unknown"
    return text


def _stage_to_funding_type(stage: str | None) -> str:
    if stage == "idea_only":
        return "Pre-Seed"
    if stage in {"mvp_no_revenue", "early_revenue"}:
        return "Seed"
    if stage == "growth":
        return "Series A"
    return str(stage or "Series A")


def _country_to_model_country(country_hint: str | None) -> str:
    c = (country_hint or "").strip().lower()
    if c == "usa":
        return "United States"
    if c == "russia":
        return "Russia"
    if c == "india":
        return "India"
    if c == "europe":
        return "Unknown"
    return country_hint or "Unknown"


def _industry_to_text(industry_hint: str | None) -> str:
    mapping = {
        "ai": "Artificial Intelligence (AI), Software, SaaS",
        "software": "Software, SaaS",
        "fintech": "Financial Services, FinTech, Payments",
        "healthcare": "Health Care, Medical, Biotechnology",
        "education": "Education, EdTech",
        "ecommerce": "E-Commerce, Marketplace, Retail",
        "logistics": "Logistics, Supply Chain, Delivery",
        "real_estate": "Real Estate, PropTech",
        "climate": "Climate, Sustainability, Energy",
        "travel": "Travel, Hospitality",
    }
    return mapping.get(str(industry_hint or "").lower(), str(industry_hint or "Software"))


@lru_cache(maxsize=1)
def _load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ML valuation model not found: {MODEL_PATH}")
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("joblib is not installed; install app/requirements.txt to enable ML valuation") from e
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_calibration() -> dict[str, Any] | None:
    if not CALIBRATION_PATH.exists():
        return None
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_residual_calibrator() -> dict[str, Any] | None:
    if not RESIDUAL_CALIBRATOR_PATH.exists():
        return None
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("joblib is not installed; install app/requirements.txt to enable residual ML calibration") from e
    return joblib.load(RESIDUAL_CALIBRATOR_PATH)


def _segment_error_stats(artifact: dict[str, Any], pred_log: float, funding_type: Any) -> tuple[str, dict[str, Any]]:
    stage = _normalize_stage(funding_type)
    bins = _normalise_bins(artifact.get("prediction_bins") or [])
    bucket = _prediction_bucket(pred_log, bins) if bins else "unknown"
    stage_bucket_key = f"{stage}||{bucket}"
    by_stage_bucket = artifact.get("by_stage_prediction_bucket_error") or {}
    by_stage = artifact.get("by_stage_error") or {}
    if stage_bucket_key in by_stage_bucket:
        return f"residual_stage+bucket:{stage_bucket_key}", by_stage_bucket[stage_bucket_key]
    if stage in by_stage:
        return f"residual_stage:{stage}", by_stage[stage]
    return "residual_global", artifact.get("global_error") or {}


def _apply_residual_calibrator(pred_log: float, frame: Any, funding_type: Any) -> dict[str, Any] | None:
    artifact = _load_residual_calibrator()
    if not artifact:
        return None

    import pandas as pd

    cols = list(artifact.get("residual_feature_cols") or [])
    cat_cols = set(artifact.get("residual_cat_cols") or [])
    if not cols:
        return None

    residual_frame = frame.copy()
    residual_frame["raw_prediction_log10"] = pred_log
    missing = [col for col in cols if col not in residual_frame.columns]
    if missing:
        for col in missing:
            residual_frame[col] = 0.0
    residual_frame = residual_frame[cols].copy()
    for col in residual_frame.columns:
        if col in cat_cols:
            residual_frame[col] = residual_frame[col].fillna("Unknown").astype(str)
        else:
            residual_frame[col] = pd.to_numeric(residual_frame[col], errors="coerce").fillna(0.0)

    model = artifact.get("model")
    residual = float(model.predict(residual_frame)[0])
    adjusted_log = pred_log + residual
    segment_name, stats = _segment_error_stats(artifact, pred_log, funding_type)
    q10 = _to_float(stats.get("q10_error_log10"))
    q90 = _to_float(stats.get("q90_error_log10"))
    if q10 is None or q90 is None:
        q10, q90 = -0.45, 0.50
    if q10 > q90:
        q10, q90 = q90, q10
    q10 -= RANGE_PADDING_LOG10
    q90 += RANGE_PADDING_LOG10

    return {
        "prediction_log10": adjusted_log,
        "valuation_mid_usd": 10**adjusted_log,
        "valuation_min_usd": 10 ** (adjusted_log + q10),
        "valuation_max_usd": 10 ** (adjusted_log + q90),
        "raw_prediction_log10": pred_log,
        "residual_prediction_log10": residual,
        "calibration_applied": True,
        "calibration_method": artifact.get("version") or "ml_residual_calibrator",
        "calibration_segment": segment_name,
        "calibration_segment_n": stats.get("n"),
        "calibration_correction_log10": residual,
        "calibration_multiplier": 10**residual,
        "calibration_q10_residual_log10": q10,
        "calibration_q90_residual_log10": q90,
    }


def _row_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _money_metric(row: dict[str, Any], key: str) -> float | None:
    value = _to_float(_row_value(row, key))
    return value if value and value > 0 else None


def _count_metric(row: dict[str, Any], key: str) -> int:
    value = _to_float(_row_value(row, key))
    return int(value) if value and value > 0 else 0


def _historical_prior_block(row: dict[str, Any], prefix: str, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "round_count": _count_metric(row, f"{prefix}_round_count"),
        "company_count": _count_metric(row, f"{prefix}_company_count"),
        "median_raised_usd": _money_metric(row, f"{prefix}_median_raised_usd"),
        "mean_raised_usd": _money_metric(row, f"{prefix}_mean_raised_usd"),
        "p90_raised_usd": _money_metric(row, f"{prefix}_p90_raised_usd"),
    }


def _build_ml_explainability(frame: Any, prediction: dict[str, Any]) -> dict[str, Any]:
    row = frame.iloc[0].to_dict()
    prior_blocks = [
        _historical_prior_block(row, "oldcb_category_prior", "category"),
        _historical_prior_block(row, "oldcb_stage_prior", "stage"),
        _historical_prior_block(row, "oldcb_category_stage_prior", "category_stage"),
    ]
    prior_blocks = [
        block
        for block in prior_blocks
        if block["round_count"] or block["company_count"] or block["median_raised_usd"] or block["p90_raised_usd"]
    ]

    investor_strength = {
        "current_investors_matched_count": _count_metric(row, "oldcb_current_investors_matched_count"),
        "prior_deal_count_sum": _count_metric(row, "oldcb_current_investors_prior_deal_count_sum"),
        "prior_deal_count_max": _count_metric(row, "oldcb_current_investors_prior_deal_count_max"),
        "prior_unique_company_count_sum": _count_metric(row, "oldcb_current_investors_prior_unique_company_count_sum"),
        "prior_median_round_raised_mean": _money_metric(row, "oldcb_current_investors_prior_median_round_raised_mean"),
        "prior_p90_round_raised_mean": _money_metric(row, "oldcb_current_investors_prior_p90_round_raised_mean"),
    }
    top_investor_flags = [
        key
        for key in [
            "has_a16z",
            "has_sequoia",
            "has_accel",
            "has_yc",
            "has_khosla",
            "has_index",
            "has_lightspeed",
            "has_general_catalyst",
            "has_bessemer",
            "has_google_ventures",
            "has_tiger_global",
            "has_softbank",
            "has_insight",
            "has_founders_fund",
        ]
        if _count_metric(row, key)
    ]
    keyword_flags = [
        key.replace("kw_", "")
        for key in row
        if key.startswith("kw_") and _count_metric(row, key)
    ][:12]
    text_signals = [
        key.replace("has_", "")
        for key in [
            "has_platform",
            "has_automation",
            "has_infrastructure",
            "has_api",
            "has_enterprise",
            "has_open_source",
            "has_patent",
            "has_clinical",
            "has_revenue_word",
            "has_customer_word",
            "has_partner_word",
        ]
        if _count_metric(row, key)
    ]

    confidence_reasons: list[str] = []
    risk_reasons: list[str] = []
    if prior_blocks:
        confidence_reasons.append("historical_priors_available")
    else:
        risk_reasons.append("historical_priors_missing_or_zero")
    if investor_strength["current_investors_matched_count"] or investor_strength["prior_deal_count_sum"]:
        confidence_reasons.append("investor_history_available")
    else:
        risk_reasons.append("investor_history_missing")
    if _count_metric(row, "description_word_count") >= 20:
        confidence_reasons.append("company_description_available")
    else:
        risk_reasons.append("company_description_sparse")
    if top_investor_flags:
        confidence_reasons.append("top_investor_signal")
    if _count_metric(row, "num_funding_rounds_num"):
        confidence_reasons.append("funding_round_count_available")

    confidence_score = len(confidence_reasons) - 0.5 * len(risk_reasons)
    explanation_confidence = "high" if confidence_score >= 4 else "medium" if confidence_score >= 2 else "low"

    raw_log = _to_float(prediction.get("raw_prediction_log10")) or _to_float(prediction.get("prediction_log10"))
    calibrated_log = _to_float(prediction.get("prediction_log10"))
    return {
        "raw_ml": {
            "prediction_log10": raw_log,
            "valuation_mid_usd": 10**raw_log if raw_log is not None else None,
        },
        "calibration": {
            "method": prediction.get("calibration_method"),
            "segment": prediction.get("calibration_segment"),
            "segment_n": prediction.get("calibration_segment_n"),
            "correction_log10": prediction.get("calibration_correction_log10"),
            "multiplier": prediction.get("calibration_multiplier"),
            "calibrated_prediction_log10": calibrated_log,
            "calibrated_mid_usd": prediction.get("valuation_mid_usd"),
            "range_min_usd": prediction.get("valuation_min_usd"),
            "range_max_usd": prediction.get("valuation_max_usd"),
        },
        "company_context": {
            "stage": row.get("Funding Type"),
            "stage_group": row.get("stage_group_derived"),
            "country": row.get("Country"),
            "region": row.get("Region"),
            "city": row.get("City"),
            "primary_industry": row.get("primary_industry_derived"),
            "round_year": _to_float(row.get("round_year_derived")),
            "round_month": _to_float(row.get("round_month_derived")),
            "company_age_at_round": _to_float(row.get("oldcb_company_age_at_round")),
            "funding_rounds_count": _to_float(row.get("num_funding_rounds_num")) or _to_float(row.get("oldcb_funding_rounds_count")),
            "description_word_count": _count_metric(row, "description_word_count"),
        },
        "historical_priors": prior_blocks,
        "acquisition_priors": {
            "category_acquisition_count": _count_metric(row, "oldcb_category_prior_acquisition_count"),
            "category_median_acquisition_price_usd": _money_metric(row, "oldcb_category_prior_median_acquisition_price_usd"),
        },
        "investor_strength": investor_strength,
        "top_investor_flags": top_investor_flags,
        "keyword_flags": keyword_flags,
        "text_signals": text_signals,
        "explanation_confidence": explanation_confidence,
        "confidence_reasons": confidence_reasons,
        "risk_reasons": risk_reasons,
    }


def _apply_calibration(pred_log: float, funding_type: Any, fallback_mae: float) -> dict[str, Any]:
    calibration = _load_calibration()
    if not calibration:
        return {
            "prediction_log10": pred_log,
            "valuation_mid_usd": 10**pred_log,
            "valuation_min_usd": 10 ** (pred_log - fallback_mae),
            "valuation_max_usd": 10 ** (pred_log + fallback_mae),
            "raw_prediction_log10": pred_log,
            "calibration_applied": False,
            "calibration_method": "none",
            "calibration_segment": "none",
            "calibration_correction_log10": 0.0,
        }

    stage = _normalize_stage(funding_type)
    bins = calibration.get("prediction_bins") or []
    bucket = _prediction_bucket(pred_log, bins) if bins else "unknown"
    stage_bucket_key = f"{stage}||{bucket}"

    segment = None
    segment_name = ""
    by_stage_bucket = calibration.get("by_stage_prediction_bucket") or {}
    by_stage = calibration.get("by_stage") or {}
    if stage_bucket_key in by_stage_bucket:
        segment = by_stage_bucket[stage_bucket_key]
        segment_name = f"stage+bucket:{stage_bucket_key}"
    elif stage in by_stage:
        segment = by_stage[stage]
        segment_name = f"stage:{stage}"
    else:
        segment = calibration.get("global") or {}
        segment_name = "global"

    correction = _to_float(segment.get("correction_log10")) or 0.0
    q10 = _to_float(segment.get("q10_residual_log10"))
    q90 = _to_float(segment.get("q90_residual_log10"))
    if q10 is None or q90 is None:
        q10 = correction - fallback_mae
        q90 = correction + fallback_mae
    if q10 > q90:
        q10, q90 = q90, q10
    q10 -= RANGE_PADDING_LOG10
    q90 += RANGE_PADDING_LOG10

    adjusted_log = pred_log + correction
    return {
        "prediction_log10": adjusted_log,
        "valuation_mid_usd": 10**adjusted_log,
        "valuation_min_usd": 10 ** (pred_log + q10),
        "valuation_max_usd": 10 ** (pred_log + q90),
        "raw_prediction_log10": pred_log,
        "calibration_applied": True,
        "calibration_method": calibration.get("version") or "median_residual_json",
        "calibration_segment": segment_name,
        "calibration_segment_n": segment.get("n"),
        "calibration_correction_log10": correction,
        "calibration_multiplier": 10**correction,
        "calibration_q10_residual_log10": q10,
        "calibration_q90_residual_log10": q90,
    }


def ml_model_status() -> dict[str, Any]:
    try:
        bundle = _load_bundle()
        calibration = _load_calibration()
        residual_calibrator = _load_residual_calibrator()
        return {
            "available": True,
            "model_path": str(MODEL_PATH),
            "calibration_path": str(CALIBRATION_PATH),
            "calibration_available": calibration is not None,
            "calibration_version": (calibration or {}).get("version"),
            "residual_calibrator_path": str(RESIDUAL_CALIBRATOR_PATH),
            "residual_calibrator_available": residual_calibrator is not None,
            "residual_calibrator_version": (residual_calibrator or {}).get("version"),
            "residual_calibrator_holdout_metrics": (residual_calibrator or {}).get("holdout_metrics"),
            "version": bundle.get("version"),
            "best_model_name": bundle.get("best_model_name"),
            "r2": bundle.get("r2"),
            "mae_log10": bundle.get("mae_log10"),
            "error_x": bundle.get("error_x"),
        }
    except Exception as e:
        return {"available": False, "model_path": str(MODEL_PATH), "error": str(e)}


def _build_input_frame(bundle: dict[str, Any], project_text: str, payload: dict[str, Any]) -> Any:
    import pandas as pd

    feature_cols: list[str] = list(bundle.get("feature_cols") or [])
    cat_cols: list[str] = list(bundle.get("cat_cols") or [])
    num_cols: list[str] = list(bundle.get("num_cols") or [])
    text_col = str(bundle.get("text_col") or "model_text_all_derived")

    country = _country_to_model_country(payload.get("country_hint"))
    industry_text = _industry_to_text(payload.get("industry_hint"))
    stage = _stage_to_funding_type(payload.get("stage"))
    year = payload.get("damodaran_year_used") or 2026
    year_f = _to_float(year) or 2026.0

    text = _clean_text(
        " ".join(
            [
                project_text or "",
                industry_text,
                country,
                str(payload.get("industry_hint") or ""),
                str(payload.get("country_hint") or ""),
            ]
        )
    )

    row: dict[str, Any] = {col: 0.0 for col in feature_cols}
    defaults: dict[str, Any] = {
        "Funding Type": stage,
        "Funding Status": "Unknown",
        "Country": country,
        "Region": "Unknown",
        "State": "Unknown",
        "City": "Unknown",
        "stage_group_derived": "growth" if stage in {"Series A", "Series B", "Series C"} else "early",
        "primary_industry_derived": industry_text.split(",")[0],
        "country_bucket_derived": country,
        "round_year_derived": year_f,
        "round_month_derived": 6.0,
        "round_quarter_derived": 2.0,
        "years_ago_2026": 2026.0 - year_f,
        text_col: text,
    }
    for key, value in defaults.items():
        if key in row:
            row[key] = value

    # Lightweight derived flags that match the v6 notebook names when present.
    keyword_flags = {
        "kw_ai": ["ai", "artificial intelligence", "machine learning", "llm", "нейросет"],
        "kw_saas": ["saas", "software as a service"],
        "kw_b2b": ["b2b", "enterprise", "business"],
        "kw_marketplace": ["marketplace"],
        "kw_fintech": ["fintech", "payment", "banking", "financial"],
        "kw_healthcare": ["health", "medical", "clinical", "patient"],
        "kw_biotech": ["biotech", "therapeutics", "genomics"],
        "kw_edtech": ["edtech", "education", "learning"],
        "kw_ecommerce": ["ecommerce", "e-commerce", "retail"],
        "kw_climate": ["climate", "carbon", "sustainability", "energy"],
        "kw_logistics": ["logistics", "delivery", "supply chain"],
        "kw_cybersecurity": ["cybersecurity", "security", "fraud"],
        "kw_devtools": ["developer", "api", "devops"],
        "kw_data": ["data", "analytics"],
    }
    for col, needles in keyword_flags.items():
        if col in row:
            row[col] = 1.0 if any(n in text for n in needles) else 0.0

    simple_flags = {
        "has_platform": "platform",
        "has_automation": "automation",
        "has_api": "api",
        "has_enterprise": "enterprise",
        "has_revenue_word": "revenue",
        "has_customer_word": "customer",
        "has_partner_word": "partner",
    }
    for col, needle in simple_flags.items():
        if col in row:
            row[col] = 1.0 if needle in text else 0.0

    if "description_word_count" in row:
        row["description_word_count"] = len(re.findall(r"[a-zа-яё0-9]+", text))
    if "description_char_len" in row:
        row["description_char_len"] = len(text)
    if "description_has_text" in row:
        row["description_has_text"] = 1.0 if len(text) > 20 else 0.0
    if "industry_count" in row:
        row["industry_count"] = max(1, len([p for p in industry_text.split(",") if p.strip()]))

    frame = pd.DataFrame([row])
    for col in cat_cols + [text_col]:
        if col in frame.columns:
            frame[col] = frame[col].fillna("Unknown").astype(str)
    for col in num_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame[feature_cols]


def predict_ml_market_benchmark(project_text: str, payload: dict[str, Any]) -> dict[str, Any]:
    bundle = _load_bundle()
    frame = _build_input_frame(bundle, project_text, payload)
    model_name = str(bundle.get("best_model_name") or "")
    if model_name.startswith("CatBoost"):
        model = bundle.get("catboost_model") or bundle.get("best_model")
        cols = list(bundle.get("catboost_cols") or bundle.get("feature_cols") or [])
        pred_log = float(model.predict(frame[cols])[0])
    else:
        model = bundle.get("best_model")
        preprocess = bundle.get("preprocess")
        if preprocess is not None:
            pred_log = float(model.predict(preprocess.transform(frame))[0])
        else:
            pred_log = float(model.predict(frame)[0])

    mae = _to_float(bundle.get("mae_log10")) or 0.423
    error_x = _to_float(bundle.get("error_x")) or (10**mae)
    stage = frame["Funding Type"].iloc[0] if "Funding Type" in frame.columns else payload.get("stage")
    calibrated = _apply_residual_calibrator(pred_log, frame, stage) or _apply_calibration(pred_log, stage, mae)
    explainability = _build_ml_explainability(frame, calibrated)
    return {
        "model_version": bundle.get("version"),
        "model_name": model_name,
        **calibrated,
        "explainability": explainability,
        "mae_log10": mae,
        "error_x": error_x,
        "r2": bundle.get("r2"),
    }


def predict_ml_from_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    """Predict directly from a CSV-like feature row.

    This path is for offline/batch testing against Crunchbase-shaped CSV files.
    It maps any matching columns from the row into the model feature frame and
    leaves missing features at conservative defaults.
    """
    import pandas as pd

    bundle = _load_bundle()
    feature_cols: list[str] = list(bundle.get("feature_cols") or [])
    cat_cols: list[str] = list(bundle.get("cat_cols") or [])
    num_cols: list[str] = list(bundle.get("num_cols") or [])
    text_col = str(bundle.get("text_col") or "model_text_all_derived")

    values: dict[str, Any] = {col: 0.0 for col in feature_cols}
    for col in feature_cols:
        if col in row and row[col] is not None:
            values[col] = row[col]

    # Friendly fallbacks for non-enriched CSVs.
    aliases = {
        "Funding Type": ["Funding Type", "stage"],
        "Funding Status": ["Funding Status", "funding_status"],
        "Country": ["Country", "country"],
        "Region": ["Region", "region"],
        "State": ["State", "state"],
        "City": ["City", "city"],
        "primary_industry_derived": ["primary_industry_derived", "Organization Industries", "industry", "industries"],
        "country_bucket_derived": ["country_bucket_derived", "Country", "country"],
        "round_year_derived": ["round_year_derived", "funded_year", "round_year"],
        "round_month_derived": ["round_month_derived", "round_month"],
    }
    for target, sources in aliases.items():
        if target in values:
            for src in sources:
                if src in row and row[src] not in {None, ""}:
                    values[target] = row[src]
                    break

    if "Funding Type" in values:
        values["Funding Type"] = _normalize_stage(values.get("Funding Type"))

    if ("round_year_derived" in values or "round_month_derived" in values) and row.get("Announced Date") not in {None, ""}:
        dt = pd.to_datetime(row.get("Announced Date"), errors="coerce")
        if pd.notna(dt):
            if "round_year_derived" in values and not _to_float(values.get("round_year_derived")):
                values["round_year_derived"] = float(dt.year)
            if "round_month_derived" in values and not _to_float(values.get("round_month_derived")):
                values["round_month_derived"] = float(dt.month)

    if text_col in values:
        text_parts = [
            row.get(text_col),
            row.get("Organization Description"),
            row.get("Organization Industries"),
            row.get("Investor Names"),
            row.get("Lead Investors"),
            row.get("Funding Status"),
            row.get("Country"),
            row.get("Region"),
            row.get("City"),
        ]
        values[text_col] = _clean_text(" ".join(str(x or "") for x in text_parts))

    year = _to_float(values.get("round_year_derived"))
    if "years_ago_2026" in values and year:
        values["years_ago_2026"] = 2026.0 - year
    month = _to_float(values.get("round_month_derived"))
    if "round_quarter_derived" in values and month:
        values["round_quarter_derived"] = int((month - 1) / 3) + 1

    text = str(values.get(text_col) or "")
    investor_names = _split_names(row.get("Investor Names"))
    lead_investor_names = _split_names(row.get("Lead Investors"))
    industry_names = _split_names(row.get("Organization Industries") or row.get("industries") or row.get("industry"))

    if "investor_name_count_from_text" in values:
        values["investor_name_count_from_text"] = len(investor_names)
    if "lead_investor_name_count_from_text" in values:
        values["lead_investor_name_count_from_text"] = len(lead_investor_names)
    if "has_named_investor_text" in values:
        values["has_named_investor_text"] = 1.0 if investor_names else 0.0
    if "has_lead_investor_text" in values:
        values["has_lead_investor_text"] = 1.0 if lead_investor_names else 0.0
    if "num_investors_num" in values and investor_names and not _to_float(values.get("num_investors_num")):
        values["num_investors_num"] = len(investor_names)
    if "log_num_investors_num" in values and investor_names and not _to_float(values.get("log_num_investors_num")):
        values["log_num_investors_num"] = math.log1p(len(investor_names))
    if "has_num_investors_num" in values:
        values["has_num_investors_num"] = 1.0 if investor_names else 0.0
    if "industry_count" in values and industry_names:
        values["industry_count"] = len(industry_names)
    if "description_word_count" in values:
        values["description_word_count"] = len(re.findall(r"[a-zа-яё0-9]+", text))
    if "description_char_len" in values:
        values["description_char_len"] = len(text)

    stage = str(values.get("Funding Type") or "")
    if "stage_group_derived" in values and values.get("stage_group_derived") in {0, 0.0, None, ""}:
        if stage in {"Pre-Seed", "Seed", "Angel", "Convertible Note", "Equity Crowdfunding"}:
            values["stage_group_derived"] = "early"
        elif stage in {"Series A", "Series B", "Series C"}:
            values["stage_group_derived"] = "growth"
        else:
            values["stage_group_derived"] = "late" if stage in {"Series D", "Series E", "Series F", "Series G"} else "unknown"
    if "is_stage_early" in values:
        values["is_stage_early"] = 1.0 if stage in {"Pre-Seed", "Seed", "Angel", "Convertible Note", "Equity Crowdfunding"} else 0.0
    if "is_stage_growth" in values:
        values["is_stage_growth"] = 1.0 if stage in {"Series A", "Series B", "Series C"} else 0.0
    if "is_stage_late" in values:
        values["is_stage_late"] = 1.0 if stage in {"Series D", "Series E", "Series F", "Series G"} else 0.0

    country = str(values.get("Country") or "").lower()
    region = str(values.get("Region") or "").lower()
    country_flags = {
        "is_us": country in {"united states", "usa", "us"},
        "is_india": country == "india",
        "is_china": country == "china",
        "is_uk": country in {"united kingdom", "uk"},
        "is_europe_region": "europe" in region,
        "is_north_america_region": "north america" in region,
        "is_asia_region": "asia" in region,
    }
    for col, flag in country_flags.items():
        if col in values:
            values[col] = 1.0 if flag else 0.0

    keyword_flags = {
        "kw_ai": ["ai", "artificial intelligence", "machine learning", "llm", "нейросет"],
        "kw_saas": ["saas", "software as a service"],
        "kw_b2b": ["b2b", "enterprise", "business"],
        "kw_marketplace": ["marketplace"],
        "kw_fintech": ["fintech", "payment", "banking", "financial"],
        "kw_healthcare": ["health", "medical", "clinical", "patient"],
        "kw_biotech": ["biotech", "therapeutics", "genomics"],
        "kw_edtech": ["edtech", "education", "learning"],
        "kw_ecommerce": ["ecommerce", "e-commerce", "retail"],
        "kw_climate": ["climate", "carbon", "sustainability", "energy"],
        "kw_logistics": ["logistics", "delivery", "supply chain"],
        "kw_cybersecurity": ["cybersecurity", "security", "fraud"],
        "kw_devtools": ["developer", "api", "devops"],
        "kw_data": ["data", "analytics"],
    }
    for col, needles in keyword_flags.items():
        if col in values and (values.get(col) in {0, 0.0, None, ""}):
            values[col] = 1.0 if any(n in text for n in needles) else 0.0

    top_investors = {
        "has_a16z": ["andreessen horowitz", "a16z"],
        "has_sequoia": ["sequoia"],
        "has_accel": ["accel"],
        "has_yc": ["y combinator"],
        "has_khosla": ["khosla"],
        "has_index": ["index ventures"],
        "has_lightspeed": ["lightspeed"],
        "has_general_catalyst": ["general catalyst"],
        "has_bessemer": ["bessemer"],
        "has_kleiner": ["kleiner perkins"],
        "has_google_ventures": ["gv", "google ventures"],
        "has_tiger_global": ["tiger global"],
    }
    investor_text = _clean_text(" ".join([str(row.get("Investor Names") or ""), str(row.get("Lead Investors") or "")]))
    for col, needles in top_investors.items():
        if col in values:
            values[col] = 1.0 if any(n in investor_text for n in needles) else 0.0
    if "has_any_top_investor" in values:
        values["has_any_top_investor"] = 1.0 if any(values.get(col) == 1.0 for col in top_investors) else 0.0

    frame = pd.DataFrame([values])
    for col in cat_cols + [text_col]:
        if col in frame.columns:
            frame[col] = frame[col].fillna("Unknown").astype(str)
    for col in num_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = frame[feature_cols]

    model_name = str(bundle.get("best_model_name") or "")
    if model_name.startswith("CatBoost"):
        model = bundle.get("catboost_model") or bundle.get("best_model")
        cols = list(bundle.get("catboost_cols") or feature_cols)
        pred_log = float(model.predict(frame[cols])[0])
    else:
        model = bundle.get("best_model")
        preprocess = bundle.get("preprocess")
        pred_log = float(model.predict(preprocess.transform(frame))[0]) if preprocess is not None else float(model.predict(frame)[0])

    mae = _to_float(bundle.get("mae_log10")) or 0.423
    calibrated = _apply_residual_calibrator(pred_log, frame, values.get("Funding Type")) or _apply_calibration(pred_log, values.get("Funding Type"), mae)
    explainability = _build_ml_explainability(frame, calibrated)
    return {
        **calibrated,
        "explainability": explainability,
        "model_name": model_name,
        "model_version": bundle.get("version"),
    }
