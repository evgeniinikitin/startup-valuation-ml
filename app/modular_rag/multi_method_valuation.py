from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modular_rag.ml_enrichment import build_ml_enrichment_row
from modular_rag.ml_valuation import predict_ml_market_benchmark
from modular_rag.ml_valuation import predict_ml_from_feature_row


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


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


def _load_benchmarks() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "vc_benchmarks.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _stage_family(stage: str) -> str:
    if stage == "idea_only":
        return "idea"
    if stage == "mvp_no_revenue":
        return "development"
    if stage == "early_revenue":
        return "startup"
    if stage == "growth":
        return "growth"
    return "startup"


def _benchmark_stage(stage: str) -> str:
    if stage == "idea_only":
        return "Pre-Seed"
    if stage in {"mvp_no_revenue", "early_revenue"}:
        return "Seed"
    return "Series A"


def _regional_discount(country_hint: str, benchmarks: dict[str, Any]) -> float:
    reg = benchmarks.get("regional_discount") or {}
    country = _norm(country_hint)
    if country == "russia":
        return float(reg.get("russia", 0.30))
    if country == "europe":
        return float(reg.get("europe", 0.60))
    if country == "india":
        return float(reg.get("india", 0.40))
    if country in {"latam", "latin america"}:
        return float(reg.get("latam", 0.35))
    return float(reg.get("us", 1.0))


def _base_market_valuation(stage: str, country_hint: str, industry_hint: str, benchmarks: dict[str, Any]) -> float | None:
    us = benchmarks.get("us_medians") or {}
    row = us.get(_benchmark_stage(stage)) or {}
    base = _to_float(row.get("pre_money_valuation_usd"))
    if not base:
        return None
    ai_multiplier = _to_float(benchmarks.get("ai_premium_multiplier")) or 1.0
    industry_multiplier = ai_multiplier if industry_hint == "ai" else 1.0
    return base * _regional_discount(country_hint, benchmarks) * industry_multiplier


def _keyword_score(text: str, positive: list[str], negative: list[str] | None = None, default: float = 0.55) -> tuple[float, list[str]]:
    t = _norm(text)
    hits = [k for k in positive if k in t]
    bad = [k for k in (negative or []) if k in t]
    score = default + min(0.35, 0.09 * len(hits)) - min(0.25, 0.10 * len(bad))
    return max(0.15, min(1.10, score)), hits[:5] + [f"negative:{k}" for k in bad[:3]]


def _range_around(mid: float, spread: float) -> tuple[float, float]:
    return max(0.0, mid / spread), mid * spread


@dataclass
class MethodEstimate:
    name: str
    value_mid_usd: float
    value_min_usd: float
    value_max_usd: float
    weight: float
    confidence: str
    rationale: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        row = {
            "name": self.name,
            "value_min_usd": round(self.value_min_usd, 2),
            "value_mid_usd": round(self.value_mid_usd, 2),
            "value_max_usd": round(self.value_max_usd, 2),
            "weight": round(self.weight, 4),
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
        if self.details:
            row["details"] = self.details
        return row


def _method_weights(stage: str) -> dict[str, float]:
    family = _stage_family(stage)
    if family == "idea":
        return {"ml": 0.25, "scorecard": 0.25, "checklist": 0.25, "vc": 0.05, "revenue_multiple": 0.00, "existing_script": 0.20}
    if family == "development":
        return {"ml": 0.25, "scorecard": 0.20, "checklist": 0.20, "vc": 0.10, "revenue_multiple": 0.05, "existing_script": 0.20}
    if family == "startup":
        return {"ml": 0.30, "scorecard": 0.10, "checklist": 0.05, "vc": 0.15, "revenue_multiple": 0.20, "existing_script": 0.20}
    return {"ml": 0.25, "scorecard": 0.05, "checklist": 0.00, "vc": 0.15, "revenue_multiple": 0.35, "existing_script": 0.20}


def _ml_adjustment(project_text: str) -> tuple[float, float, list[str]]:
    """Return (valuation multiplier, range factor, evidence notes)."""
    t = _norm(project_text)
    factors: list[tuple[str, float]] = []
    evidence = 0.0

    rules = [
        ("strong_team", 1.10, ["ex-", "serial founder", "advisor", "экс-", "founder"], 0.12),
        ("traction", 1.16, ["paying", "revenue", "arr", "mrr", "customer", "contract", "клиент"], 0.18),
        ("product", 1.08, ["mvp", "launched", "production", "api", "prototype", "запущ"], 0.12),
        ("market", 1.07, ["tam", "sam", "market", "billion", "млрд", "global"], 0.10),
        ("partnerships", 1.06, ["partner", "pilot", "partnership", "пилот", "партнер"], 0.10),
        ("defensibility", 1.06, ["patent", "proprietary", "moat", "ip", "патент"], 0.08),
    ]
    for name, mult, needles, ev in rules:
        if any(n in t for n in needles):
            factors.append((name, mult))
            evidence += ev

    penalties = [
        ("forecast_only", 0.88, ["projected", "forecast", "план", "прогноз"]),
        ("no_product_signal", 0.92, []),
        ("competition_signal", 0.92, ["crowded", "high competition", "конкур"]),
    ]
    for name, mult, needles in penalties:
        if name == "no_product_signal":
            if not any(n in t for n in ["mvp", "prototype", "launched", "production", "demo", "прототип", "запущ"]):
                factors.append((name, mult))
            continue
        if any(n in t for n in needles):
            factors.append((name, mult))

    multiplier = 1.0
    for _, mult in factors:
        multiplier *= mult
    multiplier = max(0.55, min(1.75, multiplier))

    # ML model has about ±2.65x error. More evidence narrows the displayed range;
    # weak evidence keeps it broad.
    range_factor = max(1.65, min(3.00, 2.65 - evidence))
    return multiplier, range_factor, [f"{name}={mult:.2f}" for name, mult in factors]


def _ml_market_method(project_text: str, payload: dict[str, Any], weight: float, llm: Any | None = None) -> MethodEstimate | None:
    if weight <= 0:
        return None
    enrichment_info: dict[str, Any] = {}
    if llm is not None:
        enriched_row, enrichment_info = build_ml_enrichment_row(project_text, payload, llm=llm)
        prediction = predict_ml_from_feature_row(enriched_row) if enriched_row else predict_ml_market_benchmark(project_text, payload)
    else:
        prediction = predict_ml_market_benchmark(project_text, payload)
    mid = _to_float(prediction.get("valuation_mid_usd"))
    if not mid or mid <= 0:
        return None
    multiplier, range_factor, notes = _ml_adjustment(project_text)
    adjusted_mid = mid * multiplier
    lo, hi = adjusted_mid / range_factor, adjusted_mid * range_factor
    r2 = _to_float(prediction.get("r2"))
    error_x = _to_float(prediction.get("error_x"))
    r2_text = f"{r2:.3f}" if r2 is not None else "n/a"
    error_text = f"{error_x:.2f}" if error_x is not None else "n/a"
    return MethodEstimate(
        name="ml_market_benchmark_adjusted",
        value_mid_usd=adjusted_mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence="medium",
        rationale=(
            f"{prediction.get('model_name')} prior mid=${mid:,.0f}, "
            f"R²={r2_text}, raw_error=±{error_text}x, "
            f"adjustment={multiplier:.2f}, displayed_range=±{range_factor:.2f}x, factors={notes}"
        ),
        details={
            "ml_prediction": {
                "model_name": prediction.get("model_name"),
                "model_version": prediction.get("model_version"),
                "raw_prediction_log10": prediction.get("raw_prediction_log10"),
                "prediction_log10": prediction.get("prediction_log10"),
                "calibration_method": prediction.get("calibration_method"),
                "calibration_segment": prediction.get("calibration_segment"),
                "calibration_multiplier": prediction.get("calibration_multiplier"),
                "residual_prediction_log10": prediction.get("residual_prediction_log10"),
                "ml_range_min_usd": prediction.get("valuation_min_usd"),
                "ml_range_mid_usd": prediction.get("valuation_mid_usd"),
                "ml_range_max_usd": prediction.get("valuation_max_usd"),
            },
            "project_text_adjustment": {
                "multiplier": multiplier,
                "range_factor": range_factor,
                "factors": notes,
            },
            "explainability": prediction.get("explainability") or {},
            "input_enrichment": enrichment_info,
        },
    )


def _scorecard_method(project_text: str, payload: dict[str, Any], benchmarks: dict[str, Any], weight: float) -> MethodEstimate | None:
    base = _base_market_valuation(
        str(payload.get("stage") or ""),
        str(payload.get("country_hint") or ""),
        str(payload.get("industry_hint") or ""),
        benchmarks,
    )
    if not base or weight <= 0:
        return None
    team, team_hits = _keyword_score(project_text, ["ex-", "serial founder", "founder", "team", "команд", "экс-", "advisor"], default=0.55)
    market, market_hits = _keyword_score(project_text, ["tam", "sam", "som", "market", "рынок", "billion", "млрд", "global"], default=0.60)
    product, product_hits = _keyword_score(project_text, ["mvp", "launched", "production", "patent", "api", "prototype", "запущ", "патент"], default=0.55)
    partners, partner_hits = _keyword_score(project_text, ["partner", "partnership", "pilot", "contract", "customer", "клиент", "пилот"], default=0.50)
    competition, comp_hits = _keyword_score(project_text, ["moat", "defensible", "proprietary", "unique", "ip"], ["competition", "crowded", "конкур"], default=0.55)
    funding, funding_hits = _keyword_score(project_text, ["runway", "milestone", "use of funds", "unit economics"], default=0.50)
    score = (
        0.30 * team
        + 0.25 * market
        + 0.15 * product
        + 0.10 * partners
        + 0.10 * competition
        + 0.10 * funding
    )
    mid = base * max(0.45, min(1.55, score))
    lo, hi = _range_around(mid, 1.75)
    hits = team_hits + market_hits + product_hits + partner_hits + comp_hits + funding_hits
    return MethodEstimate(
        name="scorecard",
        value_mid_usd=mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence="medium",
        rationale=f"Market benchmark adjusted by qualitative score={score:.2f}; evidence={hits[:8]}",
    )


def _checklist_method(project_text: str, payload: dict[str, Any], benchmarks: dict[str, Any], weight: float) -> MethodEstimate | None:
    if weight <= 0:
        return None
    max_val = _base_market_valuation(
        str(payload.get("stage") or ""),
        str(payload.get("country_hint") or ""),
        str(payload.get("industry_hint") or ""),
        benchmarks,
    )
    if not max_val:
        return None
    # Berkus-like maximum is deliberately capped around early-stage market benchmark.
    max_val *= 1.25
    idea, idea_hits = _keyword_score(project_text, ["problem", "pain", "solution", "идея", "проблем"], default=0.55)
    prototype, proto_hits = _keyword_score(project_text, ["mvp", "prototype", "demo", "launched", "прототип", "запущ"], default=0.35)
    team, team_hits = _keyword_score(project_text, ["founder", "team", "advisor", "экс-", "команд"], default=0.45)
    strategic, strat_hits = _keyword_score(project_text, ["partner", "pilot", "contract", "distribution", "партнер", "пилот"], default=0.35)
    rollout, rollout_hits = _keyword_score(project_text, ["gtm", "sales", "go-to-market", "pipeline", "продаж"], default=0.35)
    score = 0.20 * idea + 0.20 * prototype + 0.30 * team + 0.15 * strategic + 0.15 * rollout
    mid = max_val * max(0.20, min(1.0, score))
    lo, hi = _range_around(mid, 1.90)
    hits = idea_hits + proto_hits + team_hits + strat_hits + rollout_hits
    return MethodEstimate(
        name="checklist",
        value_mid_usd=mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence="medium",
        rationale=f"Checklist/Berkus-style intangible score={score:.2f}; evidence={hits[:8]}",
    )


def _vc_method(payload: dict[str, Any], weight: float) -> MethodEstimate | None:
    arr = _to_float(payload.get("arr_usd"))
    if not arr or arr <= 0 or weight <= 0:
        return None
    stage = str(payload.get("stage") or "")
    family = _stage_family(stage)
    growth_factor = {"idea": 12.0, "development": 8.0, "startup": 5.0, "growth": 3.0}.get(family, 5.0)
    discount_rate = {"idea": 0.9408, "development": 0.7632, "startup": 0.5872, "growth": 0.3489}.get(family, 0.5872)
    horizon = 5
    multiple_candidates = [
        _to_float(payload.get("ev_sales_used")),
        _to_float(payload.get("country_ev_sales_used")),
        3.0,
    ]
    revenue_multiple = next((m for m in multiple_candidates if m and m > 0), 3.0)
    exit_value = arr * growth_factor * revenue_multiple
    present_value = exit_value / ((1.0 + discount_rate) ** horizon)
    ask = _to_float(payload.get("investment_ask_usd")) or 0.0
    mid = max(0.0, present_value - ask)
    if mid <= 0:
        return None
    lo, hi = _range_around(mid, 2.25)
    return MethodEstimate(
        name="venture_capital",
        value_mid_usd=mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence="low",
        rationale=(
            f"ARR * growth_factor * revenue_multiple discounted over {horizon}y: "
            f"{arr:,.0f} * {growth_factor:.1f} * {revenue_multiple:.2f}, discount={discount_rate:.1%}, ask={ask:,.0f}"
        ),
    )


def _revenue_multiple_method(payload: dict[str, Any], weight: float) -> MethodEstimate | None:
    arr = _to_float(payload.get("arr_usd"))
    if not arr or arr <= 0 or weight <= 0:
        return None
    stage = str(payload.get("stage") or "")
    multiple_candidates = [
        _to_float(payload.get("ev_sales_used")),
        _to_float(payload.get("country_ev_sales_used")),
        2.5,
    ]
    multiple = next((m for m in multiple_candidates if m and m > 0), 2.5)
    stage_discount = {"idea_only": 0.20, "mvp_no_revenue": 0.35, "early_revenue": 0.60, "growth": 0.90}.get(stage, 0.60)
    mid = arr * multiple * stage_discount
    lo, hi = _range_around(mid, 1.80)
    return MethodEstimate(
        name="revenue_multiple",
        value_mid_usd=mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence="medium",
        rationale=f"ARR * industry/country EV/Sales * stage discount = {arr:,.0f} * {multiple:.2f} * {stage_discount:.2f}",
    )


def _existing_script_method(payload: dict[str, Any], weight: float) -> MethodEstimate | None:
    mid = _to_float(payload.get("valuation_mid_usd"))
    lo = _to_float(payload.get("valuation_min_usd"))
    hi = _to_float(payload.get("valuation_max_usd"))
    if not mid or mid <= 0 or weight <= 0:
        return None
    lo = lo if lo and lo > 0 else mid / 1.8
    hi = hi if hi and hi > 0 else mid * 1.8
    return MethodEstimate(
        name="existing_script_anchor",
        value_mid_usd=mid,
        value_min_usd=lo,
        value_max_usd=hi,
        weight=weight,
        confidence=str(payload.get("confidence") or "low"),
        rationale="Current production deterministic valuation used as one method, not replaced.",
    )


def evaluate_multi_method_valuation(project_text: str, deterministic_payload: dict[str, Any], llm: Any | None = None) -> dict[str, Any]:
    """Experimental Equidam-inspired valuation layer.

    This function is intentionally independent from the production deterministic
    valuation path. It consumes the existing payload and returns an additive
    analysis block, so failures here do not affect the core valuation result.
    """
    payload = deterministic_payload or {}
    benchmarks = _load_benchmarks()
    stage = str(payload.get("stage") or "unknown")
    weights = _method_weights(stage)
    methods = [
        _ml_market_method(project_text, payload, weights["ml"], llm=llm),
        _scorecard_method(project_text, payload, benchmarks, weights["scorecard"]),
        _checklist_method(project_text, payload, benchmarks, weights["checklist"]),
        _vc_method(payload, weights["vc"]),
        _revenue_multiple_method(payload, weights["revenue_multiple"]),
        _existing_script_method(payload, weights["existing_script"]),
    ]
    available = [m for m in methods if m is not None and m.value_mid_usd > 0 and m.weight > 0]
    if not available:
        raise ValueError("No multi-method valuation methods were available")
    weight_sum = sum(m.weight for m in available) or 1.0
    for m in available:
        m.weight = m.weight / weight_sum
    mid = sum(m.value_mid_usd * m.weight for m in available)
    lo = sum(m.value_min_usd * m.weight for m in available)
    hi = sum(m.value_max_usd * m.weight for m in available)
    confidence_score = sum(m.weight for m in available if m.confidence in {"medium", "high"})
    confidence = "high" if confidence_score >= 0.75 and len(available) >= 4 else "medium" if confidence_score >= 0.45 else "low"
    return {
        "version": "multi_method_v2_experimental_with_ml",
        "stage": stage,
        "stage_family": _stage_family(stage),
        "valuation_min_usd": round(lo, 2),
        "valuation_mid_usd": round(mid, 2),
        "valuation_max_usd": round(hi, 2),
        "confidence": confidence,
        "methods": [m.to_dict() for m in available],
        "notes": [
            "Experimental layer: does not replace the current production deterministic valuation.",
            "ML market benchmark is used as an adjusted market prior when the model bundle and dependencies are available.",
            "Weights are stage-based and normalized over methods available for this input.",
        ],
    }
