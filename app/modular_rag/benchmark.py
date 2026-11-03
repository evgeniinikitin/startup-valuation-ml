from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supabase import Client

from .valuation import evaluate_project_deterministic


@dataclass
class BenchmarkRow:
    name: str
    stage: str
    expected_stage: str
    stage_match: bool
    region: str
    source: str
    year: int | None
    actual_usd: float
    predicted_usd: float
    abs_error_usd: float
    ape_pct: float | None
    ratio_pred_to_real: float | None
    method: str
    confidence: str
    stage_detected: str
    stage_detection_mode: str


@dataclass
class BenchmarkMetrics:
    n: int
    r2: float
    r2_log10: float
    mae_usd: float
    median_ae_usd: float
    mape_pct: float
    median_ratio_pred_to_real: float
    underpricing_share_pct: float
    severe_underpricing_share_pct: float


def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def parse_year_from_source(source: str) -> int | None:
    m = re.search(r"\((19\d{2}|20\d{2})\)", source or "")
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(19\d{2}|20\d{2})\b", source or "")
    if m2:
        return int(m2.group(1))
    return None


def _build_benchmark_input_text(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    stage = str(item.get("stage") or "").strip()
    source = str(item.get("source") or "").strip()
    year = parse_year_from_source(source)
    description = str(item.get("description") or "").strip()
    pitch = str(item.get("pitch_text") or "").strip()
    region = str(item.get("region") or "").strip()

    parts: list[str] = []
    # Structured header helps deterministic parser avoid default fallback.
    if name:
        parts.append(f"Project: {name}")
    if stage:
        parts.append(f"Stage: {stage}")
    if year is not None:
        parts.append(f"Project year: {year}")
    if region:
        parts.append(f"Region: {region}")
    if description:
        parts.append(f"Description: {description}")
    if pitch:
        parts.append(f"Pitch:\n{pitch}")

    # Optional structured numeric hints (if present in dataset).
    maybe_fields = [
        ("ARR", "arr_usd", "USD"),
        ("MRR", "mrr_usd", "USD"),
        ("Investment ask", "investment_ask_usd", "USD"),
        ("Equity offered", "equity_offered_pct", "%"),
    ]
    for label, key, suffix in maybe_fields:
        v = item.get(key)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if suffix == "%":
            parts.append(f"{label}: {vf}%")
        else:
            parts.append(f"{label}: {vf} {suffix}")

    return "\n".join(parts).strip()


def is_ai_case(item: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            str(item.get("pitch_text") or ""),
        ]
    ).lower()
    keys = [
        " ai ",
        "llm",
        "generative",
        "генератив",
        "нейросет",
        "искусственн",
        "foundation model",
        "gpt",
    ]
    return any(k in blob for k in keys)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    if n % 2 == 1:
        return float(vals[n // 2])
    return float((vals[n // 2 - 1] + vals[n // 2]) / 2.0)


def _r2(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        return 0.0
    if len(actual) < 2:
        return 0.0
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    if ss_tot <= 0:
        return 0.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(rows: list[BenchmarkRow]) -> BenchmarkMetrics:
    actual = [r.actual_usd for r in rows]
    predicted = [r.predicted_usd for r in rows]
    n = len(rows)
    if n == 0:
        return BenchmarkMetrics(
            n=0,
            r2=0.0,
            r2_log10=0.0,
            mae_usd=0.0,
            median_ae_usd=0.0,
            mape_pct=0.0,
            median_ratio_pred_to_real=0.0,
            underpricing_share_pct=0.0,
            severe_underpricing_share_pct=0.0,
        )

    abs_err = [abs(a - p) for a, p in zip(actual, predicted)]
    ape = [(abs(a - p) / a) * 100.0 for a, p in zip(actual, predicted) if a > 0]
    ratios = [(p / a) for a, p in zip(actual, predicted) if a > 0]

    eps = 1.0
    actual_log = [math.log10(max(a, eps)) for a in actual]
    predicted_log = [math.log10(max(p, eps)) for p in predicted]

    under = sum(1 for a, p in zip(actual, predicted) if p < a)
    severe_under = sum(1 for a, p in zip(actual, predicted) if p < (a * 0.30))

    return BenchmarkMetrics(
        n=n,
        r2=round(_r2(actual, predicted), 4),
        r2_log10=round(_r2(actual_log, predicted_log), 4),
        mae_usd=round(sum(abs_err) / n, 2),
        median_ae_usd=round(_median(abs_err), 2),
        mape_pct=round((sum(ape) / len(ape)) if ape else 0.0, 2),
        median_ratio_pred_to_real=round(_median(ratios), 4),
        underpricing_share_pct=round((under / n) * 100.0, 2),
        severe_underpricing_share_pct=round((severe_under / n) * 100.0, 2),
    )


def _evaluate_with_optional_llm(
    text: str,
    supabase: Client,
    llm: Any | None,
    use_llm_stage: bool,
):
    sig = inspect.signature(evaluate_project_deterministic)
    if "llm" in sig.parameters:
        if use_llm_stage:
            return evaluate_project_deterministic(text, supabase, llm=llm)
        return evaluate_project_deterministic(text, supabase, llm=None)
    return evaluate_project_deterministic(text, supabase)


def run_benchmark(
    dataset: list[dict[str, Any]],
    supabase: Client,
    llm: Any | None = None,
    use_llm_stage: bool = False,
) -> tuple[list[BenchmarkRow], BenchmarkMetrics, str]:
    rows: list[BenchmarkRow] = []
    log_lines = [
        "=== Benchmark Run ===",
        f"dataset_size={len(dataset)}",
        f"use_llm_stage={use_llm_stage}",
    ]
    stage_map = {
        "pre-seed": "idea_only",
        "pre seed": "idea_only",
        "seed": "mvp_no_revenue",
        "series a": "growth",
        "series b": "growth",
        "series c": "growth",
    }
    for i, item in enumerate(dataset, start=1):
        name = str(item.get("name") or f"case_{i}")
        stage = str(item.get("stage") or "")
        region = str(item.get("region") or "")
        source = str(item.get("source") or "")
        year = parse_year_from_source(source)
        try:
            actual_usd = float(item.get("real_valuation_usd") or 0.0)
        except Exception:
            actual_usd = 0.0
        if actual_usd <= 0:
            log_lines.append(f"[skip] {name}: invalid real_valuation_usd={item.get('real_valuation_usd')}")
            continue

        text = str(item.get("pitch_text") or item.get("description") or "").strip()
        text = _build_benchmark_input_text(item)
        if not text:
            log_lines.append(f"[skip] {name}: empty pitch_text/description")
            continue

        try:
            result = _evaluate_with_optional_llm(
                text=text,
                supabase=supabase,
                llm=llm,
                use_llm_stage=use_llm_stage,
            )
            predicted_usd = float(result.valuation_mid_usd or 0.0)
        except Exception as e:
            log_lines.append(f"[err] {name}: {str(e)[:180]}")
            continue

        abs_error = abs(actual_usd - predicted_usd)
        ape_pct = (abs_error / actual_usd) * 100.0 if actual_usd > 0 else None
        ratio = (predicted_usd / actual_usd) if actual_usd > 0 else None
        expected = stage_map.get(stage.strip().lower(), "")
        stage_match = (result.stage == expected) if expected else False
        rows.append(
            BenchmarkRow(
                name=name,
                stage=stage,
                expected_stage=expected,
                stage_match=stage_match,
                region=region,
                source=source,
                year=year,
                actual_usd=actual_usd,
                predicted_usd=predicted_usd,
                abs_error_usd=abs_error,
                ape_pct=ape_pct,
                ratio_pred_to_real=ratio,
                method=result.method,
                confidence=result.confidence,
                stage_detected=result.stage,
                stage_detection_mode=result.stage_detection_mode,
            )
        )
        log_lines.append(
            f"[{i}] {name}: real={actual_usd:,.0f}, pred={predicted_usd:,.0f}, "
            f"ratio={(ratio or 0):.3f}, stage={result.stage}, expected={expected or 'n/a'}, "
            f"stage_match={stage_match}, year={year}"
        )

    metrics = compute_metrics(rows)
    log_lines.append(
        "summary: "
        f"n={metrics.n}, R2={metrics.r2}, R2_log10={metrics.r2_log10}, "
        f"MAE={metrics.mae_usd:,.0f}, MAPE={metrics.mape_pct}%"
    )
    return rows, metrics, "\n".join(log_lines)
