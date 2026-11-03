from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from supabase import Client


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokenize(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-zа-яё]{3,}", _norm(s))}


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:
        return None
    return f


def _median_positive(values: list[Any]) -> float | None:
    vals: list[float] = []
    for v in values:
        f = _to_float(v)
        if f and f > 0:
            vals.append(f)
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _parse_number(raw: str) -> float | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    s = s.replace("\u00a0", " ")
    s = s.replace("$", "").replace("usd", "").replace("eur", "").replace("rub", "").replace("₽", "")
    s = s.replace(" ", "")
    s = s.replace("миллионов", "млн").replace("миллиона", "млн").replace("миллиардов", "млрд")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:[a-zа-яё]+)?", s):
            s = s.replace(".", "")
    elif "," in s:
        if re.search(r",\d{1,2}(?:[a-zа-яё]+)?$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([kmb]|тыс|млн|млрд)?", s)
    if not m:
        return None
    value = float(m.group(1))
    suffix = m.group(2) or ""
    mult = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "тыс": 1_000,
        "млн": 1_000_000,
        "млрд": 1_000_000_000,
    }
    return value * mult.get(suffix, 1.0)


def _extract_amount_with_currency(text: str, patterns: list[str]) -> tuple[float | None, str | None]:
    t = text or ""
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            raw = (m.group("amount") or "").strip()
            cur = (m.group("cur") or "").strip().lower()
            val = _parse_number(raw)
            if val is None:
                continue
            if "$" in raw or "usd" in cur:
                return val, "USD"
            if "€" in raw or "eur" in cur:
                return val, "EUR"
            if "₽" in raw or "руб" in cur or "rub" in cur:
                return val, "RUB"
            return val, "USD"
    return None, None


def _to_usd(value: float | None, currency: str | None) -> float | None:
    if value is None:
        return None
    cur = (currency or "USD").upper()
    fx_rub_usd = float(os.getenv("VALUATION_FX_RUB_USD", "92"))
    fx_eur_usd = float(os.getenv("VALUATION_FX_EUR_USD", "1.08"))
    if cur == "USD":
        return value
    if cur == "RUB":
        return value / fx_rub_usd
    if cur == "EUR":
        return value * fx_eur_usd
    return value


def extract_revenue_metrics(text: str) -> dict[str, Any]:
    num = r"(?:\d{1,3}(?:[ \u00A0.,]\d{3})+|\d+)(?:[.,]\d+)?"
    monthly_patterns = [
        rf"(?:mrr|в\s*месяц|/мес|monthly)[^\n\d]{{0,50}}(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?",
        rf"(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?[^\n]{{0,50}}(?:mrr|в\s*месяц|/мес|monthly)",
    ]
    annual_patterns = [
        rf"(?:arr|в\s*год|/год|annual|годовая\s*выручка)[^\n\d]{{0,50}}(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?",
        rf"(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?[^\n]{{0,50}}(?:arr|в\s*год|/год|annual|годовая\s*выручка)",
    ]
    monthly_val, monthly_cur = _extract_amount_with_currency(text, monthly_patterns)
    annual_val, annual_cur = _extract_amount_with_currency(text, annual_patterns)
    monthly_usd = _to_usd(monthly_val, monthly_cur)
    annual_usd = _to_usd(annual_val, annual_cur)
    if annual_usd is None and monthly_usd is not None:
        annual_usd = monthly_usd * 12
    if monthly_usd is None and annual_usd is not None:
        monthly_usd = annual_usd / 12
    return {
        "monthly_usd": monthly_usd,
        "annual_usd": annual_usd,
        "monthly_raw": monthly_val,
        "monthly_currency": monthly_cur,
        "annual_raw": annual_val,
        "annual_currency": annual_cur,
    }


def detect_stage_llm(llm: Any, text: str, annual_usd: float | None) -> tuple[str | None, str | None]:
    if llm is None:
        return None, None
    clipped = (text or "")[:10000]
    prompt = f"""
Return strict JSON only.
Task: detect stage ONLY for the target startup (ignore competitors and market examples).
Allowed values: idea_only, mvp_no_revenue, early_revenue, growth.
Use ARR={annual_usd if annual_usd is not None else "None"} only if explicitly tied to target startup.
Schema:
{{"stage":"...", "confidence":"low|medium|high", "reason":"short"}}

Text:
{clipped}
    """.strip()
    try:
        raw = llm.generate(prompt, temperature=0.0, max_output_tokens=200)
        data = json.loads(raw)
        stage = str((data or {}).get("stage", "")).strip()
        conf = str((data or {}).get("confidence", "")).strip()
        if stage in {"idea_only", "mvp_no_revenue", "early_revenue", "growth"}:
            return stage, conf or None
    except Exception:
        pass
    return None, None


def detect_stage(text: str, annual_usd: float | None, llm: Any | None = None) -> tuple[str, str]:
    llm_stage, llm_conf = detect_stage_llm(llm, text, annual_usd)
    if llm_stage:
        return llm_stage, f"llm:{llm_conf or 'n/a'}"

    t = _norm(text)
    stage_line = re.search(
        r"(?:стадия|stage)\s*[:\-]?\s*(pre[-\s]?seed|seed|idea|mvp|prototype|pilot|series\s*[abc])",
        t,
    )
    if stage_line:
        raw = stage_line.group(1)
        if any(x in raw for x in ["series a", "series b", "series c"]):
            return "growth", "regex:stage_line"
        if "mvp" in raw or "prototype" in raw or "pilot" in raw:
            return "mvp_no_revenue", "regex:stage_line"
        if "pre-seed" in raw:
            return "idea_only", "regex:stage_line"
        if "seed" in raw:
            if annual_usd and annual_usd > 0:
                return "early_revenue", "regex:stage_line+arr"
            return "mvp_no_revenue", "regex:stage_line"
        if "idea" in raw:
            return "idea_only", "regex:stage_line"

    if annual_usd and annual_usd > 0:
        if annual_usd >= 1_500_000:
            return "growth", "arr_threshold"
        return "early_revenue", "arr_threshold"

    if any(x in t for x in ["mvp", "прототип", "prototype", "pilot", "пилот"]):
        return "mvp_no_revenue", "keyword:mvp"
    if any(x in t for x in ["идея", "idea", "концепц", "pre-seed", "pre seed"]):
        return "idea_only", "keyword:idea"
    return "mvp_no_revenue", "default"


def detect_country(text: str) -> str:
    t = _norm(text)
    if any(x in t for x in ["росси", "рф", "russia", "russian"]):
        return "Russia"
    if any(x in t for x in ["europe", "европ", "eu "]):
        return "Europe"
    if any(x in t for x in ["india", "инд"]):
        return "India"
    if any(x in t for x in ["usa", "united states", "сша"]):
        return "USA"
    return "Global"


def infer_industry_hint(text: str) -> str:
    t = _norm(text)
    rules = [
        ("software", ["saas", "software", "b2b", "crm", "platform", "платформ", "developer"]),
        ("ai", ["ai", "llm", "нейросет", "машинн", "искусствен"]),
        ("fintech", ["fintech", "payments", "банк", "финанс", "кредит", "страхован"]),
        ("healthcare", ["health", "med", "мед", "клиника", "биотех"]),
        ("education", ["edtech", "education", "обуч", "репетитор", "язык"]),
        ("ecommerce", ["ecommerce", "marketplace", "маркетплейс", "retail", "commerce"]),
        ("logistics", ["logistics", "delivery", "достав", "склад", "supply chain"]),
        ("real_estate", ["real estate", "proptech", "недвиж", "аренд"]),
        ("climate", ["climate", "carbon", "solar", "энерг", "co2"]),
        ("travel", ["travel", "hospitality", "туризм", "отел", "booking"]),
    ]
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    return "software"


def detect_valuation_year(text: str) -> int | None:
    t = text or ""
    current_year = datetime.now().year
    matches: list[tuple[int, int]] = []
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", t):
        year = int(m.group(1))
        if year < 1998 or year > current_year + 1:
            continue
        start = max(0, m.start() - 48)
        end = min(len(t), m.end() + 48)
        ctx = _norm(t[start:end])
        score = 1
        if any(k in ctx for k in ["дата презентац", "presentation date", "pitch date", "valuation date", "as of"]):
            score = 6
        elif any(k in ctx for k in ["на дату", "отчет", "report", "год проекта", "project year"]):
            score = 5
        elif any(k in ctx for k in ["основан", "founded", "established", "launch", "старт"]):
            score = 2
        elif any(k in ctx for k in ["выруч", "arr", "mrr", "revenue"]):
            score = 4
        matches.append((score, year))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return matches[0][1]


def _fetch_rows_all(
    supabase: Client,
    table_name: str,
    columns: str,
    filters: dict[str, Any] | None = None,
    batch_size: int = 1000,
) -> list[dict]:
    out: list[dict] = []
    offset = 0
    filters = filters or {}
    while True:
        q = supabase.table(table_name).select(columns)
        for k, v in filters.items():
            q = q.eq(k, v)
        res = q.range(offset, offset + batch_size - 1).execute()
        rows = res.data or []
        if not rows:
            break
        out.extend(rows)
        if len(rows) < batch_size:
            break
        offset += batch_size
    return out


def _fetch_rows_by_source(supabase: Client, source_id: str, limit: int = 2500) -> list[dict]:
    res = supabase.table("documents").select("content,metadata").eq("metadata->>source_id", source_id).limit(limit).execute()
    return res.data or []


def _resolve_nearest_year(target_year: int | None, available_years: list[int]) -> int | None:
    if not available_years:
        return None
    years = sorted(set(int(y) for y in available_years if y))
    if not years:
        return None
    if target_year is None:
        return years[-1]
    less_or_equal = [y for y in years if y <= target_year]
    if less_or_equal:
        return less_or_equal[-1]
    return years[0]


def _pick_best_industry_sql(rows: list[dict], startup_text: str) -> dict | None:
    if not rows:
        return None
    text_tokens = _tokenize(startup_text)
    best = None
    best_score = -1
    for r in rows:
        name = str(r.get("industry_name") or "")
        score = len(text_tokens & _tokenize(name))
        if score > best_score:
            best_score = score
            best = r
    return best or rows[0]


def _country_aliases(country_hint: str) -> list[str]:
    c = _norm(country_hint)
    if c == "russia":
        return ["russia", "russian federation", "russian", "россия", "рф"]
    if c == "usa":
        return ["usa", "united states", "us", "america", "сша"]
    if c == "india":
        return ["india", "индия"]
    if c == "europe":
        return ["europe", "eu", "европа"]
    if c == "global":
        return ["global", "world"]
    return [c]


def _pick_country_row_sql(rows: list[dict], country_hint: str, dataset_family: str | None = None) -> dict | None:
    if not rows:
        return None
    aliases = _country_aliases(country_hint)
    candidates = [r for r in rows if dataset_family is None or _norm(str(r.get("dataset_family") or "")) == _norm(dataset_family)]
    if not candidates:
        candidates = rows
    best = None
    best_score = -1
    for r in candidates:
        country = _norm(str(r.get("country") or ""))
        score = 0
        for a in aliases:
            if a in country or country in a:
                score = max(score, 3)
            else:
                score = max(score, len(_tokenize(a) & _tokenize(country)))
        if score > best_score:
            best_score = score
            best = r
    return best or candidates[0]


def _pick_country_ev_sales_from_companies(rows: list[dict], country_hint: str) -> float | None:
    aliases = _country_aliases(country_hint)
    vals: list[float] = []
    for r in rows:
        c = _norm(str(r.get("country") or ""))
        if not c:
            continue
        if not any(a in c or c in a for a in aliases):
            continue
        f = _to_float(r.get("ev_sales"))
        if f and f > 0:
            vals.append(f)
    return _median_positive(vals)


def _extract_deal_terms(text: str) -> dict[str, Any]:
    num = r"(?:\d{1,3}(?:[ \u00A0.,]\d{3})+|\d+)(?:[.,]\d+)?"
    ask_patterns = [
        rf"(?:требуем\w*\s+объем\s+финанс\w*|нужно\s+инвестиц\w*|raise|raising|investment\s+ask)[^\n\d]{{0,50}}(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?",
        rf"(?P<amount>[\$€₽]?\s*{num}\s*(?:k|m|b|тыс|млн|млрд)?)\s*(?P<cur>usd|eur|rub|руб(?:\.|лей|ля)?)?[^\n]{{0,40}}(?:инвестиц|финанс|raise|round)",
    ]
    equity_match = re.search(r"(\d{1,2}(?:[\.,]\d+)?)\s*%[^\n]{0,35}(?:дол|equity|компан)", text or "", flags=re.IGNORECASE)
    ask_raw, ask_cur = _extract_amount_with_currency(text, ask_patterns)
    ask_usd = _to_usd(ask_raw, ask_cur)
    equity_pct = None
    if equity_match:
        eq = equity_match.group(1).replace(",", ".")
        try:
            equity_pct = float(eq)
        except Exception:
            equity_pct = None
    return {
        "investment_ask_raw": ask_raw,
        "investment_ask_currency": ask_cur,
        "investment_ask_usd": ask_usd,
        "equity_offered_pct": equity_pct,
    }


def _round(v: float | None) -> float | None:
    if v is None:
        return None
    return float(f"{v:.2f}")


def _load_benchmarks() -> dict[str, Any]:
    p = Path(__file__).resolve().parents[1] / "vc_benchmarks.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _regional_discount(country_hint: str, benchmarks: dict[str, Any]) -> float:
    reg = (benchmarks.get("regional_discount") or {})
    c = _norm(country_hint)
    if c == "russia":
        return float(reg.get("russia", 0.30))
    if c == "europe":
        return float(reg.get("europe", 0.60))
    if c == "india":
        return float(reg.get("india", 0.40))
    if c == "usa":
        return float(reg.get("us", 1.0))
    return float(reg.get("us", 1.0))


def _stage_to_benchmark_key(stage: str) -> str:
    if stage == "idea_only":
        return "Pre-Seed"
    if stage == "mvp_no_revenue":
        return "Seed"
    if stage == "early_revenue":
        return "Seed"
    return "Series A"


def _scorecard_anchor(stage: str, country_hint: str, industry_hint: str, text: str, benchmarks: dict[str, Any]) -> tuple[float | None, list[str]]:
    notes: list[str] = []
    us = benchmarks.get("us_medians") or {}
    key = _stage_to_benchmark_key(stage)
    base = _to_float((us.get(key) or {}).get("pre_money_valuation_usd"))
    if not base:
        return None, notes
    reg = _regional_discount(country_hint, benchmarks)
    ai_mult = _to_float(benchmarks.get("ai_premium_multiplier")) or 1.0
    ai_adj = ai_mult if industry_hint == "ai" else 1.0
    t = _norm(text)
    team = 1.08 if any(x in t for x in ["ex-google", "ex-meta", "экс-", "serial founder", "серийный"]) else 0.95
    traction = 1.08 if any(x in t for x in ["paying users", "платящ", "pilot", "пилот", "contracts", "контракт"]) else 0.92
    product = 1.05 if any(x in t for x in ["mvp", "production", "релиз", "launched", "запущ"]) else 0.90
    comp = 0.90 if any(x in t for x in ["high competition", "высокая конкур", "crowded"]) else 1.00
    adj = reg * ai_adj * team * traction * product * comp
    val = base * adj
    notes.append(f"scorecard_base={base:,.0f} ({key})")
    notes.append(f"scorecard_adjustment={adj:.3f} (region/team/traction/product/competition)")
    return val, notes


def _berkus_anchor(stage: str, country_hint: str, text: str, benchmarks: dict[str, Any]) -> tuple[float | None, list[str]]:
    if stage not in {"idea_only", "mvp_no_revenue"}:
        return None, []
    reg = _regional_discount(country_hint, benchmarks)
    t = _norm(text)
    scores = {
        "idea": 1.0 if any(x in t for x in ["problem", "проблем", "pain"]) else 0.4,
        "prototype": 1.0 if any(x in t for x in ["mvp", "prototype", "прототип"]) else 0.2,
        "team": 1.0 if any(x in t for x in ["team", "команд", "founder"]) else 0.3,
        "strategic": 1.0 if any(x in t for x in ["partner", "партнер", "advisor", "ментор"]) else 0.2,
        "rollout": 1.0 if any(x in t for x in ["go-to-market", "gtm", "sales", "продаж"]) else 0.2,
    }
    # 5 factors * 500k USD each as US baseline.
    us = 500_000.0 * sum(scores.values())
    val = us * reg
    notes = [f"berkus_factors={scores}", f"berkus_us={us:,.0f}", f"berkus_region_discount={reg:.2f}"]
    return val, notes


def _quality_gate(stage: str, arr: float | None, deal: dict[str, Any], text: str) -> tuple[float, list[str]]:
    penalty = 1.0
    flags: list[str] = []
    ask = deal.get("investment_ask_usd")
    eq = deal.get("equity_offered_pct")
    t = _norm(text)

    if stage in {"idea_only", "mvp_no_revenue"} and arr and arr > 0:
        penalty *= 0.85
        flags.append("stage_arr_inconsistency")
    if arr and arr > 0 and any(x in t for x in ["projected", "forecast", "план", "прогноз"]) and not any(x in t for x in ["fact", "actual", "факт", "202"]):
        penalty *= 0.88
        flags.append("revenue_may_be_forecast")
    if ask and ask > 0 and eq and eq > 0:
        implied_pre = ask / (eq / 100.0) - ask
        if implied_pre < 0:
            penalty *= 0.9
            flags.append("invalid_deal_terms")
    if not any(x in t for x in ["mvp", "prototype", "прототип", "launch", "релиз"]):
        penalty *= 0.93
        flags.append("no_product_evidence")
    return penalty, flags


def _pick_keyword_comparables(rows: list[dict], startup_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not rows:
        return []
    q = _tokenize(startup_text)
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        content = str(r.get("content") or "")
        if not content:
            continue
        meta = r.get("metadata") or {}
        t = _tokenize(content)
        overlap = len(q & t)
        if overlap == 0:
            continue
        # Prefer sources with more valuation relevance.
        sid = _norm(str(meta.get("source_id") or ""))
        boost = 0.0
        if "sec_" in sid or "investment" in sid:
            boost = 0.12
        elif "indian" in sid or "market" in sid:
            boost = 0.08
        elif "trustmrr" in sid:
            boost = -0.05
        score = overlap / max(1, len(q)) + boost
        scored.append((score, {"score": round(score, 4), "content": content[:600], "metadata": meta}))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen = set()
    for _, item in scored:
        sig = (item["content"][:120], str(item.get("metadata", {}).get("source_id", "")))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _build_mini_report(
    stage: str,
    method: str,
    vmin: float,
    mid: float,
    vmax: float,
    confidence: str,
    analogs: list[dict[str, Any]],
    investment_comps: list[dict[str, Any]],
    notes: list[str],
) -> str:
    lines = [
        f"Stage: {stage}",
        f"Method: {method}",
        f"Valuation range (USD): {vmin:,.0f} - {vmax:,.0f}, mid={mid:,.0f}",
        f"Confidence: {confidence}",
        f"Analogs found: {len(analogs)} | Investment comps found: {len(investment_comps)}",
    ]
    if notes:
        lines.append("Notes: " + "; ".join(notes))
    return "\n".join(lines)


@dataclass
class DeterministicValuationResult:
    stage: str
    stage_detection_mode: str
    method: str
    valuation_min_usd: float
    valuation_mid_usd: float
    valuation_max_usd: float
    scenario_downside_usd: float
    scenario_base_usd: float
    scenario_upside_usd: float
    confidence: str
    country_hint: str
    industry_hint: str
    ev_sales_used: float | None
    country_ev_sales_used: float | None
    arr_usd: float | None
    mrr_usd: float | None
    investment_ask_usd: float | None
    equity_offered_pct: float | None
    damodaran_year_used: int | None
    country_risk_premium_used: float | None
    tax_rate_used: float | None
    quality_gate_flags: list[str]
    notes: list[str]
    calculation_steps: list[str]
    analogs: list[dict[str, Any]]
    investment_comps: list[dict[str, Any]]
    mini_report: str


def evaluate_project_deterministic(text: str, supabase: Client, llm: Any | None = None) -> DeterministicValuationResult:
    startup_text = text or ""
    revenue = extract_revenue_metrics(startup_text)
    arr = revenue.get("annual_usd")
    mrr = revenue.get("monthly_usd")
    stage, stage_mode = detect_stage(startup_text, arr, llm=llm)
    country_hint = detect_country(startup_text)
    industry_hint = infer_industry_hint(startup_text)
    valuation_year_hint = detect_valuation_year(startup_text)
    deal = _extract_deal_terms(startup_text)
    benchmarks = _load_benchmarks()

    available_year_rows = _fetch_rows_all(supabase, "damodaran_industry_metrics", "year")
    available_years = [int(r.get("year")) for r in available_year_rows if r.get("year") is not None]
    damodaran_year_used = _resolve_nearest_year(valuation_year_hint, available_years)

    ps_rows: list[dict] = []
    country_rows: list[dict] = []
    company_rows: list[dict] = []
    sql_mode = False
    if damodaran_year_used is not None:
        ps_rows = _fetch_rows_all(
            supabase,
            "damodaran_industry_metrics",
            "dataset_family,year,industry_name,number_of_firms,ev_sales",
            filters={"dataset_family": "psdata", "year": damodaran_year_used},
        )
        country_rows = _fetch_rows_all(
            supabase,
            "damodaran_country_metrics",
            "dataset_family,year,country,country_risk_premium,equity_risk_premium,default_spread,tax_rate",
            filters={"year": damodaran_year_used},
        )
        company_rows = _fetch_rows_all(
            supabase,
            "damodaran_global_companies",
            "year,country,ev_sales",
            filters={"year": damodaran_year_used},
        )
        sql_mode = bool(ps_rows)

    strict_sql = os.getenv("VALUATION_STRICT_DAMODARAN_SQL", "0").strip() in {"1", "true", "yes"}
    if not sql_mode and strict_sql:
        raise RuntimeError("Damodaran SQL tables are empty for valuation. Load SQL data first.")

    if not sql_mode:
        ps_rows = _fetch_rows_by_source(supabase, "damodaran_ps")
        country_rows = _fetch_rows_by_source(supabase, "damodaran_country")

    # Comparables pools with explicit priority.
    investment_pool: list[dict] = []
    for sid in (
        "sec_formd_q1_2026",
        "sec_investments_vc",
        "investment_summary.csv",
        "investments_vc.csv",
        "indian.csv",
        "markets.csv",
        "global_market_sizes.csv",
        "trustmrr_master_database.csv",
    ):
        investment_pool.extend(_fetch_rows_by_source(supabase, sid, limit=2000))

    analog_pool = investment_pool
    analogs = _pick_keyword_comparables(analog_pool, startup_text, top_k=5)
    investment_rows = []
    for r in investment_pool:
        sid = _norm(str((r.get("metadata") or {}).get("source_id", "")))
        if ("sec_" in sid) or ("investment" in sid):
            investment_rows.append(r)
    investment_comps = _pick_keyword_comparables(investment_rows, startup_text, top_k=5)

    ev_sales = None
    country_ev_sales = None
    country_risk_premium = None
    tax_rate = None
    if sql_mode:
        best_ps = _pick_best_industry_sql(ps_rows, startup_text)
        ev_sales = _to_float(best_ps.get("ev_sales") if best_ps else None) or _median_positive([r.get("ev_sales") for r in ps_rows])
        country_ev_sales = _pick_country_ev_sales_from_companies(company_rows, country_hint) or _median_positive([r.get("ev_sales") for r in company_rows])
        c_crp = _pick_country_row_sql(country_rows, country_hint, dataset_family="ctryprem")
        c_tax = _pick_country_row_sql(country_rows, country_hint, dataset_family="countrytaxrates")
        country_risk_premium = _to_float(c_crp.get("country_risk_premium") if c_crp else None)
        tax_rate = _to_float(c_tax.get("tax_rate") if c_tax else None)
    else:
        # Legacy fallback (documents)
        if ps_rows:
            meta = (ps_rows[0].get("metadata") or {})
            ev_sales = _to_float(meta.get("metric_ev_sales"))
        if country_rows:
            meta = (country_rows[0].get("metadata") or {})
            country_ev_sales = _to_float(meta.get("metric_median_ev_sales"))

    notes: list[str] = []
    calc: list[str] = []
    calc.append(f"1) detect_stage -> {stage} ({stage_mode})")
    calc.append(f"2) detect_country -> {country_hint}")
    calc.append(f"3) infer_industry_hint -> {industry_hint}")
    calc.append(
        "3.1) valuation_year_detect -> "
        f"hint={valuation_year_hint if valuation_year_hint is not None else 'None'}, "
        f"damodaran_year_used={damodaran_year_used if damodaran_year_used is not None else 'fallback'}"
    )
    calc.append(
        "4) extracted_metrics -> "
        f"ARR={arr if arr is not None else 'None'} USD, "
        f"MRR={mrr if mrr is not None else 'None'} USD, "
        f"ASK={deal.get('investment_ask_usd') if deal.get('investment_ask_usd') is not None else 'None'} USD, "
        f"EQUITY={deal.get('equity_offered_pct') if deal.get('equity_offered_pct') is not None else 'None'}%"
    )
    calc.append(
        "5) damodaran_inputs -> "
        f"industry_ev_sales={ev_sales if ev_sales is not None else 'None'}, "
        f"country_ev_sales={country_ev_sales if country_ev_sales is not None else 'None'}, "
        f"country_risk_premium={country_risk_premium if country_risk_premium is not None else 'None'}, "
        f"tax_rate={tax_rate if tax_rate is not None else 'None'}, mode={'sql' if sql_mode else 'documents_fallback'}"
    )
    calc.append(f"5.1) comparable_search -> analogs={len(analogs)}, investment_comps={len(investment_comps)}")

    candidates: list[tuple[str, float, float]] = []
    # (name, mid_value, weight)
    if arr and arr > 0:
        mult_candidates = [x for x in [ev_sales, country_ev_sales] if isinstance(x, (int, float)) and x > 0]
        base_mult = sum(mult_candidates) / len(mult_candidates) if mult_candidates else 1.4
        stage_discount = {
            "growth": 0.85,
            "early_revenue": 0.55,
            "mvp_no_revenue": 0.30,
            "idea_only": 0.18,
        }.get(stage, 0.55)
        country_factor = 1.0
        if country_risk_premium and country_risk_premium > 0:
            country_factor = max(0.55, min(1.05, 1.0 - 2.4 * country_risk_premium))
        mid_rev = arr * base_mult * stage_discount * country_factor
        candidates.append(("revenue_comps_damodaran_formula", mid_rev, 0.45))
        calc.append(
            "6) revenue_comps -> "
            f"ARR*mult*stage_discount*country_factor = {arr:,.2f}*{base_mult:.4f}*{stage_discount:.4f}*{country_factor:.4f}={mid_rev:,.2f}"
        )
        notes.append(f"revenue_comps: base_mult={base_mult:.2f}, stage_discount={stage_discount:.2f}, country_factor={country_factor:.2f}")

    ask_usd = deal.get("investment_ask_usd")
    equity_pct = deal.get("equity_offered_pct")
    if ask_usd and ask_usd > 0 and equity_pct and 0 < equity_pct < 100:
        post = ask_usd / (equity_pct / 100.0)
        pre = post - ask_usd
        candidates.append(("deal_implied_formula", pre, 0.35))
        calc.append(f"7) deal_implied -> pre={pre:,.2f} (ask={ask_usd:,.2f}, equity={equity_pct:.2f}%)")

    scorecard_mid, scorecard_notes = _scorecard_anchor(stage, country_hint, industry_hint, startup_text, benchmarks)
    if scorecard_mid:
        candidates.append(("scorecard_benchmark_formula", scorecard_mid, 0.25))
        notes.extend(scorecard_notes)
        calc.append(f"8) scorecard_anchor -> {scorecard_mid:,.2f}")

    berkus_mid, berkus_notes = _berkus_anchor(stage, country_hint, startup_text, benchmarks)
    if berkus_mid:
        candidates.append(("berkus_formula", berkus_mid, 0.30))
        notes.extend(berkus_notes)
        calc.append(f"9) berkus_anchor -> {berkus_mid:,.2f}")

    if not candidates:
        # Last-resort conservative prior.
        bands = {
            "idea_only": 12_000.0,
            "mvp_no_revenue": 45_000.0,
            "early_revenue": 180_000.0,
            "growth": 800_000.0,
        }
        candidates.append(("deterministic_stage_band_formula", bands.get(stage, 45_000.0), 1.0))
        calc.append("10) fallback_stage_band -> no methods available, use deterministic prior")

    weight_sum = sum(w for _, _, w in candidates) or 1.0
    mid = sum(v * w for _, v, w in candidates) / weight_sum
    method = "triangulated_" + "+".join([name for name, _, _ in candidates])
    calc.append(f"11) triangulation -> weighted_mid={mid:,.2f} from {len(candidates)} methods")

    # Quality gate.
    q_penalty, q_flags = _quality_gate(stage, arr, deal, startup_text)
    if q_penalty < 1.0:
        calc.append(f"12) quality_gate_penalty -> {q_penalty:.4f} flags={q_flags}")
        mid *= q_penalty
    else:
        calc.append("12) quality_gate_penalty -> 1.0000 (no major inconsistencies)")

    # If deal-implied exists, keep triangulated result within a reasonable distance.
    if ask_usd and ask_usd > 0 and equity_pct and 0 < equity_pct < 100:
        implied_pre = ask_usd / (equity_pct / 100.0) - ask_usd
        lower = implied_pre * 0.55
        upper = implied_pre * 2.30
        old_mid = mid
        mid = min(max(mid, lower), upper)
        calc.append(f"13) deal_anchor_clamp -> clamp({old_mid:,.2f}, {lower:,.2f}, {upper:,.2f})={mid:,.2f}")

    # Sensitivity.
    downside = mid * 0.65
    upside = mid * 1.45
    vmin = downside
    vmax = upside
    calc.append(f"14) sensitivity -> downside={downside:,.2f}, base={mid:,.2f}, upside={upside:,.2f}")

    confidence = "low"
    if len(candidates) >= 3 and q_penalty >= 0.9:
        confidence = "high"
    elif len(candidates) >= 2:
        confidence = "medium"

    notes.append(f"methods={[(n, round(v,2), w) for n, v, w in candidates]}")
    notes.append(f"quality_gate_penalty={q_penalty:.2f}")
    notes.append(f"damodaran_mode={'sql' if sql_mode else 'documents_fallback'}")

    mini = _build_mini_report(stage, method, vmin, mid, vmax, confidence, analogs, investment_comps, notes)
    return DeterministicValuationResult(
        stage=stage,
        stage_detection_mode=stage_mode,
        method=method,
        valuation_min_usd=_round(vmin) or 0.0,
        valuation_mid_usd=_round(mid) or 0.0,
        valuation_max_usd=_round(vmax) or 0.0,
        scenario_downside_usd=_round(downside) or 0.0,
        scenario_base_usd=_round(mid) or 0.0,
        scenario_upside_usd=_round(upside) or 0.0,
        confidence=confidence,
        country_hint=country_hint,
        industry_hint=industry_hint,
        ev_sales_used=_round(ev_sales),
        country_ev_sales_used=_round(country_ev_sales),
        arr_usd=_round(arr),
        mrr_usd=_round(mrr),
        investment_ask_usd=_round(ask_usd),
        equity_offered_pct=equity_pct,
        damodaran_year_used=damodaran_year_used,
        country_risk_premium_used=_round(country_risk_premium),
        tax_rate_used=_round(tax_rate),
        quality_gate_flags=q_flags,
        notes=notes,
        calculation_steps=calc,
        analogs=analogs,
        investment_comps=investment_comps,
        mini_report=mini,
    )


def valuation_result_to_dict(result: DeterministicValuationResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "stage_detection_mode": result.stage_detection_mode,
        "method": result.method,
        "valuation_min_usd": result.valuation_min_usd,
        "valuation_mid_usd": result.valuation_mid_usd,
        "valuation_max_usd": result.valuation_max_usd,
        "scenario_downside_usd": result.scenario_downside_usd,
        "scenario_base_usd": result.scenario_base_usd,
        "scenario_upside_usd": result.scenario_upside_usd,
        "confidence": result.confidence,
        "country_hint": result.country_hint,
        "industry_hint": result.industry_hint,
        "ev_sales_used": result.ev_sales_used,
        "country_ev_sales_used": result.country_ev_sales_used,
        "arr_usd": result.arr_usd,
        "mrr_usd": result.mrr_usd,
        "investment_ask_usd": result.investment_ask_usd,
        "equity_offered_pct": result.equity_offered_pct,
        "damodaran_year_used": result.damodaran_year_used,
        "country_risk_premium_used": result.country_risk_premium_used,
        "tax_rate_used": result.tax_rate_used,
        "quality_gate_flags": result.quality_gate_flags,
        "notes": result.notes,
        "calculation_steps": result.calculation_steps,
        "analogs": result.analogs,
        "investment_comps": result.investment_comps,
        "mini_report": result.mini_report,
    }


def generate_ai_valuation_diagnostics(
    llm: Any,
    project_text: str,
    valuation_payload: dict[str, Any],
) -> str:
    raw_text = (project_text or "").strip()
    payload_json = json.dumps(valuation_payload or {}, ensure_ascii=False, indent=2)
    clipped_text = raw_text[:12000]
    prompt = f"""
Ты независимый senior VC analyst.
Твоя задача: провести АУДИТ качества входных данных и корректности расчета valuation.

Формат ответа: Markdown с разделами:
1. Input Audit
2. Calculation Audit
3. Red Flags in Valuation
4. What to Improve Next
5. Verdict (оценка качества расчета 0-100 и краткий вывод)

Правила:
- Не придумывай фактов, опирайся только на вход и JSON расчета.
- Отмечай конкретные несостыковки чисел, стадий, метода, года мультипликаторов.
- Если диапазон valuation выглядит завышенным/заниженным, объясни почему.
- Пиши по-русски, коротко и предметно.

Входной текст проекта:
\"\"\"
{clipped_text}
\"\"\"

Результат детерминированного расчета (JSON):
```json
{payload_json}
```
    """.strip()
    system = (
        "Ты аккуратный финансовый аудитор. "
        "Сначала проверяешь данные, затем математику, потом даешь улучшения."
    )
    return llm.generate(
        prompt,
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=2048,
    )
