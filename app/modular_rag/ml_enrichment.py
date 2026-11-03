from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ENRICHED_CSV_PATH = Path(os.getenv("VALUATION_DATA_CSV", "data/valuation_enriched.csv"))


def _norm_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9а-яё]+", " ", text)
    text = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|company|co|plc|gmbh|sarl|sas|bv|oy|ab)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _json_from_llm(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _stage_from_payload(stage: Any) -> str:
    mapping = {
        "idea_only": "Pre-Seed",
        "mvp_no_revenue": "Seed",
        "early_revenue": "Seed",
        "growth": "Series A",
    }
    return mapping.get(str(stage or ""), str(stage or "Venture - Series Unknown"))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def _load_enriched_index() -> pd.DataFrame:
    if not ENRICHED_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ENRICHED_CSV_PATH, low_memory=False)
    if "Organization Name" not in df.columns:
        return pd.DataFrame()
    df["_match_name"] = df["Organization Name"].map(_norm_name)
    return df


def extract_enrichment_facts(project_text: str, payload: dict[str, Any], llm: Any | None = None) -> dict[str, Any]:
    """Extract company facts for ML feature enrichment.

    This is deliberately conservative: it returns fields only when the text
    appears to describe the target startup, not market examples.
    """
    defaults = {
        "company_name": "",
        "funding_type": _stage_from_payload(payload.get("stage")),
        "country": payload.get("country_hint") or "",
        "region": "",
        "state": "",
        "city": "",
        "industries": payload.get("industry_hint") or "",
        "description": (project_text or "")[:1200],
        "investor_names": "",
        "lead_investors": "",
        "announced_date": "",
        "founded_year": "",
    }
    if llm is None:
        return defaults

    prompt = f"""
Return strict JSON only.
Extract facts about the TARGET startup from the text below.
Do not use competitors, market examples, generic benchmark companies, or investor portfolio examples as the target.
If a field is not explicitly present, return an empty string.

Schema:
{{
  "company_name": "",
  "funding_type": "Pre-Seed|Seed|Series A|Series B|Series C|Series D|Series E|Series F|Series G|Venture - Series Unknown",
  "country": "",
  "region": "",
  "state": "",
  "city": "",
  "industries": "",
  "description": "",
  "investor_names": "",
  "lead_investors": "",
  "announced_date": "",
  "founded_year": ""
}}

Known weak hints from deterministic parser:
- stage={payload.get("stage")}
- country_hint={payload.get("country_hint")}
- industry_hint={payload.get("industry_hint")}
- year_hint={payload.get("damodaran_year_used")}

Text:
{(project_text or "")[:9000]}
    """.strip()
    try:
        raw = llm.generate(prompt, temperature=0.0, max_output_tokens=900)
        data = _json_from_llm(raw)
    except Exception:
        data = {}

    out = dict(defaults)
    for key in out:
        value = data.get(key)
        if _safe_str(value):
            out[key] = _safe_str(value)
    return out


def _score_match(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return min(len(query), len(candidate)) / max(len(query), len(candidate))
    q_tokens = set(query.split())
    c_tokens = set(candidate.split())
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)


def find_enriched_company_row(company_name: str, funding_type: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    df = _load_enriched_index()
    query = _norm_name(company_name)
    if df.empty or not query:
        return None, {"source": "local_enriched_crunchbase", "matched": False, "reason": "missing_index_or_company_name"}

    exact = df[df["_match_name"].eq(query)].copy()
    candidates = exact if not exact.empty else df[df["_match_name"].str.contains(re.escape(query), na=False)].copy()
    if candidates.empty:
        sample = df[["_match_name"]].dropna().drop_duplicates().copy()
        sample["_score"] = sample["_match_name"].map(lambda name: _score_match(query, name))
        best_names = sample.sort_values("_score", ascending=False).head(8)
        candidates = df[df["_match_name"].isin(best_names["_match_name"])].copy()
        candidates["_score"] = candidates["_match_name"].map(lambda name: _score_match(query, name))
    else:
        candidates["_score"] = candidates["_match_name"].map(lambda name: _score_match(query, name))

    if funding_type and "Funding Type" in candidates.columns:
        ft = str(funding_type or "").strip().lower()
        candidates["_stage_bonus"] = candidates["Funding Type"].astype(str).str.lower().eq(ft).astype(float) * 0.08
    else:
        candidates["_stage_bonus"] = 0.0
    if "Announced Date" in candidates.columns:
        dates = pd.to_datetime(candidates["Announced Date"], errors="coerce")
        candidates["_date_rank"] = dates.rank(method="first", na_option="bottom")
    else:
        candidates["_date_rank"] = 0.0
    candidates["_final_score"] = candidates["_score"] + candidates["_stage_bonus"]
    best = candidates.sort_values(["_final_score", "_date_rank"], ascending=[False, False]).head(1)
    if best.empty:
        return None, {"source": "local_enriched_crunchbase", "matched": False, "reason": "no_candidates"}

    row = best.iloc[0].drop(labels=[c for c in ["_match_name", "_score", "_stage_bonus", "_date_rank", "_final_score"] if c in best.columns]).to_dict()
    score = float(best.iloc[0]["_final_score"])
    matched = score >= 0.72
    return (row if matched else None), {
        "source": "local_enriched_crunchbase",
        "matched": matched,
        "company_name_query": company_name,
        "matched_company_name": row.get("Organization Name"),
        "matched_funding_type": row.get("Funding Type"),
        "match_score": score,
        "index_path": str(ENRICHED_CSV_PATH),
    }


def build_ml_enrichment_row(project_text: str, payload: dict[str, Any], llm: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = extract_enrichment_facts(project_text, payload, llm=llm)
    row, match_info = find_enriched_company_row(facts.get("company_name") or "", facts.get("funding_type") or "")
    if row is None:
        row = {}

    # Overlay extracted current-input facts without touching target/money columns.
    aliases = {
        "Organization Name": facts.get("company_name"),
        "Funding Type": facts.get("funding_type"),
        "Country": facts.get("country"),
        "Region": facts.get("region"),
        "State": facts.get("state"),
        "City": facts.get("city"),
        "Organization Industries": facts.get("industries"),
        "Organization Description": facts.get("description"),
        "Investor Names": facts.get("investor_names"),
        "Lead Investors": facts.get("lead_investors"),
        "Announced Date": facts.get("announced_date"),
    }
    for key, value in aliases.items():
        if _safe_str(value):
            row[key] = _safe_str(value)
    if _safe_str(facts.get("founded_year")):
        row["oldcb_founded_year"] = _safe_str(facts.get("founded_year"))

    info = {
        "facts": facts,
        "match": match_info,
        "enrichment_mode": "local_crunchbase_match" if match_info.get("matched") else "llm_extracted_sparse_facts",
        "web_search_used": False,
        "web_search_note": "No external web-search provider is configured; enrichment used LLM extraction plus local enriched Crunchbase index.",
    }
    return row, info
