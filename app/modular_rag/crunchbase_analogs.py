from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CRUNCHBASE_CANDIDATE_PATHS = [
    Path(os.getenv("VALUATION_DATA_CSV", "data/valuation_enriched.csv")),
    ROOT / "crunchbase_merged_enriched_v6_historical_priors_no_money.csv",
    ROOT / "ml_benchmark_holdout_enriched_v6.csv",
]

TEXT_COLS = [
    "Organization Name",
    "Organization Description",
    "Organization Industries",
    "primary_industry_derived",
    "model_text_all_derived",
    "Investor Names",
    "Lead Investors",
    "Country",
    "Region",
    "City",
    "Funding Type",
]

OUTPUT_COLS = [
    "Organization Name",
    "Funding Type",
    "Announced Date",
    "Pre-Money Valuation",
    "Pre-Money Valuation Currency",
    "Money Raised",
    "Money Raised Currency",
    "Total Funding Amount",
    "Total Funding Amount Currency",
    "Organization Industries",
    "Organization Description",
    "Country",
    "Region",
    "City",
    "Investor Names",
    "Lead Investors",
    "oldcb_category_prior_median_raised_usd",
    "oldcb_stage_prior_median_raised_usd",
    "oldcb_category_stage_prior_median_raised_usd",
]

STAGE_ALIASES = {
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
    "private equity": "Private Equity",
}

INDUSTRY_HINTS = {
    "Artificial Intelligence (AI)": ["ai", "artificial intelligence", "machine learning", "llm", "generative", "нейросет"],
    "Finance": ["fintech", "finance", "financial", "banking", "payments", "bank", "кредит", "банк"],
    "Cyber Security": ["cyber", "security", "fraud", "кибер", "безопасн"],
    "Health Care": ["health", "medical", "clinical", "patient", "healthcare", "медицин"],
    "E-Commerce": ["ecommerce", "e-commerce", "retail", "marketplace", "commerce"],
    "Education": ["education", "edtech", "learning", "school", "университет", "обуч"],
    "Logistics": ["logistics", "delivery", "supply chain", "доставк", "логист"],
    "Real Estate": ["real estate", "proptech", "property", "rental", "недвиж"],
    "Apps": ["mobile app", "apps", "consumer app", "прилож"],
    "Other": [],
}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def _clean_text(value: Any) -> str:
    text = _safe_str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9а-яё\s,+#.&/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "their",
        "company",
        "startup",
        "platform",
        "solution",
        "product",
        "для",
        "или",
        "это",
        "как",
        "что",
        "при",
        "его",
        "она",
        "они",
        "компания",
        "стартап",
    }
    return [t for t in re.findall(r"[a-zа-яё0-9+#.]{3,}", _clean_text(text)) if t not in stop]


def _normalise_stage(value: Any) -> str:
    text = _clean_text(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return STAGE_ALIASES.get(text, _safe_str(value))


def _infer_stage(query: str) -> str:
    low = _clean_text(query)
    for alias, stage in STAGE_ALIASES.items():
        if alias in low or alias.replace(" ", "-") in low:
            return stage
    if any(k in low for k in ["mvp", "early revenue", "ранняя выруч"]):
        return "Seed"
    return ""


def _infer_industries(query: str) -> list[str]:
    low = _clean_text(query)
    matches = []
    for label, needles in INDUSTRY_HINTS.items():
        if needles and any(n in low for n in needles):
            matches.append(label)
    return matches


def _infer_geo(query: str) -> tuple[str, str]:
    low = _clean_text(query)
    countries = {
        "United States": ["united states", "usa", "us market", "america", "сша"],
        "Russia": ["russia", "russian", "россия", "россий"],
        "India": ["india", "индия"],
        "United Kingdom": ["united kingdom", "uk", "london", "британ"],
        "Germany": ["germany", "berlin", "герман"],
        "France": ["france", "paris", "франц"],
    }
    for country, needles in countries.items():
        if any(n in low for n in needles):
            return country, ""
    if any(n in low for n in ["europe", "европа", "европей"]):
        return "", "Europe"
    if any(n in low for n in ["asia", "ази"]):
        return "", "Asia"
    return "", ""


def _first_existing_path() -> Path | None:
    for path in CRUNCHBASE_CANDIDATE_PATHS:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_crunchbase() -> tuple[pd.DataFrame, str]:
    path = _first_existing_path()
    if path is None:
        return pd.DataFrame(), ""
    df = pd.read_csv(path, low_memory=False)
    if "Organization Name" not in df.columns:
        return pd.DataFrame(), str(path)

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    text_parts = []
    for col in TEXT_COLS:
        text_parts.append(df[col].fillna("").astype(str))
    df["_analog_text"] = text_parts[0]
    for part in text_parts[1:]:
        df["_analog_text"] = df["_analog_text"] + " " + part
    df["_analog_text_clean"] = df["_analog_text"].map(_clean_text)
    df["_stage_norm"] = df["Funding Type"].map(_normalise_stage) if "Funding Type" in df.columns else ""
    df["_valuation_usd"] = pd.to_numeric(df.get("Pre-Money Valuation"), errors="coerce")
    if "Pre-Money Valuation Currency" in df.columns:
        df.loc[~df["Pre-Money Valuation Currency"].fillna("").astype(str).eq("USD"), "_valuation_usd"] = pd.NA
    return df, str(path)


def _bm25_like_score(query_tokens: list[str], content: str) -> float:
    if not query_tokens or not content:
        return 0.0
    content_tokens = _tokens(content)
    if not content_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in content_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    content_len = len(content_tokens)
    for token in set(query_tokens):
        tf = counts.get(token, 0)
        if tf:
            score += (tf * 2.2) / (tf + 1.2 + 0.2 * content_len / 120.0)
        else:
            # substring hits help with terms like fintech/payments or Russian stems.
            if token in content:
                score += 0.35
    return score / max(1.0, len(set(query_tokens)) ** 0.5)


def _money(value: Any) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num) or float(num) <= 0:
        return ""
    num = float(num)
    if num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"${num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"${num / 1_000:.0f}K"
    return f"${num:.0f}"


def _format_row_content(row: pd.Series, score: float, reasons: list[str]) -> str:
    name = _safe_str(row.get("Organization Name"))
    stage = _safe_str(row.get("Funding Type"))
    date = _safe_str(row.get("Announced Date"))
    industries = _safe_str(row.get("Organization Industries") or row.get("primary_industry_derived"))
    country = _safe_str(row.get("Country"))
    region = _safe_str(row.get("Region"))
    valuation = _money(row.get("Pre-Money Valuation"))
    raised = _money(row.get("Money Raised"))
    total = _money(row.get("Total Funding Amount"))
    investors = _safe_str(row.get("Investor Names"))
    desc = _safe_str(row.get("Organization Description")) or _safe_str(row.get("model_text_all_derived"))
    priors = []
    for label, col in [
        ("category median raised", "oldcb_category_prior_median_raised_usd"),
        ("stage median raised", "oldcb_stage_prior_median_raised_usd"),
        ("category+stage median raised", "oldcb_category_stage_prior_median_raised_usd"),
    ]:
        val = _money(row.get(col))
        if val:
            priors.append(f"{label}: {val}")
    facts = [
        f"{name} | {stage} | {date}".strip(" |"),
        f"Industry: {industries}" if industries else "",
        f"Geo: {country}, {region}".strip(" ,") if country or region else "",
        f"Pre-money valuation: {valuation}" if valuation else "",
        f"Money raised: {raised}" if raised else "",
        f"Total funding: {total}" if total else "",
        f"Investors: {investors[:220]}" if investors else "",
        f"Priors: {'; '.join(priors)}" if priors else "",
        f"Why similar: {', '.join(reasons)}; structured_score={score:.3f}" if reasons else f"structured_score={score:.3f}",
        f"Description: {desc[:520]}" if desc else "",
    ]
    return "\n".join([f for f in facts if f])


class CrunchbaseAnalogRetriever:
    """Structured + lexical analog search over local Crunchbase CSV.

    This intentionally does not create embeddings. It combines stage, industry,
    geography, valuation priors and BM25-like token matching.
    """

    def __init__(self, path: Path | None = None):
        self.path_override = path

    def search(self, query: str, top_k: int = 10) -> tuple[list[dict[str, Any]], str]:
        if self.path_override:
            df = pd.read_csv(self.path_override, low_memory=False)
            source_path = str(self.path_override)
            if "_analog_text_clean" not in df.columns:
                base, _ = _load_crunchbase()
                if str(self.path_override) == _first_existing_path():
                    df = base
        else:
            df, source_path = _load_crunchbase()
        if df.empty:
            return [], "crunchbase_unavailable"

        query_tokens = _tokens(query)
        inferred_stage = _infer_stage(query)
        inferred_industries = _infer_industries(query)
        country, region = _infer_geo(query)

        work = df.copy()
        has_text = work["_analog_text_clean"].str.len().gt(40)
        has_value = pd.to_numeric(work.get("_valuation_usd"), errors="coerce").notna()
        base_mask = has_text & (has_value | work["Funding Type"].notna())

        focused_mask = pd.Series(False, index=work.index)
        if inferred_stage:
            focused_mask |= work["_stage_norm"].eq(inferred_stage)
        for industry in inferred_industries:
            industry_tokens = _tokens(industry)
            for token in industry_tokens:
                focused_mask |= work["_analog_text_clean"].str.contains(re.escape(token), na=False)
        if country:
            focused_mask |= work["Country"].fillna("").astype(str).str.lower().eq(country.lower())
        if region:
            focused_mask |= work["Region"].fillna("").astype(str).str.lower().eq(region.lower())

        if focused_mask.any():
            work = work[base_mask & focused_mask].copy()
            # Include some high-signal rows outside the focus as a recall cushion.
            if len(work) < 800:
                cushion = df[base_mask & ~focused_mask].head(1200)
                work = pd.concat([work, cushion], ignore_index=False)
        else:
            work = work[base_mask].copy()

        # Keep the scan bounded while preserving useful late-stage and high-signal rows.
        if len(work) > 8000:
            work = work.head(8000).copy()

        scored: list[tuple[float, pd.Series, list[str]]] = []
        for _, row in work.iterrows():
            content = _safe_str(row.get("_analog_text_clean") or row.get("model_text_all_derived") or row.get("Organization Description"))
            text_score = _bm25_like_score(query_tokens, content)
            score = text_score
            reasons = []

            row_stage = _safe_str(row.get("_stage_norm") or row.get("Funding Type"))
            if inferred_stage and row_stage == inferred_stage:
                score += 1.3
                reasons.append(f"same stage {row_stage}")
            elif inferred_stage and row_stage:
                # Adjacent stage is still useful, especially for sparse early-stage cases.
                order = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Series F", "Series G"]
                if inferred_stage in order and row_stage in order and abs(order.index(inferred_stage) - order.index(row_stage)) == 1:
                    score += 0.45
                    reasons.append(f"adjacent stage {row_stage}")

            industry_blob = _clean_text(f"{row.get('Organization Industries')} {row.get('primary_industry_derived')}")
            for industry in inferred_industries:
                if _clean_text(industry) in industry_blob:
                    score += 1.1
                    reasons.append(f"same sector {industry}")
                elif any(t in industry_blob for t in _tokens(industry)):
                    score += 0.45
                    reasons.append(f"related sector {industry}")

            if country and _safe_str(row.get("Country")).lower() == country.lower():
                score += 0.55
                reasons.append(f"same country {country}")
            elif region and _safe_str(row.get("Region")).lower() == region.lower():
                score += 0.35
                reasons.append(f"same region {region}")

            if pd.to_numeric(pd.Series([row.get("Pre-Money Valuation")]), errors="coerce").notna().iloc[0]:
                score += 0.25
                reasons.append("has historical valuation")
            if pd.to_numeric(pd.Series([row.get("oldcb_category_stage_prior_median_raised_usd")]), errors="coerce").fillna(0).iloc[0] > 0:
                score += 0.15
                reasons.append("has category+stage prior")
            if int(pd.to_numeric(pd.Series([row.get("has_any_top_investor")]), errors="coerce").fillna(0).iloc[0] or 0) > 0:
                score += 0.12
                reasons.append("top investor signal")

            if score > 0.35:
                scored.append((score, row, reasons or ["lexical match"]))

        scored.sort(key=lambda item: item[0], reverse=True)
        rows: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for score, row, reasons in scored:
            name = _safe_str(row.get("Organization Name")).lower()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            content = _format_row_content(row, score, reasons)
            rows.append(
                {
                    "id": f"crunchbase:{name}",
                    "content": content,
                    "similarity": float(score),
                    "metadata": {
                        "source_id": "local_crunchbase_structured",
                        "source_path": source_path,
                        "organization_name": row.get("Organization Name"),
                        "funding_type": row.get("Funding Type"),
                        "country": row.get("Country"),
                        "region": row.get("Region"),
                        "pre_money_valuation": row.get("Pre-Money Valuation"),
                        "retrieval_method": "structured_lexical_no_embeddings",
                    },
                }
            )
            if len(rows) >= top_k:
                break

        mode = "crunchbase_structured_lexical_no_embeddings"
        if inferred_stage:
            mode += f"+stage:{inferred_stage}"
        if inferred_industries:
            mode += f"+industry:{','.join(inferred_industries[:3])}"
        return rows, mode
