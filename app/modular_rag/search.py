from __future__ import annotations

import re
from typing import Any

from supabase import Client

from .llm import LLMClient


def infer_top_k_from_query(query: str, default_top_k: int = 6, max_top_k: int = 20) -> int:
    match = re.search(r"\b(?:top|топ)\s*(\d{1,2})\b", query.lower())
    if not match:
        return default_top_k
    return max(1, min(max_top_k, int(match.group(1))))


def deduplicate_results(results: list[dict], signature_len: int = 220) -> list[dict]:
    seen = set()
    unique = []
    for item in results or []:
        sig = (item.get("content") or "")[:signature_len]
        if sig not in seen:
            seen.add(sig)
            unique.append(item)
    return unique


def _result_score(row: dict) -> float:
    return float(row.get("hybrid_score", row.get("similarity", 0)) or 0.0)


def _result_key(row: dict, signature_len: int = 220) -> str:
    doc_id = row.get("id")
    if doc_id:
        return f"id:{doc_id}"
    content = (row.get("content") or "")[:signature_len]
    meta = row.get("metadata") or {}
    source_id = meta.get("source_id", "")
    chunk_index = meta.get("chunk_index", "")
    return f"sig:{source_id}:{chunk_index}:{content}"


def _merge_ranked_results(result_sets: list[list[dict]]) -> list[dict]:
    merged: dict[str, dict] = {}
    hits_per_key: dict[str, int] = {}

    for variant_index, rows in enumerate(result_sets):
        penalty = 0.015 * variant_index
        for row in rows:
            key = _result_key(row)
            candidate = dict(row)
            score = max(0.0, _result_score(candidate) - penalty)
            candidate["_merged_score"] = score
            hits_per_key[key] = hits_per_key.get(key, 0) + 1

            existing = merged.get(key)
            if existing is None or score > existing.get("_merged_score", -1.0):
                merged[key] = candidate

    merged_rows = list(merged.values())
    for row in merged_rows:
        key = _result_key(row)
        row["_merged_score"] = row.get("_merged_score", 0.0) + 0.01 * (hits_per_key.get(key, 1) - 1)
        if "hybrid_score" in row:
            row["hybrid_score"] = row["_merged_score"]
        else:
            row["similarity"] = row["_merged_score"]

    merged_rows.sort(key=lambda item: item.get("_merged_score", 0.0), reverse=True)
    for row in merged_rows:
        row.pop("_merged_score", None)
    return merged_rows


def _is_funding_query(query: str) -> bool:
    q = query.lower()
    markers = [
        "funding",
        "fundings",
        "raised",
        "investment",
        "investments",
        "финанс",
        "инвестиц",
        "раунд",
    ]
    return any(m in q for m in markers)


def _extract_funding_value(content: str) -> float:
    text = (content or "").lower()
    if not text:
        return -1.0

    # Try to capture a number close to funding-related keywords.
    keyword_match = re.search(
        r"(funding|raised|investment|инвестиц|финанс|раунд)[^0-9$€£₽]{0,40}([0-9][0-9.,\s]{0,20})([kmb]|thousand|million|billion|тыс|млн|млрд)?",
        text,
    )
    if not keyword_match:
        # fallback to any monetary-looking number
        keyword_match = re.search(
            r"([$€£₽]?\s*[0-9][0-9.,\s]{0,20})([kmb]|thousand|million|billion|тыс|млн|млрд)?",
            text,
        )
        if not keyword_match:
            return -1.0
        raw_num = keyword_match.group(1)
        raw_suffix = keyword_match.group(2) or ""
    else:
        raw_num = keyword_match.group(2)
        raw_suffix = keyword_match.group(3) or ""

    num = raw_num.replace(" ", "").replace(",", "")
    try:
        value = float(num)
    except ValueError:
        return -1.0

    suffix = raw_suffix.strip().lower()
    if suffix in {"k", "thousand", "тыс"}:
        value *= 1_000
    elif suffix in {"m", "million", "млн"}:
        value *= 1_000_000
    elif suffix in {"b", "billion", "млрд"}:
        value *= 1_000_000_000
    return value


def _funding_aware_rank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    if not _is_funding_query(query):
        return rows[:top_k]

    scored = []
    for r in rows:
        content = r.get("content") or ""
        funding = _extract_funding_value(content)
        sim = float(r.get("similarity", r.get("hybrid_score", 0)) or 0)
        scored.append((r, funding, sim))

    with_funding = [x for x in scored if x[1] >= 0]
    without_funding = [x for x in scored if x[1] < 0]

    with_funding.sort(key=lambda x: (x[1], x[2]), reverse=True)
    without_funding.sort(key=lambda x: x[2], reverse=True)

    # Strict behavior for funding queries:
    # return only entries with explicit funding values.
    if with_funding:
        return [x[0] for x in with_funding[:top_k]]

    # If nothing has funding at all, fall back to similarity results.
    return [x[0] for x in without_funding[:top_k]]


def _lexical_score(query: str, content: str) -> float:
    q_tokens = {w for w in re.findall(r"[a-zа-яё0-9]{3,}", (query or "").lower())}
    if not q_tokens:
        return 0.0
    c = (content or "").lower()
    hits = sum(1 for t in q_tokens if t in c)
    return hits / max(1, len(q_tokens))


class SearchService:
    def __init__(self, supabase: Client, llm: LLMClient):
        self.supabase = supabase
        self.llm = llm

    def _hybrid_search(
        self,
        *,
        q_emb: list[float],
        query_text: str,
        match_count: int,
        filter_obj: dict[str, Any],
    ) -> list[dict]:
        response = self.supabase.rpc(
            "hybrid_search",
            {
                "query_embedding": q_emb,
                "query_text": query_text,
                "match_count": match_count,
                "filter": filter_obj,
                "vector_weight": 0.75,
                "text_weight": 0.25,
            },
        ).execute()
        return response.data or []

    def _vector_search(
        self,
        *,
        q_emb: list[float],
        match_count: int,
        filter_obj: dict[str, Any],
    ) -> list[dict]:
        response = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": q_emb,
                "match_count": match_count,
                "filter": filter_obj,
            },
        ).execute()
        return response.data or []

    def _lexical_fallback_search(
        self,
        *,
        query: str,
        match_count: int,
        filter_obj: dict[str, Any],
    ) -> list[dict]:
        # Cheap fallback for local DB timeouts in vector RPCs. It intentionally
        # avoids pgvector functions and ranks a limited candidate set in Python.
        q = self.supabase.table("documents").select("id,content,metadata")
        source_id = (filter_obj or {}).get("source_id")
        if source_id:
            q = q.eq("metadata->>source_id", source_id)
        res = q.limit(1200).execute()
        rows = res.data or []
        scored: list[dict] = []
        for row in rows:
            score = _lexical_score(query, row.get("content") or "")
            if score <= 0:
                continue
            item = dict(row)
            item["similarity"] = score
            scored.append(item)
        scored.sort(key=lambda r: float(r.get("similarity") or 0), reverse=True)
        return scored[:match_count]

    def search(
        self,
        *,
        query: str,
        filter_obj: dict[str, Any] | None = None,
        desired_top_k: int | None = None,
    ) -> tuple[list[dict], str]:
        top_k = desired_top_k or infer_top_k_from_query(query)
        match_count = max(20, min(80, top_k * 4))
        filt = filter_obj or {}
        query_variants = self.llm.build_crosslingual_search_queries(query)
        result_sets: list[list[dict]] = []
        used_hybrid = True
        used_lexical_fallback = False

        for variant in query_variants:
            q_emb = self.llm.embed(variant, "retrieval_query")
            try:
                rows = self._hybrid_search(
                    q_emb=q_emb,
                    query_text=variant,
                    match_count=match_count,
                    filter_obj=filt,
                )
            except Exception:
                used_hybrid = False
                try:
                    rows = self._vector_search(
                        q_emb=q_emb,
                        match_count=match_count,
                        filter_obj=filt,
                    )
                except Exception:
                    used_lexical_fallback = True
                    rows = self._lexical_fallback_search(
                        query=variant,
                        match_count=match_count,
                        filter_obj=filt,
                    )
            result_sets.append(rows)

        rows = _merge_ranked_results(result_sets)
        rows = deduplicate_results(rows)
        rows = _funding_aware_rank(query, rows, top_k=top_k)
        if used_lexical_fallback:
            mode_prefix = "lexical_fallback"
        else:
            mode_prefix = "hybrid_search" if used_hybrid else "match_documents"
        mode = (
            f"{mode_prefix}+crosslingual[{len(query_variants)}]"
            if len(query_variants) > 1
            else mode_prefix
        )
        return rows, mode
