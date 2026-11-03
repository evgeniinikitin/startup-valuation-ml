from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Optional

import google.generativeai as genai
import numpy as np
from google.generativeai import types as genai_types

from .cache import EmbeddingCache
from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings, cache: EmbeddingCache):
        self.settings = settings
        self.cache = cache
        self._query_translation_cache: dict[str, list[str]] = {}
        genai.configure(api_key=settings.gemini_api_key)

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        arr = np.array(vec, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vec
        return (arr / norm).tolist()

    def _hash_text(self, text: str, task_type: str) -> str:
        payload = (
            f"{self.settings.embed_cache_version}:{self.settings.embedding_model}:{task_type}:{(text or '').strip()}"
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def embed(self, text: str, task_type: str) -> list[float]:
        key = self._hash_text(text, task_type)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        result = genai.embed_content(
            model=self.settings.embedding_model,
            content=text,
            task_type=task_type,
            request_options={"timeout": 30},
        )
        emb = self._normalize(result["embedding"])
        self.cache.set(key, task_type, emb)
        return emb

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 3072,
    ) -> str:
        models = [self.settings.text_model, self.settings.text_fallback_model]
        last_error: Optional[Exception] = None

        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
                response = model.generate_content(
                    prompt,
                    generation_config=genai_types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                    request_options={"timeout": 120},
                )
                text = (response.text or "").strip()
                if text:
                    return text
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"Generation failed on all text models. Last error: {last_error}")

    def build_crosslingual_search_queries(self, query: str) -> list[str]:
        clean = (query or "").strip()
        if not clean:
            return []
        qkey = clean.lower()
        cached = self._query_translation_cache.get(qkey)
        if cached:
            return list(cached)

        # Long retrieval seeds are already handled reasonably well by embeddings.
        # We expand only compact user-style queries where BM25/hybrid search benefits most.
        if len(clean) > 240 or len(clean.split()) > 24:
            self._query_translation_cache[qkey] = [clean]
            return [clean]

        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", clean))
        has_latin = bool(re.search(r"[A-Za-z]", clean))
        if has_cyrillic == has_latin:
            self._query_translation_cache[qkey] = [clean]
            return [clean]

        target_language = "English" if has_cyrillic else "Russian"
        prompt = f"""
Return strict JSON only.
Translate the search query into {target_language} for cross-lingual retrieval.
Rules:
- Preserve company names, product names, abbreviations, numbers, currencies, and dates.
- Do not explain anything.
- Do not paraphrase broadly.
- Return this schema only:
{{"queries":["original query", "translated query"]}}

Query:
{clean}
        """.strip()

        try:
            raw = self.generate(prompt, temperature=0.0, max_output_tokens=220)
            data = json.loads(raw)
            queries = data.get("queries", []) if isinstance(data, dict) else []
        except Exception:
            self._query_translation_cache[qkey] = [clean]
            return [clean]

        unique: list[str] = []
        seen = set()
        for item in [clean, *queries]:
            value = str(item or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(value)
        out = unique or [clean]
        # Keep cache bounded.
        if len(self._query_translation_cache) > 500:
            self._query_translation_cache.clear()
        self._query_translation_cache[qkey] = out
        return out

    def extract_pdf_text_gemini(self, pdf_bytes: bytes, prompt: str) -> str:
        tmp_path = ""
        uploaded = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            uploaded = genai.upload_file(tmp_path)

            # Use explicit model call with file input for robust OCR/document parsing.
            model = genai.GenerativeModel(self.settings.text_model)
            response = model.generate_content(
                [uploaded, prompt],
                generation_config=genai_types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
                request_options={"timeout": 180},
            )
            return (response.text or "").encode("utf-8", "replace").decode("utf-8")
        finally:
            if uploaded is not None:
                try:
                    genai.delete_file(uploaded.name)
                except Exception:
                    pass
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
