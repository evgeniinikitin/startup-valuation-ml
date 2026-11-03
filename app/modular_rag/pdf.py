from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

from .llm import LLMClient


@dataclass
class PdfExtractionResult:
    text: str
    method: str
    page_count: int
    failed_pages: int
    quality_score: float


def _pypdf_extract(pdf_bytes: bytes) -> tuple[str, int, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    failed_pages = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            parts.append(txt)
        except Exception:
            failed_pages += 1
            parts.append("")
    text = "\n".join(parts).encode("utf-8", "replace").decode("utf-8")
    return text, len(reader.pages), failed_pages


def _text_quality_score(text: str, page_count: int, failed_pages: int) -> float:
    clean = (text or "").strip()
    if not clean:
        return 0.0

    chars = len(clean)
    words = len(re.findall(r"\b\w+\b", clean))
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", clean))
    alnum_ratio = sum(ch.isalnum() for ch in clean) / max(chars, 1)
    chars_per_page = chars / max(page_count, 1)

    score = 0.0
    score += min(1.0, chars_per_page / 1200) * 0.35
    score += min(1.0, words / 1200) * 0.25
    score += min(1.0, letters / 5000) * 0.20
    score += min(1.0, alnum_ratio / 0.75) * 0.20

    if failed_pages >= max(1, page_count // 5):
        score *= 0.5
    return max(0.0, min(score, 1.0))


def _balanced_excerpt(text: str, max_chars: int = 70000) -> str:
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean
    part = max_chars // 3
    mid_start = max(0, len(clean) // 2 - part // 2)
    mid_end = min(len(clean), mid_start + part)
    return (
        clean[:part]
        + "\n\n[... фрагмент пропущен ...]\n\n"
        + clean[mid_start:mid_end]
        + "\n\n[... фрагмент пропущен ...]\n\n"
        + clean[-part:]
    )


def _looks_like_meta_or_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    markers = [
        "не имею доступа к файловой системе",
        "не могу напрямую работать с файлами",
        "предоставьте текст",
        "предоставленный вами текст не содержит информации",
        "паспорт проекта",
        "не найдено в pdf",
    ]
    return any(m in t for m in markers)


def extract_pdf_text(
    pdf_bytes: bytes,
    llm: LLMClient,
    *,
    mode: str = "gemini_first",
    min_quality: float = 0.45,
) -> PdfExtractionResult:
    pypdf_text, page_count, failed_pages = _pypdf_extract(pdf_bytes)
    quality = _text_quality_score(pypdf_text, page_count, failed_pages)

    ocr_prompt = (
        "Считай текст со всех страниц PDF максимально полно. "
        "Не сокращай, не придумывай. Сохрани цифры, таблицы и подписи. "
        "Если есть таблицы, верни их в markdown."
    )

    if mode == "pypdf_only":
        return PdfExtractionResult(pypdf_text, "pypdf", page_count, failed_pages, quality)

    if mode == "gemini_first":
        gemini_text = llm.extract_pdf_text_gemini(pdf_bytes, ocr_prompt)
        gemini_clean = gemini_text.strip()
        if gemini_clean and not _looks_like_meta_or_refusal(gemini_text):
            if quality >= min_quality and len(gemini_clean) < len((pypdf_text or "").strip()) * 0.35:
                return PdfExtractionResult(pypdf_text, "pypdf_preferred_over_short_gemini", page_count, failed_pages, quality)
            return PdfExtractionResult(gemini_text, "gemini", page_count, failed_pages, quality)
        return PdfExtractionResult(pypdf_text, "pypdf_fallback", page_count, failed_pages, quality)

    # auto mode: pypdf first, Gemini only when quality is low.
    if quality >= min_quality:
        return PdfExtractionResult(pypdf_text, "pypdf", page_count, failed_pages, quality)

    gemini_text = llm.extract_pdf_text_gemini(pdf_bytes, ocr_prompt)
    if gemini_text.strip() and not _looks_like_meta_or_refusal(gemini_text):
        return PdfExtractionResult(gemini_text, "gemini_fallback", page_count, failed_pages, quality)
    return PdfExtractionResult(pypdf_text, "pypdf_low_quality_fallback", page_count, failed_pages, quality)


def summary_source_excerpt(text: str) -> str:
    return _balanced_excerpt(text, max_chars=50000)


def retrieval_source_excerpt(text: str) -> str:
    return _balanced_excerpt(text, max_chars=14000)


def report_source_excerpt(text: str) -> str:
    return _balanced_excerpt(text, max_chars=85000)
