from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_dir: Path
    supabase_url: str
    supabase_service_key: str
    gemini_api_key: str
    text_model: str
    text_fallback_model: str
    embedding_model: str
    embed_cache_version: str
    eval_instruction_path: Path
    embedding_cache_db_path: Path


def load_settings(app_dir: Path) -> Settings:
    load_dotenv(app_dir / ".env", override=True)
    return Settings(
        app_dir=app_dir,
        supabase_url=os.getenv("SUPABASE_URL", "").strip(),
        supabase_service_key=os.getenv("SUPABASE_SERVICE_KEY", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        text_model=os.getenv("GEMINI_TEXT_MODEL_PRIMARY", "models/gemini-2.5-flash-lite").strip(),
        text_fallback_model=os.getenv("GEMINI_TEXT_MODEL_FALLBACK_1", "models/gemini-2.5-flash").strip(),
        embedding_model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-2-preview").strip(),
        embed_cache_version=os.getenv("EMBED_CACHE_VERSION", "modular-v1").strip(),
        eval_instruction_path=app_dir / "evaluation_instruction.md",
        embedding_cache_db_path=app_dir / "embedding_cache.db",
    )
