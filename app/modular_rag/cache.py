from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


class EmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
              text_hash TEXT PRIMARY KEY,
              task_type TEXT NOT NULL,
              embedding TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def get(self, text_hash: str) -> Optional[list[float]]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def set(self, text_hash: str, task_type: str, embedding: list[float]):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, task_type, embedding) VALUES (?, ?, ?)",
            (text_hash, task_type, json.dumps(embedding)),
        )
        conn.commit()
        conn.close()

