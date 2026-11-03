from __future__ import annotations

from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client

from .llm import LLMClient


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def ingest_document(
    *,
    supabase: Client,
    llm: LLMClient,
    text: str,
    filename: str,
    source_id: str | None = None,
) -> tuple[str, int]:
    sid = source_id or str(uuid4())
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = llm.embed(chunk, "retrieval_document")
        payload = {
            "content": chunk,
            "metadata": {
                "source_id": sid,
                "filename": filename,
                "chunk_index": i,
                "doc_type": "uploaded_pdf",
            },
            "embedding": embedding,
        }
        supabase.table("documents").insert(payload).execute()

    return sid, len(chunks)

