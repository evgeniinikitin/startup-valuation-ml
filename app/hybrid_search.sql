-- =============================================================
-- Hybrid Search Migration: Add full-text search (BM25) support
-- Run this AFTER database_setup.sql
-- =============================================================

-- 1. Add a tsvector column for full-text search
ALTER TABLE documents ADD COLUMN IF NOT EXISTS fts tsvector;

-- 2. Populate existing rows
UPDATE documents SET fts = to_tsvector('english', coalesce(content, ''));

-- 3. Auto-populate on INSERT/UPDATE
CREATE OR REPLACE FUNCTION documents_fts_trigger() RETURNS trigger AS $$
BEGIN
  NEW.fts := to_tsvector('english', coalesce(NEW.content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_fts_update ON documents;
CREATE TRIGGER documents_fts_update 
  BEFORE INSERT OR UPDATE ON documents
  FOR EACH ROW EXECUTE FUNCTION documents_fts_trigger();

-- 4. GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING GIN (fts);

-- 5. Hybrid search function: combines vector similarity with BM25 text relevance
CREATE OR REPLACE FUNCTION hybrid_search (
  query_embedding vector(3072),
  query_text text DEFAULT '',
  match_count int DEFAULT 15,
  filter jsonb DEFAULT '{}',
  vector_weight float DEFAULT 0.7,
  text_weight float DEFAULT 0.3
) RETURNS TABLE (
  id uuid,
  content text,
  metadata jsonb,
  similarity float,
  text_rank float,
  hybrid_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    (1 - (d.embedding <=> query_embedding))::float AS similarity,
    CASE 
      WHEN query_text = '' THEN 0.0
      ELSE ts_rank(d.fts, plainto_tsquery('english', query_text))::float
    END AS text_rank,
    (
      vector_weight * (1 - (d.embedding <=> query_embedding)) +
      CASE 
        WHEN query_text = '' THEN 0.0
        ELSE text_weight * ts_rank(d.fts, plainto_tsquery('english', query_text))
      END
    )::float AS hybrid_score
  FROM documents d
  WHERE d.metadata @> filter
  ORDER BY hybrid_score DESC
  LIMIT match_count;
END;
$$;
