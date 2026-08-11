"""
Review Ingestion & Embedding - schema setup.

Applies schema.sql (pgvector extension + restaurant_reviews table + FK
index) to the configured Postgres database. Safe to re-run: uses
`if not exists` throughout. The vector similarity index is created
separately by embed.py, after embeddings are populated (an HNSW index
builds more efficiently against data that already exists).
"""

from pathlib import Path

from app.storage.db import get_connection

SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


def main() -> None:
    sql = SCHEMA_SQL.read_text()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema applied: pgvector extension + restaurant_reviews table ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
