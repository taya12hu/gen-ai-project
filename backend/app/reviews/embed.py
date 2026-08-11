"""
Review Ingestion & Embedding (step 2: embed).

Fills the embedding column for every restaurant_reviews row ingested by
ingest.py, using a local sentence-transformers model (no API key, no
per-call cost/rate limit - chosen over a hosted embeddings API given the
volume of reviews). Runs in chunks so memory stays bounded regardless of
how many reviews there are. The HNSW similarity index is created at the
end, once there's data to build it against (cheaper and better-quality
than building it against an empty table and maintaining it incrementally
during ingest).

Writes go through a staging temp table loaded via COPY, then a single
UPDATE...FROM join, rather than a parameterized "UPDATE ... FROM (VALUES
...)" statement: with 384-dim vectors, a few hundred rows of literal SQL
text is a multi-megabyte statement that the server has to parse - COPY's
streaming protocol avoids that parsing cost entirely and is an order of
magnitude faster for this volume (tens of thousands of reviews).
"""

import io
import logging
import time

import psycopg2

from app.reviews.embedding_model import EMBEDDING_DIM, get_embedder
from app.storage.db import get_connection

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2000
MAX_RETRIES = 5

STAGING_TABLE_SQL = "create temporary table if not exists embedding_staging (id bigint primary key, embedding vector(384));"


def fetch_unembedded_ids(conn) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("select id from restaurant_reviews where embedding is null order by id;")
        return [row[0] for row in cur.fetchall()]


def fetch_texts(conn, ids: list[int]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "select id, review_text from restaurant_reviews where id = any(%s) order by id;",
            (ids,),
        )
        by_id = dict(cur.fetchall())
    return [by_id[i] for i in ids]


def write_chunk(conn, chunk_ids: list[int], embeddings) -> None:
    buffer = io.StringIO()
    for review_id, embedding in zip(chunk_ids, embeddings):
        vector_text = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
        buffer.write(f"{review_id}\t{vector_text}\n")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.execute("truncate embedding_staging;")
        cur.copy_expert("copy embedding_staging (id, embedding) from stdin", buffer)
        cur.execute(
            "update restaurant_reviews as r set embedding = s.embedding "
            "from embedding_staging as s where r.id = s.id;"
        )
    conn.commit()


def main() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(STAGING_TABLE_SQL)
        conn.commit()

        pending_ids = fetch_unembedded_ids(conn)
        print(f"{len(pending_ids)} reviews need embeddings.")

        if pending_ids:
            model = get_embedder()
            assert model.get_embedding_dimension() == EMBEDDING_DIM

            done = 0
            for i in range(0, len(pending_ids), CHUNK_SIZE):
                chunk_ids = pending_ids[i : i + CHUNK_SIZE]

                # The Supabase pooler occasionally drops idle/long-lived connections
                # mid-run; each chunk is independently committed, so on a dropped
                # connection we just reconnect (and recreate the temp staging table,
                # which is connection-scoped) and retry that one chunk rather than
                # losing all progress made so far.
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        texts = fetch_texts(conn, chunk_ids)
                        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
                        write_chunk(conn, chunk_ids, embeddings)
                        break
                    except psycopg2.OperationalError as exc:
                        if attempt == MAX_RETRIES:
                            raise
                        logger.warning("Connection error (%s); reconnecting (attempt %d/%d)", exc, attempt, MAX_RETRIES)
                        print(f"Connection error ({exc}); reconnecting (attempt {attempt}/{MAX_RETRIES})...")
                        try:
                            conn.close()
                        except Exception:
                            pass
                        time.sleep(2 * attempt)
                        conn = get_connection()
                        with conn.cursor() as cur:
                            cur.execute(STAGING_TABLE_SQL)
                        conn.commit()

                done += len(chunk_ids)
                print(f"Embedded {done}/{len(pending_ids)}")

        with conn.cursor() as cur:
            # Building an HNSW index over ~200k+ 384-dim vectors comfortably exceeds
            # the pooler's default statement_timeout, so it needs its own generous
            # timeout. Parallel index builds need dynamic shared-memory segments for
            # each worker, which overflowed this Supabase instance's /dev/shm - forcing
            # a single-worker (non-parallel) build avoids that entirely, at the cost of
            # a slower but reliable build.
            cur.execute("set statement_timeout = '30min';")
            cur.execute("set max_parallel_maintenance_workers = 0;")
            cur.execute(
                "create index if not exists restaurant_reviews_embedding_idx "
                "on restaurant_reviews using hnsw (embedding vector_cosine_ops);"
            )
        conn.commit()
        print("HNSW similarity index ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
