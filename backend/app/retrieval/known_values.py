"""
Retrieval - known place/cuisine values.

Fetches the set of place and cuisine values that actually exist in the
restaurants table. Used by query understanding to snap free-text place/
cuisine mentions in a chat message to canonical DB casing (see
app.query_understanding).
"""

from app.storage.db import get_connection


def get_known_places() -> set[str]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select distinct place from restaurants;")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def get_known_cuisines() -> set[str]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select distinct unnest(cuisines) from restaurants;")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
