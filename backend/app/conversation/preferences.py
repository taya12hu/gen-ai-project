"""
Conversation & Personalization Storage.

CRUD helpers for durable, cross-session user_preferences (free-text
key/value facts like dietary=vegetarian, ambience=quiet). These are read
by the hybrid retrieval engine as soft defaults and written by the query
understanding step whenever it detects the user stating a preference in a
chat message.
"""

import logging

from psycopg2.extras import execute_values

from app.storage.db import get_connection

logger = logging.getLogger(__name__)


def get_preferences(user_id: int) -> dict[str, str]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select key, value from user_preferences where user_id = %s;", (user_id,))
            return dict(cur.fetchall())
    finally:
        conn.close()


def upsert_preferences(user_id: int, preferences: dict[str, str]) -> None:
    """Insert new preference facts or overwrite existing ones for the same key."""
    if not preferences:
        return

    rows = [(user_id, key, value) for key, value in preferences.items()]
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "insert into user_preferences (user_id, key, value) values %s "
                "on conflict (user_id, key) do update "
                "set value = excluded.value, updated_at = now();",
                rows,
            )
        conn.commit()
        logger.info("Upserted %d preference fact(s) for user_id=%s", len(rows), user_id)
    finally:
        conn.close()
