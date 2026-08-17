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


def list_preferences(user_id: int) -> list[dict]:
    """All stored preference facts for a user, most recently updated first -
    for a user-facing "what do you remember about me" view, unlike
    get_preferences() which returns a plain key->value dict for retrieval."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select key, value, updated_at from user_preferences "
                "where user_id = %s order by updated_at desc;",
                (user_id,),
            )
            rows = cur.fetchall()
        return [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]
    finally:
        conn.close()


def delete_preference(user_id: int, key: str) -> bool:
    """Forgets one stored preference fact. Returns False if there was nothing
    to delete, so the caller can turn that into a 404 rather than a silent
    no-op."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from user_preferences where user_id = %s and key = %s;", (user_id, key))
            deleted = cur.rowcount > 0
        conn.commit()
        if deleted:
            logger.info("Deleted preference key=%s for user_id=%s", key, user_id)
        return deleted
    finally:
        conn.close()
