"""
Conversation & Personalization Storage.

CRUD helpers for durable, cross-session user_preferences (free-text
key/value facts like dietary=vegetarian, ambience=quiet). These are read
by the hybrid retrieval engine as soft defaults and written by the query
understanding step whenever it detects the user stating a preference in a
chat message.

The *key* is a closed vocabulary (PREFERENCE_KEYS); the *value* stays free
text. That asymmetry is deliberate: values are what the user said and can't
be enumerated ahead of time, but keys are what the retrieval layer switches
on, and an unknown key is not a harmless extra row - it's a fact the system
stores, shows the user in the preferences panel, and then silently never
applies to anything. Query understanding picks the key itself, so before
this vocabulary existed a model that answered "diet" instead of "dietary"
produced exactly that: a remembered preference that quietly did nothing.

normalize_preference_key is applied on the way in (so nothing unusable is
stored) and on the way out of get_preferences (so rows written before the
vocabulary existed still work rather than being dead weight).
"""

import logging

from psycopg2.extras import execute_values

from app.storage.db import get_connection

logger = logging.getLogger(__name__)

# The complete set of preference facts the system knows how to act on. Adding
# a key here is not enough on its own - app.chat.service decides how each one
# reaches retrieval.
PREFERENCE_KEYS: tuple[str, ...] = ("dietary", "ambience", "occasion", "vibe")

# Variants a model plausibly reaches for instead of the canonical key. Kept
# deliberately conservative: a wrong mapping silently files a fact under the
# wrong heading, which is worse than dropping it, so anything genuinely
# ambiguous is left out and simply rejected.
_KEY_ALIASES: dict[str, str] = {
    "diet": "dietary",
    "diets": "dietary",
    "dietary_preference": "dietary",
    "dietary_preferences": "dietary",
    "dietary_restriction": "dietary",
    "dietary_restrictions": "dietary",
    "food_restriction": "dietary",
    "food_restrictions": "dietary",
    "ambiance": "ambience",
    "atmosphere": "ambience",
    "occasions": "occasion",
    "vibes": "vibe",
}


def normalize_preference_key(key: str) -> str | None:
    """Canonical key for `key`, or None if it isn't one the system can act on."""
    candidate = key.strip().lower().replace(" ", "_").replace("-", "_")
    if candidate in PREFERENCE_KEYS:
        return candidate
    return _KEY_ALIASES.get(candidate)


def normalize_preferences(preferences: dict[str, str]) -> dict[str, str]:
    """Canonicalizes keys and drops anything outside the vocabulary.

    Rejections are logged rather than raised: a preference the model invented
    shouldn't fail a chat turn the user is waiting on, but it does need to be
    visible, because a rejection means the model isn't following the prompt's
    key list and that's worth noticing in the logs.
    """
    normalized: dict[str, str] = {}
    for key, value in preferences.items():
        canonical = normalize_preference_key(key)
        if canonical is None:
            logger.warning("Discarding preference under unrecognized key %r", key)
            continue
        value = value.strip()
        if not value:
            logger.warning("Discarding preference %r with an empty value", canonical)
            continue
        normalized[canonical] = value
    return normalized


def get_preferences(user_id: int) -> dict[str, str]:
    """Preference facts as a plain key->value dict, for retrieval.

    Keys are normalized on read so rows stored before the vocabulary existed
    still reach retrieval. Where a legacy key and its canonical form both
    exist, the canonical row wins.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select key, value from user_preferences where user_id = %s order by updated_at asc;",
                (user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    resolved: dict[str, str] = {}
    for key, value in rows:
        canonical = normalize_preference_key(key)
        if canonical is None:
            continue
        # A row already stored under the canonical key is authoritative over a
        # legacy alias, regardless of which was written more recently.
        if canonical in resolved and key != canonical:
            continue
        resolved[canonical] = value
    return resolved


def upsert_preferences(user_id: int, preferences: dict[str, str]) -> dict[str, str]:
    """Insert new preference facts or overwrite existing ones for the same key.

    Returns what was actually stored after normalization, which is not
    necessarily what was passed in - the caller needs that to tell the user
    truthfully what got remembered.
    """
    preferences = normalize_preferences(preferences)
    if not preferences:
        return {}

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
        logger.info("Upserted %d preference fact(s) for user_id=%s: %s", len(rows), user_id, sorted(preferences))
        return preferences
    finally:
        conn.close()


def list_preferences(user_id: int) -> list[dict]:
    """All stored preference facts for a user, most recently updated first -
    for a user-facing "what do you remember about me" view, unlike
    get_preferences() which returns a plain key->value dict for retrieval.

    Unlike get_preferences this does *not* drop rows with unrecognized keys:
    a legacy row the system can no longer act on is exactly the thing a user
    should be able to see and delete, and hiding it would leave it
    undeletable.
    """
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
    no-op. Matches on the key exactly as stored, so a legacy row can be
    deleted by the key the user is actually shown."""
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
