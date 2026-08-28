"""
Retrieval - known place/cuisine values.

Fetches the set of place and cuisine values that actually exist in the
restaurants table. Used by query understanding to snap free-text place/
cuisine mentions in a chat message to canonical DB casing (see
app.query_understanding).

Both queries are aggregates over the whole restaurants table
(`select distinct place`, `select distinct unnest(cuisines)`), and they were
previously run on *every single chat turn* - two full scans of ~9,200 rows
to rebuild a list that only changes when the dataset is re-ingested, which
never happens while the server is running. They're cached in-process for the
lifetime of the process instead; call refresh() after a re-ingest (or just
restart, which is what the retrieval cache already expects - see
app.retrieval.hybrid.clear_cache).
"""

import logging
import threading

from app.storage.db import get_connection

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_known_places: set[str] | None = None
_known_cuisines: set[str] | None = None


def refresh() -> None:
    """Drops the cached value sets so the next lookup re-reads the database."""
    global _known_places, _known_cuisines
    with _lock:
        _known_places = None
        _known_cuisines = None


def _load() -> tuple[set[str], set[str]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select distinct place from restaurants;")
            places = {row[0] for row in cur.fetchall()}
            cur.execute("select distinct unnest(cuisines) from restaurants;")
            cuisines = {row[0] for row in cur.fetchall()}
        return places, cuisines
    finally:
        conn.close()


def _ensure_loaded() -> None:
    global _known_places, _known_cuisines
    if _known_places is not None and _known_cuisines is not None:
        return
    with _lock:
        # Re-check under the lock: two threads can both miss above, and only
        # the first should pay for the scan.
        if _known_places is not None and _known_cuisines is not None:
            return
        places, cuisines = _load()
        _known_places, _known_cuisines = places, cuisines
        logger.info("Loaded known values: %d place(s), %d cuisine(s)", len(places), len(cuisines))


def get_known_places() -> set[str]:
    _ensure_loaded()
    return _known_places


def get_known_cuisines() -> set[str]:
    _ensure_loaded()
    return _known_cuisines
