"""
Phase 4 - Preference Input Layer.

Fetches the set of place and cuisine values that actually exist in the
restaurants table (Phase 3). Used to validate/normalize incoming user
preferences against real data instead of trusting free-text input blindly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3_storage_indexing"))
from db import get_connection  # noqa: E402


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
