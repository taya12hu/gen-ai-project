"""
Phase 5 - Retrieval / Filtering Engine.

Turns a validated UserPreferences (Phase 4) into a small, ranked shortlist
of candidate restaurants from the database (Phase 3). This shortlist is the
*only* restaurant data the LLM will see in Phase 7 - keeping it here (not
the whole dataset) is what keeps the LLM grounded and cheap.

Filtering: place (exact) + cuisines (array overlap - matches a restaurant
serving *any* of the requested cuisines) are always enforced - they're the
preferences a user picks deliberately about *what kind of place* they want.
price/rating are optional and, when given, enforced as a strict pass first;
if that yields nothing, we relax whichever of price/rating were provided
and retry, so the user still gets a shortlist (of the right place/cuisine)
instead of a hard empty result.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3_storage_indexing"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4_preference_input"))

from db import get_connection  # noqa: E402
from preferences import UserPreferences  # noqa: E402

CANDIDATE_COLUMNS = ["id", "name", "place", "city", "cuisines", "price", "rating", "rest_type", "votes"]

DEFAULT_LIMIT = 10


@dataclass
class RetrievalResult:
    candidates: list[dict]
    relaxed: bool  # True if price/rating constraints had to be dropped to find matches


def _row_to_dict(row) -> dict:
    return dict(zip(CANDIDATE_COLUMNS, row))


def _query(conn, prefs: UserPreferences, enforce_price_rating: bool, limit: int) -> list[dict]:
    sql = (
        "select id, name, place, city, cuisines, price, rating, rest_type, votes "
        "from restaurants where place = %s and cuisines && %s"
    )
    params: list = [prefs.place, prefs.cuisines]

    if enforce_price_rating:
        if prefs.max_price is not None:
            sql += " and price <= %s"
            params.append(prefs.max_price)
        if prefs.min_rating is not None:
            sql += " and rating >= %s"
            params.append(prefs.min_rating)

    sql += " order by rating desc, votes desc limit %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_candidates(prefs: UserPreferences, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
    conn = get_connection()
    try:
        strict = _query(conn, prefs, enforce_price_rating=True, limit=limit)
        if strict:
            return RetrievalResult(candidates=strict, relaxed=False)

        relaxed = _query(conn, prefs, enforce_price_rating=False, limit=limit)
        return RetrievalResult(candidates=relaxed, relaxed=True)
    finally:
        conn.close()
