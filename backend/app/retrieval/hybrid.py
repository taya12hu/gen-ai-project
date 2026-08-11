"""
Hybrid Retrieval Engine.

Combines structured SQL filtering (exact/cheap on place, cuisine, price,
rating) with pgvector semantic search over restaurant_reviews for the
fuzzy part of a query ("quiet", "good for a date") that no amount of
column filtering can answer.

Structured facts stay structured; only the qualitative remainder goes
through embeddings - and when both are present, the semantic search is
restricted to the structured candidate pool, so "quiet date spot in
Koramangala under Rs800" never returns a quiet place outside Koramangala
just because its reviews scored well semantically.

Three entry points:
- get_hybrid_candidates: the main "search" path (query understanding's
  intent=search).
- get_reviews_for_restaurant: the "followup_question" path, for a message
  that refers back to one restaurant already named in the conversation.
- (internally) semantic_search_reviews: the shared pgvector query.
"""

import logging
from dataclasses import dataclass, field

from pgvector.psycopg2 import register_vector

from app.retrieval.cache import CACHE_MISS, TTLCache
from app.reviews.embedding_model import get_embedder
from app.storage.db import get_connection

logger = logging.getLogger(__name__)

RESTAURANT_COLUMNS = ["id", "name", "place", "city", "cuisines", "price", "rating", "rest_type", "votes"]

DEFAULT_LIMIT = 10
SNIPPETS_PER_RESTAURANT = 3
SEMANTIC_POOL_MULTIPLIER = 8  # pull more reviews than needed so aggregation-by-restaurant has room

# The restaurant/review data is a one-time Hugging Face dataset load (app.data/app.reviews)
# with no live refresh pipeline, so a long TTL is safe - it isn't going to serve
# stale results against data that's actively changing. Bounded by max_size (see
# TTLCache), not by how long entries live, so this doesn't grow unbounded either.
# If the dataset is ever manually re-ingested while the server is running, call
# clear_cache() (or just restart the process) to avoid serving week-old results.
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week
_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)


def clear_cache() -> None:
    _cache.clear()


@dataclass
class ReviewSnippet:
    id: int
    text: str
    rating: float | None
    similarity: float


@dataclass
class RestaurantCandidate:
    id: int
    name: str
    place: str
    city: str
    cuisines: list[str]
    price: float
    rating: float
    rest_type: str | None
    votes: int
    review_snippets: list[ReviewSnippet] = field(default_factory=list)


@dataclass
class HybridFilters:
    place: str | None = None
    cuisines: list[str] = field(default_factory=list)
    max_price: float | None = None
    min_rating: float | None = None


@dataclass
class HybridRetrievalResult:
    candidates: list[RestaurantCandidate]
    relaxed: bool  # price/rating constraints were dropped to find any structured matches
    used_semantic: bool


def _row_to_candidate(row) -> RestaurantCandidate:
    return RestaurantCandidate(**dict(zip(RESTAURANT_COLUMNS, row)))


def _structured_query(conn, filters: HybridFilters, enforce_price_rating: bool, limit: int) -> list[RestaurantCandidate]:
    sql = "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where true"
    params: list = []

    if filters.place:
        sql += " and place = %s"
        params.append(filters.place)
    if filters.cuisines:
        sql += " and cuisines && %s"
        params.append(filters.cuisines)
    if enforce_price_rating:
        if filters.max_price is not None:
            sql += " and price <= %s"
            params.append(filters.max_price)
        if filters.min_rating is not None:
            sql += " and rating >= %s"
            params.append(filters.min_rating)

    sql += " order by rating desc, votes desc limit %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_candidate(r) for r in rows]


def _has_any_filter(filters: HybridFilters) -> bool:
    return bool(filters.place or filters.cuisines or filters.max_price is not None or filters.min_rating is not None)


def _structured_cache_key(filters: HybridFilters, limit: int) -> tuple:
    return ("structured", filters.place, tuple(sorted(filters.cuisines)), filters.max_price, filters.min_rating, limit)


def _get_structured_candidates_uncached(filters: HybridFilters, limit: int) -> HybridRetrievalResult:
    conn = get_connection()
    try:
        strict = _structured_query(conn, filters, enforce_price_rating=True, limit=limit)
        if strict:
            logger.info("Structured retrieval: %d candidate(s), no relaxation needed", len(strict))
            return HybridRetrievalResult(candidates=strict, relaxed=False, used_semantic=False)

        relaxed = _structured_query(conn, filters, enforce_price_rating=False, limit=limit)
        logger.info("Structured retrieval: strict filters matched nothing, relaxed to %d candidate(s)", len(relaxed))
        return HybridRetrievalResult(candidates=relaxed, relaxed=True, used_semantic=False)
    finally:
        conn.close()


def get_structured_candidates(filters: HybridFilters, limit: int = DEFAULT_LIMIT) -> HybridRetrievalResult:
    """Structured-only ranking (rating desc, votes desc), with relax-on-empty logic generalized to
    all-optional filters. Used when there's no vibe_query to run semantic search with. Cached (see
    cache.py): identical filters/limit within the TTL window skip the DB round trip entirely."""
    key = _structured_cache_key(filters, limit)
    cached = _cache.get(key)
    if cached is not CACHE_MISS:
        return cached

    result = _get_structured_candidates_uncached(filters, limit)
    _cache.set(key, result)
    return result


def _semantic_search(conn, vibe_query: str, restaurant_ids: list[int] | None, review_limit: int) -> list[tuple[int, int, str, float | None, float]]:
    """Returns (review_id, restaurant_id, review_text, review_rating, similarity) rows, best matches first."""
    query_embedding = get_embedder().encode(vibe_query)

    sql = (
        "select id, restaurant_id, review_text, review_rating, "
        "1 - (embedding <=> %s) as similarity "
        "from restaurant_reviews where embedding is not null"
    )
    params: list = [query_embedding]

    if restaurant_ids is not None:
        sql += " and restaurant_id = any(%s)"
        params.append(restaurant_ids)

    sql += " order by embedding <=> %s limit %s"
    params.extend([query_embedding, review_limit])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _attach_snippets(conn, ranked_restaurant_ids: list[int], snippets_by_restaurant: dict[int, list[ReviewSnippet]]) -> list[RestaurantCandidate]:
    if not ranked_restaurant_ids:
        return []

    with conn.cursor() as cur:
        cur.execute(
            "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where id = any(%s);",
            (ranked_restaurant_ids,),
        )
        by_id = {row[0]: _row_to_candidate(row) for row in cur.fetchall()}

    candidates = []
    for rid in ranked_restaurant_ids:
        candidate = by_id.get(rid)
        if candidate is None:
            continue
        candidate.review_snippets = snippets_by_restaurant.get(rid, [])
        candidates.append(candidate)
    return candidates


def _hybrid_cache_key(filters: HybridFilters, vibe_query: str | None, limit: int) -> tuple:
    return (
        "hybrid",
        filters.place,
        tuple(sorted(filters.cuisines)),
        filters.max_price,
        filters.min_rating,
        vibe_query,
        limit,
    )


def get_hybrid_candidates(filters: HybridFilters, vibe_query: str | None, limit: int = DEFAULT_LIMIT) -> HybridRetrievalResult:
    """Cached (see retrieval_cache.py): identical (filters, vibe_query, limit) within the TTL window
    skip both the DB round trip and, for a vibe query, the embed + pgvector search."""
    if not vibe_query:
        return get_structured_candidates(filters, limit=limit)

    key = _hybrid_cache_key(filters, vibe_query, limit)
    cached = _cache.get(key)
    if cached is not CACHE_MISS:
        return cached

    result = _get_hybrid_candidates_uncached(filters, vibe_query, limit)
    _cache.set(key, result)
    return result


def _get_hybrid_candidates_uncached(filters: HybridFilters, vibe_query: str, limit: int) -> HybridRetrievalResult:
    conn = get_connection()
    try:
        register_vector(conn)

        restaurant_pool: list[int] | None = None
        relaxed = False
        if _has_any_filter(filters):
            structured = get_structured_candidates(filters, limit=max(limit * SEMANTIC_POOL_MULTIPLIER, 50))
            restaurant_pool = [c.id for c in structured.candidates]
            relaxed = structured.relaxed
            if not restaurant_pool:
                # Structured filters matched nothing at all (even relaxed) - nothing to rank semantically.
                logger.info("Hybrid retrieval: structured filters matched nothing, skipping semantic search")
                return HybridRetrievalResult(candidates=[], relaxed=True, used_semantic=True)

        review_rows = _semantic_search(
            conn, vibe_query, restaurant_pool, review_limit=limit * SEMANTIC_POOL_MULTIPLIER
        )

        snippets_by_restaurant: dict[int, list[ReviewSnippet]] = {}
        ranked_ids: list[int] = []
        for review_id, restaurant_id, text, rating, similarity in review_rows:
            if restaurant_id not in snippets_by_restaurant:
                snippets_by_restaurant[restaurant_id] = []
                ranked_ids.append(restaurant_id)
            if len(snippets_by_restaurant[restaurant_id]) < SNIPPETS_PER_RESTAURANT:
                snippets_by_restaurant[restaurant_id].append(
                    ReviewSnippet(
                        id=review_id,
                        text=text,
                        rating=float(rating) if rating is not None else None,
                        similarity=float(similarity),
                    )
                )

        if not ranked_ids:
            # No reviews matched semantically (e.g. sparse pool) - fall back to structured ranking.
            logger.info("Hybrid retrieval: semantic search matched no reviews, falling back to structured ranking")
            fallback = get_structured_candidates(filters, limit=limit)
            return HybridRetrievalResult(candidates=fallback.candidates, relaxed=True, used_semantic=False)

        candidates = _attach_snippets(conn, ranked_ids[:limit], snippets_by_restaurant)
        logger.info("Hybrid retrieval: %d candidate(s) ranked semantically", len(candidates))
        return HybridRetrievalResult(candidates=candidates, relaxed=relaxed, used_semantic=True)
    finally:
        conn.close()


def get_reviews_for_restaurant(restaurant_id: int, vibe_query: str | None, limit: int = SNIPPETS_PER_RESTAURANT) -> RestaurantCandidate | None:
    """Followup-question path: pull one restaurant's facts plus its most relevant review
    snippets (semantically ranked against vibe_query if given, else most recent/highest-rated).
    Cached (see cache.py)."""
    key = ("restaurant", restaurant_id, vibe_query, limit)
    cached = _cache.get(key)
    if cached is not CACHE_MISS:
        return cached

    result = _get_reviews_for_restaurant_uncached(restaurant_id, vibe_query, limit)
    _cache.set(key, result)
    return result


def _get_reviews_for_restaurant_uncached(restaurant_id: int, vibe_query: str | None, limit: int) -> RestaurantCandidate | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where id = %s;",
                (restaurant_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        candidate = _row_to_candidate(row)

        if vibe_query:
            register_vector(conn)
            review_rows = _semantic_search(conn, vibe_query, [restaurant_id], review_limit=limit)
            candidate.review_snippets = [
                ReviewSnippet(
                    id=review_id,
                    text=text,
                    rating=float(rating) if rating is not None else None,
                    similarity=float(similarity),
                )
                for review_id, _, text, rating, similarity in review_rows
            ]
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "select id, review_text, review_rating from restaurant_reviews "
                    "where restaurant_id = %s order by review_rating desc nulls last limit %s;",
                    (restaurant_id, limit),
                )
                candidate.review_snippets = [
                    ReviewSnippet(id=review_id, text=text, rating=float(rating) if rating is not None else None, similarity=1.0)
                    for review_id, text, rating in cur.fetchall()
                ]
        return candidate
    finally:
        conn.close()


def get_restaurants_by_ids(restaurant_ids: list[int]) -> dict[int, RestaurantCandidate]:
    """Batch fact-only lookup (no reviews) for a known set of restaurant ids -
    used to replay a persisted turn exactly as it was shown, not to run a
    fresh ranked search. Not cached: reload is already a single indexed
    query, and caching by an unordered id list adds complexity for no
    measurable win here."""
    if not restaurant_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where id = any(%s);",
                (restaurant_ids,),
            )
            return {row[0]: _row_to_candidate(row) for row in cur.fetchall()}
    finally:
        conn.close()


def get_review_snippets_by_ids(review_ids: list[int]) -> dict[int, list[ReviewSnippet]]:
    """Batch lookup of specific review rows by id, grouped by restaurant_id -
    the exact snippets a persisted turn showed (see get_restaurants_by_ids),
    not a fresh top-rated/semantic re-selection that could differ from what
    the user actually saw."""
    if not review_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, restaurant_id, review_text, review_rating "
                "from restaurant_reviews where id = any(%s);",
                (review_ids,),
            )
            rows = cur.fetchall()
        by_restaurant: dict[int, list[ReviewSnippet]] = {}
        for review_id, restaurant_id, text, rating in rows:
            by_restaurant.setdefault(restaurant_id, []).append(
                ReviewSnippet(id=review_id, text=text, rating=float(rating) if rating is not None else None, similarity=1.0)
            )
        return by_restaurant
    finally:
        conn.close()
