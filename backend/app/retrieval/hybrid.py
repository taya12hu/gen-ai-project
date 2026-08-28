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
just because its reviews scored well.

The two halves produce two orderings of the same restaurants, and both are
kept: see app.retrieval.fusion for why they're merged by rank (RRF) rather
than by discarding the structured signal the moment a vibe query appears.
Soft numeric constraints are loosened a step at a time when nothing matches,
rather than dropped wholesale - see app.retrieval.relaxation.

Three entry points:
- get_hybrid_candidates: the main "search" path (query understanding's
  intent=search).
- get_reviews_for_restaurant: the "followup_question" path, for a message
  that refers back to one restaurant already named in the conversation.
- (internally) _semantic_search: the shared pgvector query.
"""

import logging
from dataclasses import dataclass, field

from app.retrieval.cache import CACHE_MISS, TTLCache
from app.retrieval.fusion import SEMANTIC_TOP_K, reciprocal_rank_fusion, semantic_score
from app.retrieval.relaxation import AppliedRelaxation, relaxation_ladder
from app.reviews.embedding_model import get_embedder
from app.storage.db import ensure_vector_registered, get_connection

logger = logging.getLogger(__name__)

RESTAURANT_COLUMNS = ["id", "name", "place", "city", "cuisines", "price", "rating", "rest_type", "votes"]

DEFAULT_LIMIT = 10
SNIPPETS_PER_RESTAURANT = 3

# Reviews to pull from pgvector before aggregating by restaurant. Each
# restaurant holds up to 15 reviews (see app.reviews.ingest), so a small
# review pool can collapse to only a handful of distinct restaurants; this has
# to be generous enough that aggregation has real choice.
MIN_SEMANTIC_REVIEW_POOL = 300
SEMANTIC_REVIEW_MULTIPLIER = 40

# HNSW explores `ef_search` candidates per query. The default (40) is fine for
# an unfiltered search but far too small for a *filtered* one: pgvector
# post-filters, so with a restaurant_id restriction covering a few percent of
# the corpus most of those 40 get discarded and the query silently returns far
# fewer rows than asked for. See _vector_search_settings.
MIN_HNSW_EF_SEARCH = 400

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
    relaxation: AppliedRelaxation | None
    used_semantic: bool

    @property
    def relaxed(self) -> bool:
        """True when price/rating constraints had to be loosened to find
        anything. Kept as a plain bool because most callers only need to know
        *whether* to mention it; `relaxation.describe()` says what changed."""
        return self.relaxation is not None and self.relaxation.is_relaxed

    def relaxation_note(self) -> str | None:
        return self.relaxation.describe() if self.relaxation is not None else None


def _row_to_candidate(row) -> RestaurantCandidate:
    return RestaurantCandidate(**dict(zip(RESTAURANT_COLUMNS, row)))


def _structured_query(
    conn, filters: HybridFilters, relaxation: AppliedRelaxation, limit: int
) -> list[RestaurantCandidate]:
    sql = "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where true"
    params: list = []

    if filters.place:
        sql += " and place = %s"
        params.append(filters.place)
    if filters.cuisines:
        sql += " and cuisines && %s"
        params.append(filters.cuisines)
    if relaxation.used_max_price is not None:
        sql += " and price <= %s"
        params.append(relaxation.used_max_price)
    if relaxation.used_min_rating is not None:
        sql += " and rating >= %s"
        params.append(relaxation.used_min_rating)

    sql += " order by rating desc, votes desc, id limit %s"
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
        ladder = relaxation_ladder(filters.max_price, filters.min_rating)

        # Probe the loosest rung first when there's a ladder to walk. Whether
        # *anything* matches is decided entirely by place/cuisine, which are
        # never relaxed - so if the fully-relaxed rung is empty, every rung is,
        # and walking them all is a guaranteed-wasted series of round trips on
        # exactly the query that already disappointed the user.
        if len(ladder) > 1 and not _structured_query(conn, filters, ladder[-1], limit=1):
            logger.info("Structured retrieval: place/cuisine filters match nothing; skipping relaxation ladder")
            return HybridRetrievalResult(candidates=[], relaxation=ladder[-1], used_semantic=False)

        for attempt in ladder:
            candidates = _structured_query(conn, filters, attempt, limit)
            if candidates:
                if attempt.is_relaxed:
                    logger.info(
                        "Structured retrieval: %d candidate(s) after relaxing (%s)",
                        len(candidates), attempt.describe(),
                    )
                else:
                    logger.info("Structured retrieval: %d candidate(s), no relaxation needed", len(candidates))
                return HybridRetrievalResult(candidates=candidates, relaxation=attempt, used_semantic=False)

        # Every rung matched nothing: place/cuisine alone rule everything out,
        # and those are never relaxed (see app.retrieval.relaxation).
        logger.info("Structured retrieval: no candidates even with all numeric constraints dropped")
        return HybridRetrievalResult(candidates=[], relaxation=ladder[-1], used_semantic=False)
    finally:
        conn.close()


def _resolve_relaxation(conn, filters: HybridFilters) -> AppliedRelaxation | None:
    """The first relaxation rung that matches anything, or None if none does.

    The hybrid path needs to know *which* constraints ended up applying, but
    not which restaurants satisfy them - those go into the vector query as
    predicates rather than as a list of ids (see _semantic_search). Each probe
    is `limit 1` against indexed columns, and the common case answers on the
    first rung.
    """
    ladder = relaxation_ladder(filters.max_price, filters.min_rating)

    # Whether *anything* matches is decided entirely by place/cuisine, which
    # are never relaxed - so if the fully-relaxed rung is empty, every rung is.
    if len(ladder) > 1 and not _structured_query(conn, filters, ladder[-1], limit=1):
        return None

    for attempt in ladder:
        if _structured_query(conn, filters, attempt, limit=1):
            return attempt
    return None


def get_structured_candidates(filters: HybridFilters, limit: int = DEFAULT_LIMIT) -> HybridRetrievalResult:
    """Structured-only ranking (rating desc, votes desc), walking the relaxation ladder until
    something matches. Used when there's no vibe_query to run semantic search with. Cached (see
    cache.py): identical filters/limit within the TTL window skip the DB round trip entirely."""
    key = _structured_cache_key(filters, limit)
    cached = _cache.get(key)
    if cached is not CACHE_MISS:
        return cached

    result = _get_structured_candidates_uncached(filters, limit)
    _cache.set(key, result)
    return result


def _vector_search_settings(review_limit: int, filtered: bool) -> str:
    """SQL prelude raising HNSW's search effort for this transaction only.

    Returned as a string to be prepended to the search statement rather than
    executed on its own. Against a remote database every statement is a round
    trip costing ~200ms, and a bare `SET` bought nothing else - sending it
    attached to the query it configures makes it free.

    `set local` (not `set`) still matters: connections are pooled and handed
    to the next caller afterwards, so a session-level GUC would leak into
    unrelated queries. Transaction scope means it is undone by the rollback
    the pool performs on release.

    Filtered and unfiltered searches get opposite treatment - see the branch
    below for the measurements behind that.
    """
    ef_search = max(MIN_HNSW_EF_SEARCH, review_limit)

    if filtered:
        # Scan the filtered subset exactly rather than approximately.
        #
        # Filtered HNSW is post-filtered: it walks the graph and discards
        # everything outside the predicate, so it grinds hardest at *moderate*
        # selectivity - exactly where real queries live. Measured over six
        # distinct query vectors, "quiet in Whitefield" (584 restaurants, ~6.5%
        # of reviews) took 1.797s via the index against 0.179s scanning the
        # subset directly; a neighbouring case of almost the same size (BTM,
        # 516) took 0.164s either way, because the planner's selectivity
        # estimate happened to land differently. That cliff is the problem: not
        # that the index is slow on average, but that whether you fall off it
        # is unpredictable.
        #
        # Scanning is bounded and predictable instead - worst case measured was
        # 0.813s with a filter matching nearly the whole corpus, still better
        # than the index's worst case. Approximation only earns its keep when
        # the candidate set is genuinely unbounded, which is the unfiltered
        # branch below.
        return "set local enable_indexscan = off; set local enable_indexonlyscan = off; "

    return f"set local hnsw.ef_search = {ef_search}; "


def _semantic_search(
    conn,
    vibe_query: str,
    review_limit: int,
    filters: HybridFilters | None = None,
    relaxation: AppliedRelaxation | None = None,
    restaurant_ids: list[int] | None = None,
) -> list[tuple[int, int, float | None, float]]:
    """Returns (review_id, restaurant_id, review_rating, similarity) rows, best matches first.

    Restriction comes in one of two forms. `filters`/`relaxation` join to
    `restaurants` and apply the structured predicates directly - that is the
    search path, and it lets every eligible restaurant compete regardless of
    how many there are. `restaurant_ids` restricts to an explicit list, which
    is what the single-restaurant follow-up path needs.

    Expressing the search path as a join is what removed the old pool cap. The
    previous shape fetched the top N matching restaurants by rating, then
    passed their ids back in - so rating decided who was allowed to compete
    before the vibe query was read at all, and three areas in the dataset
    (Whitefield 584, BTM 516, HSR 509) exceeded the cap and silently lost their
    lowest-rated members. Rating is already one of the two fusion signals; it
    should not also be a gate.

    Deliberately does NOT select review_text. Scoring needs a similarity per
    review; only the handful of snippets actually displayed need their text.
    Text is fetched afterwards, for the survivors only, by
    _fetch_candidates_with_texts.
    """
    query_embedding = get_embedder().encode(vibe_query)

    predicates: list[str] = []
    params: list = [query_embedding]

    if filters is not None:
        if filters.place:
            predicates.append("r.place = %s")
            params.append(filters.place)
        if filters.cuisines:
            predicates.append("r.cuisines && %s")
            params.append(filters.cuisines)
        if relaxation is not None:
            if relaxation.used_max_price is not None:
                predicates.append("r.price <= %s")
                params.append(relaxation.used_max_price)
            if relaxation.used_min_rating is not None:
                predicates.append("r.rating >= %s")
                params.append(relaxation.used_min_rating)
    if restaurant_ids is not None:
        predicates.append("v.restaurant_id = any(%s)")
        params.append(restaurant_ids)

    prelude = _vector_search_settings(review_limit, filtered=bool(predicates))

    sql = prelude + (
        "select v.id, v.restaurant_id, v.review_rating, "
        "1 - (v.embedding <=> %s) as similarity from restaurant_reviews v "
    )
    # Only join when a structured predicate needs it - an unfiltered vibe query
    # has no reason to touch the restaurants table at all.
    if filters is not None and (filters.place or filters.cuisines or relaxation is not None):
        sql += "join restaurants r on r.id = v.restaurant_id "
    sql += "where v.embedding is not null"
    for predicate in predicates:
        sql += f" and {predicate}"

    sql += " order by v.embedding <=> %s limit %s"
    params.extend([query_embedding, review_limit])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _fetch_candidates_with_texts(
    conn, restaurant_ids: list[int], review_ids: list[int]
) -> tuple[dict[int, RestaurantCandidate], dict[int, str]]:
    """Full restaurant rows and the bodies of the snippets they'll show, in one
    statement.

    These were two queries, which on a remote database is two round trips at
    ~200ms each - more than the transfer they were saving. The left join keeps
    restaurants that have no surviving snippet (a structured-only match), and
    the restaurant columns repeat per review row, which is cheap at five
    restaurants and three snippets each.
    """
    if not restaurant_ids:
        return {}, {}
    columns = ", ".join(f"r.{c}" for c in RESTAURANT_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"select {columns}, v.id, v.review_text from restaurants r "
            "left join restaurant_reviews v on v.restaurant_id = r.id and v.id = any(%s) "
            "where r.id = any(%s);",
            (review_ids, restaurant_ids),
        )
        rows = cur.fetchall()

    n = len(RESTAURANT_COLUMNS)
    facts: dict[int, RestaurantCandidate] = {}
    texts: dict[int, str] = {}
    for row in rows:
        restaurant_id = row[0]
        if restaurant_id not in facts:
            facts[restaurant_id] = _row_to_candidate(row[:n])
        review_id, review_text = row[n], row[n + 1]
        if review_id is not None:
            texts[review_id] = review_text
    return facts, texts


def _fetch_ranking_facts(conn, restaurant_ids: list[int]) -> dict[int, tuple[float, int]]:
    """(rating, votes) only, for building the structured ranking.

    The full restaurant row is needed for the handful of candidates that end
    up being returned, not for the couple of hundred that merely take part in
    ranking. Fetching every column for all of them was transferring names,
    cuisine arrays and addresses that were about to be discarded.
    """
    if not restaurant_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "select id, rating, votes from restaurants where id = any(%s);",
            (restaurant_ids,),
        )
        return {row[0]: (float(row[1]), row[2]) for row in cur.fetchall()}


def _fetch_restaurants(conn, restaurant_ids: list[int]) -> dict[int, RestaurantCandidate]:
    if not restaurant_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "select " + ", ".join(RESTAURANT_COLUMNS) + " from restaurants where id = any(%s);",
            (restaurant_ids,),
        )
        return {row[0]: _row_to_candidate(row) for row in cur.fetchall()}


def _group_reviews_by_restaurant(
    review_rows: list[tuple[int, int, float | None, float]],
) -> tuple[dict[int, list[tuple[int, float | None, float]]], dict[int, list[float]]]:
    """Splits raw pgvector rows into per-restaurant snippet candidates
    (capped, for display) and per-restaurant similarity lists (uncapped, for
    scoring).

    Snippet candidates carry (review_id, rating, similarity) rather than a
    finished ReviewSnippet, because the text isn't loaded yet - see
    _semantic_search. Rows arrive best-match-first, so taking the first
    SNIPPETS_PER_RESTAURANT per restaurant keeps its strongest evidence.
    """
    snippets: dict[int, list[tuple[int, float | None, float]]] = {}
    similarities: dict[int, list[float]] = {}

    for review_id, restaurant_id, rating, similarity in review_rows:
        similarity = float(similarity)
        similarities.setdefault(restaurant_id, []).append(similarity)
        bucket = snippets.setdefault(restaurant_id, [])
        if len(bucket) < SNIPPETS_PER_RESTAURANT:
            bucket.append((review_id, float(rating) if rating is not None else None, similarity))
    return snippets, similarities


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
    """Cached (see cache.py): identical (filters, vibe_query, limit) within the TTL window
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


def _fuse_and_hydrate(
    conn,
    snippets: dict[int, list[tuple[int, float | None, float]]],
    similarities: dict[int, list[float]],
    ranking_facts: dict[int, tuple[float, int]],
    limit: int,
) -> list[RestaurantCandidate]:
    """Merges the two rankings, then loads full data for the survivors only.

    Two independent orderings of the same restaurants. Semantic: the strength
    *and* the amount of supporting review evidence. Structured: the
    restaurant's own quality, which pure semantic ranking would otherwise
    throw away entirely.

    Hydration happens after fusion rather than before it, so the expensive
    columns - the full restaurant row, and the review bodies - are fetched for
    `limit` restaurants instead of the couple of hundred that merely took part
    in ranking.
    """
    semantic_ranking = sorted(
        similarities.keys(),
        key=lambda rid: (-semantic_score(similarities[rid]), rid),
    )
    structured_ranking = sorted(
        (rid for rid in similarities if rid in ranking_facts),
        key=lambda rid: (-ranking_facts[rid][0], -ranking_facts[rid][1], rid),
    )

    fused = reciprocal_rank_fusion([structured_ranking, semantic_ranking])
    top_ids = [f.restaurant_id for f in fused if f.restaurant_id in ranking_facts][:limit]

    snippet_ids = [review_id for rid in top_ids for review_id, _, _ in snippets.get(rid, [])]
    facts, texts = _fetch_candidates_with_texts(conn, top_ids, snippet_ids)

    candidates: list[RestaurantCandidate] = []
    for rid in top_ids:
        candidate = facts.get(rid)
        if candidate is None:
            continue
        candidate.review_snippets = [
            ReviewSnippet(id=review_id, text=texts[review_id], rating=rating, similarity=similarity)
            for review_id, rating, similarity in snippets.get(rid, [])
            if review_id in texts
        ]
        candidates.append(candidate)
    return candidates


def _get_hybrid_candidates_uncached(filters: HybridFilters, vibe_query: str, limit: int) -> HybridRetrievalResult:
    # Structured work happens *before* this function takes a connection of its
    # own, and the semantic-empty fallback happens after it has released one.
    # Holding two pooled connections at once for a single request would make
    # the pool's effective capacity half its size, and half of a bounded pool
    # is exactly the kind of thing that only shows up under concurrency (see
    # MAX_POOL_SIZE in app.storage.db, which the request threadpool is now
    # sized against).
    review_limit = max(limit * SEMANTIC_REVIEW_MULTIPLIER, MIN_SEMANTIC_REVIEW_POOL)

    conn = get_connection()
    candidates: list[RestaurantCandidate] | None = None
    similarities: dict[int, list[float]] = {}
    relaxation: AppliedRelaxation | None = None
    try:
        ensure_vector_registered(conn)

        if _has_any_filter(filters):
            # Which constraints apply, not which restaurants satisfy them -
            # the predicates go straight into the search below, so every
            # eligible restaurant competes however many there are.
            relaxation = _resolve_relaxation(conn, filters)
            if relaxation is None:
                logger.info("Hybrid retrieval: structured filters matched nothing, skipping semantic search")
                return HybridRetrievalResult(
                    candidates=[],
                    relaxation=relaxation_ladder(filters.max_price, filters.min_rating)[-1],
                    used_semantic=True,
                )
            if relaxation.is_relaxed:
                logger.info("Hybrid retrieval: relaxed constraints (%s)", relaxation.describe())

        review_rows = _semantic_search(
            conn,
            vibe_query,
            review_limit=review_limit,
            filters=filters if _has_any_filter(filters) else None,
            relaxation=relaxation,
        )
        snippets, similarities = _group_reviews_by_restaurant(review_rows)

        if similarities:
            # Rank on (rating, votes) alone - the full restaurant rows are
            # only worth fetching for the few that survive fusion.
            ranking_facts = _fetch_ranking_facts(conn, list(similarities.keys()))
            candidates = _fuse_and_hydrate(conn, snippets, similarities, ranking_facts, limit)
    finally:
        conn.close()

    if candidates is None:
        # No reviews matched semantically (e.g. sparse pool) - fall back to structured ranking.
        logger.info("Hybrid retrieval: semantic search matched no reviews, falling back to structured ranking")
        fallback = get_structured_candidates(filters, limit=limit)
        return HybridRetrievalResult(
            candidates=fallback.candidates, relaxation=fallback.relaxation, used_semantic=False
        )
    logger.info(
        "Hybrid retrieval: %d candidate(s) across %d matched restaurant(s), fused (top-%d semantic evidence)",
        len(candidates), len(similarities), SEMANTIC_TOP_K,
    )
    return HybridRetrievalResult(candidates=candidates, relaxation=relaxation, used_semantic=True)


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
            ensure_vector_registered(conn)
            review_rows = _semantic_search(
                conn, vibe_query, review_limit=limit, restaurant_ids=[restaurant_id]
            )
            _, texts = _fetch_candidates_with_texts(conn, [restaurant_id], [row[0] for row in review_rows])
            candidate.review_snippets = [
                ReviewSnippet(
                    id=review_id,
                    text=texts[review_id],
                    rating=float(rating) if rating is not None else None,
                    similarity=float(similarity),
                )
                for review_id, _, rating, similarity in review_rows
                if review_id in texts
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
        return _fetch_restaurants(conn, restaurant_ids)
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
