"""
Tests for the Hybrid Retrieval Engine.

Runs against the live database (restaurants + restaurant_reviews/
embeddings). Indiranagar + Chinese is a known-populous combination (89
rows, price 40-1700, rating 2.1-4.5), so the structured-only assertions
mirror that; the semantic assertions target review content that should
genuinely be present given ~225k real Zomato reviews (ambience/service/
quiet are common review themes).
"""

from app.retrieval.hybrid import (
    HybridFilters,
    get_hybrid_candidates,
    get_reviews_for_restaurant,
    get_structured_candidates,
)


# --- structured-only path (no vibe_query) -----------------------------------


def test_structured_only_matches_place_and_cuisine():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0)
    result = get_structured_candidates(filters, limit=10)

    assert result.relaxed is False
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert c.place == "Indiranagar"
        assert "Chinese" in c.cuisines
        assert c.price <= 800
        assert c.rating >= 4.0


def test_structured_only_with_no_filters_returns_top_rated():
    result = get_structured_candidates(HybridFilters(), limit=5)
    assert len(result.candidates) == 5
    ratings = [c.rating for c in result.candidates]
    assert ratings == sorted(ratings, reverse=True)


def test_get_hybrid_candidates_without_vibe_query_matches_structured_path():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"])
    structured = get_structured_candidates(filters, limit=10)
    hybrid = get_hybrid_candidates(filters, vibe_query=None, limit=10)

    assert [c.id for c in hybrid.candidates] == [c.id for c in structured.candidates]
    assert hybrid.used_semantic is False


# --- hybrid: structured + semantic --------------------------------------------


def test_hybrid_with_vibe_query_restricts_to_structured_pool():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"])
    result = get_hybrid_candidates(filters, vibe_query="quiet ambience good for a date", limit=5)

    assert result.used_semantic is True
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert c.place == "Indiranagar"
        assert "Chinese" in c.cuisines


def test_hybrid_candidates_carry_review_snippets():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"])
    result = get_hybrid_candidates(filters, vibe_query="good service and ambience", limit=5)

    assert len(result.candidates) > 0
    assert any(c.review_snippets for c in result.candidates)


def test_hybrid_vibe_only_search_has_no_place_restriction():
    result = get_hybrid_candidates(HybridFilters(), vibe_query="quiet and romantic", limit=5)
    assert result.used_semantic is True
    assert len(result.candidates) > 0


def test_hybrid_impossible_structured_filters_returns_empty():
    filters = HybridFilters(place="Indiranagar", cuisines=["Klingon Cuisine"])
    result = get_hybrid_candidates(filters, vibe_query="quiet", limit=5)
    assert result.candidates == []


# --- followup: single-restaurant lookup ---------------------------------------


def test_get_reviews_for_restaurant_returns_facts_and_snippets():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"])
    seed = get_structured_candidates(filters, limit=1).candidates[0]

    result = get_reviews_for_restaurant(seed.id, vibe_query="ambience")
    assert result is not None
    assert result.id == seed.id
    assert result.name == seed.name


def test_get_reviews_for_restaurant_without_vibe_query_uses_top_rated_reviews():
    filters = HybridFilters(place="Indiranagar", cuisines=["Chinese"])
    seed = get_structured_candidates(filters, limit=1).candidates[0]

    result = get_reviews_for_restaurant(seed.id, vibe_query=None, limit=3)
    assert result is not None
    assert len(result.review_snippets) <= 3


def test_get_reviews_for_restaurant_returns_none_for_unknown_id():
    assert get_reviews_for_restaurant(-1, vibe_query="ambience") is None
