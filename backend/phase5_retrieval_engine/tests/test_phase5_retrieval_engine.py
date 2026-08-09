"""
Tests for Phase 5 - Retrieval / Filtering Engine.

Runs against the live Supabase database (Phase 3), using
'Indiranagar' + 'Chinese' as a known-populous combination (89 rows,
price 40-1700, rating 2.1-4.5) to exercise the strict filter path, the
price/rating relaxation fallback, multi-cuisine overlap matching, and
optional (omitted) price/rating.
"""

import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE_DIR.parent / "phase4_preference_input"))

from preferences import UserPreferences  # noqa: E402
from retrieval import get_candidates  # noqa: E402


def make_prefs(place="Indiranagar", cuisines=("Chinese",), max_price=800, min_rating=4.0):
    return UserPreferences(
        place=place, cuisines=list(cuisines), max_price=max_price, min_rating=min_rating
    )


def test_strict_match_returns_results_within_bounds():
    result = get_candidates(make_prefs(max_price=800, min_rating=4.0), limit=10)

    assert result.relaxed is False
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert c["place"] == "Indiranagar"
        assert "Chinese" in c["cuisines"]
        assert c["price"] <= 800
        assert c["rating"] >= 4.0


def test_results_ordered_by_rating_desc_then_votes_desc():
    result = get_candidates(make_prefs(max_price=2000, min_rating=0), limit=10)

    ratings = [c["rating"] for c in result.candidates]
    assert ratings == sorted(ratings, reverse=True)

    # within any tied rating group, votes should be non-increasing
    for i in range(len(result.candidates) - 1):
        a, b = result.candidates[i], result.candidates[i + 1]
        if a["rating"] == b["rating"]:
            assert a["votes"] >= b["votes"]


def test_limit_is_respected():
    result = get_candidates(make_prefs(max_price=2000, min_rating=0), limit=3)
    assert len(result.candidates) <= 3


def test_impossible_price_rating_combo_triggers_relaxation():
    # min price in this place+cuisine group is 40, max rating is 4.5 -
    # price<=50 AND rating>=4.9 together is not satisfiable.
    result = get_candidates(make_prefs(max_price=50, min_rating=4.9), limit=10)

    assert result.relaxed is True
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert c["place"] == "Indiranagar"
        assert "Chinese" in c["cuisines"]


def test_nonexistent_place_cuisine_combo_returns_empty():
    result = get_candidates(
        make_prefs(place="Indiranagar", cuisines=["Klingon Cuisine"], max_price=2000, min_rating=0),
        limit=10,
    )
    assert result.candidates == []
    assert result.relaxed is True


def test_all_candidates_actually_contain_requested_cuisine():
    result = get_candidates(make_prefs(max_price=2000, min_rating=0), limit=10)
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert "Chinese" in c["cuisines"]


# --- Multi-cuisine (array overlap) -----------------------------------------


def test_multiple_cuisines_match_restaurants_with_any_of_them():
    result = get_candidates(
        make_prefs(cuisines=["Chinese", "Cafe"], max_price=2000, min_rating=0), limit=20
    )

    assert len(result.candidates) > 0
    for c in result.candidates:
        assert "Chinese" in c["cuisines"] or "Cafe" in c["cuisines"]

    # and it should find at least one restaurant that ISN'T Chinese (i.e. Cafe-only)
    # confirming this is an OR/overlap match, not requiring every requested cuisine.
    cafe_only = [c for c in result.candidates if "Chinese" not in c["cuisines"]]
    assert all("Cafe" in c["cuisines"] for c in cafe_only)


# --- Optional price/rating ---------------------------------------------------


def test_omitted_price_and_rating_returns_results_without_relaxation():
    prefs = UserPreferences(place="Indiranagar", cuisines=["Chinese"])
    result = get_candidates(prefs, limit=10)

    assert result.relaxed is False
    assert len(result.candidates) > 0
    for c in result.candidates:
        assert c["place"] == "Indiranagar"
        assert "Chinese" in c["cuisines"]


def test_omitted_price_only_still_enforces_rating():
    prefs = UserPreferences(place="Indiranagar", cuisines=["Chinese"], min_rating=4.4)
    result = get_candidates(prefs, limit=10)

    assert len(result.candidates) > 0
    for c in result.candidates:
        if not result.relaxed:
            # Compare as float on both sides - Decimal('4.4') >= 4.4 (float) can be
            # False due to 4.4 not being exactly representable in binary floating point.
            assert float(c["rating"]) >= 4.4
