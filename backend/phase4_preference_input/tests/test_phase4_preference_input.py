"""
Tests for Phase 4 - Preference Input Layer.

Unit tests cover UserPreferences validation and normalize_preferences'
matching logic against fake known-value sets (no DB needed). Integration
tests confirm known_values.py returns real data from the live Phase 3
database and that normalize_preferences works against it end to end.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PHASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE_DIR))

from preferences import UnknownValueError, UserPreferences, normalize_preferences  # noqa: E402


# --- UserPreferences validation -----------------------------------------------


def test_valid_preferences_construct():
    prefs = UserPreferences(
        place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0
    )
    assert prefs.place == "Indiranagar"
    assert prefs.max_price == 800


def test_max_price_and_min_rating_are_optional():
    prefs = UserPreferences(place="Indiranagar", cuisines=["Chinese"])
    assert prefs.max_price is None
    assert prefs.min_rating is None


def test_multiple_cuisines_accepted():
    prefs = UserPreferences(place="Indiranagar", cuisines=["Chinese", "Cafe", "North Indian"])
    assert prefs.cuisines == ["Chinese", "Cafe", "North Indian"]


def test_empty_cuisines_list_rejected():
    with pytest.raises(ValidationError):
        UserPreferences(place="Indiranagar", cuisines=[])


@pytest.mark.parametrize("bad_price", [0, -1, -100])
def test_max_price_must_be_positive_when_given(bad_price):
    with pytest.raises(ValidationError):
        UserPreferences(place="X", cuisines=["Y"], max_price=bad_price, min_rating=3.0)


@pytest.mark.parametrize("bad_rating", [-0.1, 5.1, 10])
def test_min_rating_must_be_within_0_to_5_when_given(bad_rating):
    with pytest.raises(ValidationError):
        UserPreferences(place="X", cuisines=["Y"], max_price=500, min_rating=bad_rating)


@pytest.mark.parametrize("blank", ["", "   "])
def test_place_cannot_be_blank(blank):
    with pytest.raises(ValidationError):
        UserPreferences(place=blank, cuisines=["Chinese"])


@pytest.mark.parametrize("blank", ["", "   "])
def test_cuisines_cannot_contain_blank_values(blank):
    with pytest.raises(ValidationError):
        UserPreferences(place="Indiranagar", cuisines=["Chinese", blank])


def test_place_and_cuisines_are_trimmed():
    prefs = UserPreferences(place="  Indiranagar  ", cuisines=["  Chinese  ", "Cafe "])
    assert prefs.place == "Indiranagar"
    assert prefs.cuisines == ["Chinese", "Cafe"]


# --- normalize_preferences: matching against known values --------------------


KNOWN_PLACES = {"Indiranagar", "Koramangala", "HSR"}
KNOWN_CUISINES = {"Chinese", "North Indian", "Cafe"}


def test_normalize_matches_exact_case():
    prefs = normalize_preferences(
        "Indiranagar", ["Chinese"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
    )
    assert prefs.place == "Indiranagar"
    assert prefs.cuisines == ["Chinese"]


def test_normalize_matches_case_insensitively_and_maps_to_canonical_casing():
    prefs = normalize_preferences(
        "indiranagar", ["chinese"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
    )
    assert prefs.place == "Indiranagar"
    assert prefs.cuisines == ["Chinese"]


def test_normalize_matches_multiple_cuisines():
    prefs = normalize_preferences(
        "Indiranagar", ["chinese", "cafe"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
    )
    assert prefs.cuisines == ["Chinese", "Cafe"]


def test_normalize_accepts_missing_price_and_rating():
    prefs = normalize_preferences(
        "Indiranagar", ["Chinese"], None, None, KNOWN_PLACES, KNOWN_CUISINES
    )
    assert prefs.max_price is None
    assert prefs.min_rating is None


def test_normalize_rejects_unknown_place():
    with pytest.raises(UnknownValueError, match="Unknown place"):
        normalize_preferences(
            "Atlantis", ["Chinese"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
        )


def test_normalize_rejects_unknown_cuisine():
    with pytest.raises(UnknownValueError, match="Unknown cuisine"):
        normalize_preferences(
            "Indiranagar", ["Klingon"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
        )


def test_normalize_rejects_if_any_cuisine_in_list_is_unknown():
    with pytest.raises(UnknownValueError, match="Unknown cuisine"):
        normalize_preferences(
            "Indiranagar", ["Chinese", "Klingon"], 800, 4.0, KNOWN_PLACES, KNOWN_CUISINES
        )


def test_normalize_still_enforces_range_validation():
    with pytest.raises(ValidationError):
        normalize_preferences(
            "Indiranagar", ["Chinese"], -50, 4.0, KNOWN_PLACES, KNOWN_CUISINES
        )


# --- Integration tests against the live database ------------------------------


@pytest.fixture(scope="module")
def known_places():
    from known_values import get_known_places

    return get_known_places()


@pytest.fixture(scope="module")
def known_cuisines():
    from known_values import get_known_cuisines

    return get_known_cuisines()


def test_known_places_is_non_empty_and_contains_indiranagar(known_places):
    assert len(known_places) > 0
    assert "Indiranagar" in known_places


def test_known_cuisines_is_non_empty_and_contains_chinese(known_cuisines):
    assert len(known_cuisines) > 0
    assert "Chinese" in known_cuisines


def test_normalize_preferences_end_to_end_against_live_data(known_places, known_cuisines):
    prefs = normalize_preferences(
        "indiranagar", ["chinese"], 800, 4.0, known_places, known_cuisines
    )
    assert prefs.place == "Indiranagar"
    assert prefs.cuisines == ["Chinese"]


def test_normalize_preferences_rejects_unknown_place_against_live_data(
    known_places, known_cuisines
):
    with pytest.raises(UnknownValueError):
        normalize_preferences(
            "Nowhereville", ["Chinese"], 800, 4.0, known_places, known_cuisines
        )
