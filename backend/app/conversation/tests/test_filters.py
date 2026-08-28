"""
Tests for conversation search state.

Pure merge logic, no database. This is where "what about somewhere cheaper?"
either keeps the area you were searching in or silently loses it, so the three
cases - set, clear, leave alone - are pinned individually.
"""

import pytest

from app.conversation.filters import FILTER_DIMENSIONS, SearchState, merge
from app.query_understanding.understanding import QueryUnderstanding


def _u(**kwargs) -> QueryUnderstanding:
    return QueryUnderstanding(intent="search", **kwargs)


# --- set ----------------------------------------------------------------------


def test_a_mentioned_value_sets_the_dimension():
    state = merge(SearchState(), _u(place="Indiranagar", cuisines=["Chinese"], max_price=800.0))
    assert state.place == "Indiranagar"
    assert state.cuisines == ("Chinese",)
    assert state.max_price == 800.0


def test_a_new_value_replaces_the_old_one():
    """"Actually make it Koramangala" - no need to clear first."""
    state = merge(SearchState(place="Indiranagar"), _u(place="Koramangala"))
    assert state.place == "Koramangala"


# --- leave alone --------------------------------------------------------------


def test_silence_leaves_a_constraint_in_place():
    """The behaviour that makes constraints persist at all: a message that
    doesn't mention the budget is not a request to drop it."""
    state = merge(SearchState(place="Indiranagar", max_price=800.0), _u(cuisines=["Chinese"]))
    assert state.place == "Indiranagar"
    assert state.max_price == 800.0
    assert state.cuisines == ("Chinese",)


def test_an_empty_understanding_changes_nothing():
    before = SearchState(place="HSR", cuisines=("Chinese",), max_price=500.0, min_rating=4.0)
    assert merge(before, _u()) == before


# --- clear --------------------------------------------------------------------


def test_an_explicit_clear_drops_the_dimension():
    """"Anywhere, actually" - the case that cannot be inferred from silence,
    which is why the model is asked for it explicitly."""
    state = merge(SearchState(place="Indiranagar", max_price=800.0), _u(), cleared=["place"])
    assert state.place is None
    assert state.max_price == 800.0


def test_clearing_several_dimensions_at_once():
    before = SearchState(place="BTM", cuisines=("Chinese",), max_price=500.0, min_rating=4.0)
    state = merge(before, _u(), cleared=["price", "rating"])
    assert (state.max_price, state.min_rating) == (None, None)
    assert (state.place, state.cuisines) == ("BTM", ("Chinese",))


def test_a_value_wins_over_a_clear_for_the_same_dimension():
    """"Anywhere, but make it Chinese" mentions both; the mention is the more
    specific instruction."""
    state = merge(SearchState(), _u(cuisines=["Chinese"]), cleared=["cuisines"])
    assert state.cuisines == ("Chinese",)


def test_unknown_dimension_names_are_ignored():
    """A hallucinated name must not silently fail to clear anything - the user
    would believe a constraint was removed while it kept applying."""
    before = SearchState(place="Indiranagar", max_price=800.0)
    assert merge(before, _u(), cleared=["budget", "location"]) == before


# --- display ------------------------------------------------------------------


def test_chips_describe_constraints_in_the_user_s_terms():
    chips = SearchState(place="Indiranagar", max_price=800.0, min_rating=4.0).as_chips()
    labels = {c["dimension"]: c["label"] for c in chips}
    assert labels["place"] == "Indiranagar"
    assert labels["price"] == "under Rs 800"
    assert labels["rating"] == "4.0+ stars"


def test_multiple_cuisines_are_one_chip():
    """One chip per dimension, so removing it removes the whole constraint -
    matching what clearing "cuisines" actually does."""
    chips = SearchState(cuisines=("Chinese", "Thai")).as_chips()
    assert len(chips) == 1
    assert chips[0]["label"] == "Chinese, Thai"


def test_empty_state_has_no_chips():
    assert SearchState().as_chips() == []
    assert SearchState().is_empty() is True


def test_every_dimension_can_be_cleared_by_name():
    """FILTER_DIMENSIONS is a shared vocabulary - the model emits these names
    and the frontend addresses them - so each must actually work."""
    full = SearchState(place="BTM", cuisines=("Chinese",), max_price=500.0, min_rating=4.0)
    for dimension in FILTER_DIMENSIONS:
        assert full.cleared(dimension) != full
    assert full.cleared("place").cleared("cuisines").cleared("price").cleared("rating").is_empty()


def test_clearing_an_unknown_dimension_raises():
    with pytest.raises(ValueError):
        SearchState().cleared("budget")


# --- round trip ----------------------------------------------------------------


def test_state_survives_a_json_round_trip():
    before = SearchState(place="HSR", cuisines=("Chinese", "Thai"), max_price=500.0, min_rating=4.0)
    assert SearchState.from_json(before.to_json()) == before


def test_missing_or_empty_stored_state_reads_as_unconstrained():
    assert SearchState.from_json(None).is_empty()
    assert SearchState.from_json({}).is_empty()
