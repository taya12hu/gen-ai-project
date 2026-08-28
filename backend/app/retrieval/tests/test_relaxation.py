"""
Tests for the constraint relaxation ladder.

Pure logic, no database: the ladder decides *what to try*, and the queries it
produces are exercised by test_hybrid.py against real data.
"""

from app.retrieval.relaxation import (
    MAX_LADDER_STEPS,
    PRICE_WIDEN_FACTOR,
    RATING_FLOOR,
    AppliedRelaxation,
    relaxation_ladder,
)


def test_no_constraints_produces_a_single_unrelaxed_rung():
    ladder = relaxation_ladder(None, None)
    assert len(ladder) == 1
    assert ladder[0].is_relaxed is False
    assert ladder[0].describe() is None


def test_first_rung_is_always_exactly_what_was_asked_for():
    ladder = relaxation_ladder(500, 4.5)
    first = ladder[0]
    assert (first.used_max_price, first.used_min_rating) == (500, 4.5)
    assert first.is_relaxed is False


def test_last_rung_always_drops_every_numeric_constraint():
    """The terminating rung. Without it a query could walk the whole ladder
    and still have nothing to fall back on."""
    last = relaxation_ladder(500, 4.5)[-1]
    assert last.used_max_price is None
    assert last.used_min_rating is None


def test_second_rung_moves_one_dimension_by_one_step():
    """Not two steps in one dimension, and not both dimensions at once - the
    smallest possible concession comes first."""
    second = relaxation_ladder(500, 4.5)[1]
    moved = [second.price_relaxed, second.rating_relaxed]
    assert moved.count(True) == 1
    assert second.used_max_price == 500 * PRICE_WIDEN_FACTOR


def test_price_only_query_never_touches_rating():
    for rung in relaxation_ladder(800, None):
        assert rung.used_min_rating is None
        assert rung.rating_relaxed is False


def test_rating_ladder_stops_at_the_floor_before_dropping_the_constraint():
    used = [r.used_min_rating for r in relaxation_ladder(None, 4.5)]
    assert None in used
    assert all(v is None or v >= RATING_FLOOR for v in used)


def test_ladder_is_capped_but_keeps_both_endpoints():
    ladder = relaxation_ladder(500, 4.5)
    assert len(ladder) <= MAX_LADDER_STEPS
    assert (ladder[0].used_max_price, ladder[0].used_min_rating) == (500, 4.5)
    assert (ladder[-1].used_max_price, ladder[-1].used_min_rating) == (None, None)


def test_ladder_has_no_duplicate_attempts():
    """Duplicates arise naturally when one dimension's ladder is shorter than
    the other's; each would cost a real round trip for a query already run."""
    ladder = relaxation_ladder(500, 4.5)
    seen = [(r.used_max_price, r.used_min_rating) for r in ladder]
    assert len(seen) == len(set(seen))


# --- describe(): what the reply is allowed to tell the user -------------------


def test_describe_is_none_when_nothing_was_relaxed():
    assert AppliedRelaxation(500, 4.0, 500, 4.0).describe() is None


def test_describe_names_the_specific_widened_budget():
    note = AppliedRelaxation(750, None, 500, None).describe()
    assert "750" in note and "500" in note


def test_describe_names_the_specific_lowered_rating():
    note = AppliedRelaxation(None, 4.0, None, 4.5).describe()
    assert "4.0" in note and "4.5" in note


def test_describe_covers_both_dimensions_when_both_moved():
    note = AppliedRelaxation(750, 4.0, 500, 4.5).describe()
    assert "750" in note and "4.0" in note


def test_dropping_a_constraint_entirely_is_described_as_such():
    note = AppliedRelaxation(None, None, 500, 4.5).describe()
    assert "no budget limit" in note
    assert "no rating floor" in note


def test_an_unrequested_constraint_is_never_reported_as_relaxed():
    """used_* is None here because the user never set one, not because
    anything was loosened."""
    relaxation = AppliedRelaxation(None, None, None, None)
    assert relaxation.is_relaxed is False
    assert relaxation.describe() is None
