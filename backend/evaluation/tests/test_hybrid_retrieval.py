"""
Tests for the hybrid (filters + vibe) retrieval evaluation.

Unlike the semantic suite, these ARE quality gates, and deliberately so. That
suite refuses to assert a recall threshold because matching a hand-picked id
out of 9,000 restaurants is close to a lottery - a fair judgement about a
fuzzy metric. Containment is not fuzzy. "Every restaurant returned for
'under Rs 800 in Koramangala' is in Koramangala and under Rs 800" is a
property the retriever either has or doesn't, so a violation here is a bug
rather than a bad draw, and asserting on it is right.

Runs against the live database. No Groq/Gemini call - the LLM judge only
enters via run_eval.py.
"""

import pytest

from app.retrieval.hybrid import clear_cache
from evaluation.hybrid_retrieval import HYBRID_SCENARIOS, HybridScenario, run_scenario


@pytest.fixture(scope="module", autouse=True)
def _cold_cache():
    """The retrieval TTL cache lives a week, so a previous run's results would
    otherwise be replayed instead of exercising the retriever."""
    clear_cache()


@pytest.mark.parametrize("scenario", HYBRID_SCENARIOS, ids=lambda s: s.label()[:60])
def test_scenario_holds_its_constraints(scenario: HybridScenario):
    """The whole point of the hybrid path: semantic ranking may reorder
    candidates but may never smuggle in one the filters excluded."""
    result = run_scenario(scenario)
    assert result.violations == [], "; ".join(result.violations)


def test_impossible_filters_do_not_silently_widen():
    """The dangerous failure mode, pinned separately because it is the one a
    user cannot detect: an unmatchable filter must return nothing rather than
    quietly degrading into an unfiltered vibe search, which would return
    plausible-looking restaurants that ignore what was actually asked."""
    scenario = HybridScenario(
        "Indiranagar", ("Klingon Cuisine",), None, None, "quiet", expect_found=False
    )
    assert run_scenario(scenario).candidates == []


def test_relaxation_is_reported_when_it_happens():
    """A relaxed search must say so, and say what moved - a reply that widens
    the budget silently is worse than one that finds nothing."""
    scenario = HybridScenario(
        "Kaggadasapura", ("Chinese",), 800, 4.9, "cosy and quiet",
        expect_found=True, expect_relaxed=True,
    )
    result = run_scenario(scenario)

    assert result.relaxed is True
    assert result.relaxation_note is not None
    assert "4.9" in result.relaxation_note  # names what was originally asked for


def test_unrelaxed_search_reports_no_relaxation_note():
    scenario = HybridScenario(
        "HSR", ("North Indian",), None, 4.0, "friendly and attentive staff", expect_found=True
    )
    result = run_scenario(scenario)

    assert result.relaxed is False
    assert result.relaxation_note is None


def test_every_returned_restaurant_has_review_evidence():
    """A vibe query asks a qualitative question; a candidate with no supporting
    review text cannot honestly answer it."""
    scenario = HybridScenario(
        "Whitefield", (), None, None, "quiet place away from crowds", expect_found=True
    )
    result = run_scenario(scenario)

    assert result.candidates
    assert result.evidence_rate == 1.0


def test_broad_pool_is_searchable_end_to_end():
    """Whitefield has 584 restaurants and used to exceed the 500-restaurant
    pool cap, which silently made its lowest-rated members unreachable. This
    pins that the broad-pool path works and stays reasonably fast, so a future
    change that reintroduces a cap - or falls back onto filtered HNSW's
    latency cliff - fails here."""
    scenario = HybridScenario(
        "Whitefield", (), None, None, "good value for money", expect_found=True
    )
    result = run_scenario(scenario)

    assert len(result.candidates) == 5
    assert result.violations == []
    # Generous: this is a cliff detector, not a benchmark. Filtered HNSW on
    # this pool measured 1.8s against 0.18s scanning, and a regression would
    # blow well past this.
    assert result.latency_seconds < 8.0
