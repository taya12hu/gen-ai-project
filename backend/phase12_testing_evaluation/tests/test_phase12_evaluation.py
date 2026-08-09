"""
Phase 12 - Testing & Evaluation.

Runs the full real pipeline (Phase 4 -> 8) across a diverse set of actual
place+cuisine combinations pulled from the live data - different areas,
cuisines, price tiers, and rating thresholds - plus two deliberate edge
cases (a combo that forces the Phase 5 relaxation fallback, and a combo
with zero matches). For each scenario we check the two things Phase 12 is
meant to evaluate:

1. Grounding - the LLM only recommends restaurants that are actually in
   the candidate shortlist we gave it (no hallucination).
2. Relevance - what it recommends actually satisfies the requested place,
   cuisine, and (when not relaxed) price/rating bounds.

This complements the per-phase unit/integration tests already written in
Phases 1-10 by testing the pipeline as a whole across varied real inputs,
rather than one hardcoded case per phase. Auth (Phase 9) is tested
separately in Phase 10's API test suite; these scenarios call the pipeline
functions directly, bypassing the HTTP/auth layer intentionally, to keep
the focus on grounding/relevance rather than transport concerns.
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
for phase in [
    "phase4_preference_input",
    "phase5_retrieval_engine",
    "phase6_prompt_construction",
    "phase7_llm_engine",
    "phase8_response_formatting",
]:
    sys.path.insert(0, str(BACKEND_DIR / phase))

from preferences import UserPreferences  # noqa: E402
from retrieval import get_candidates  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402
from llm_client import get_recommendation  # noqa: E402
from response_formatter import format_response  # noqa: E402

# (place, cuisines, max_price, min_rating, expect_found, expect_relaxed)
# The first 6 are diverse real combinations spanning different areas, cuisines,
# and price tiers. The rest are deliberate edge cases: forced relaxation, zero
# matches, multiple cuisines (OR/overlap match), and omitted price/rating.
SCENARIOS = [
    ("Church Street", ["North Indian"], 1000, 4.0, True, False),
    ("Ulsoor", ["South Indian"], 1000, 3.0, True, False),
    ("Marathahalli", ["Desserts"], 500, 4.0, True, False),
    ("MG Road", ["North Indian"], 3000, 4.0, True, False),
    ("BTM", ["Biryani"], 900, 4.0, True, False),
    ("Whitefield", ["Fast Food"], 1000, 4.3, True, False),
    # Kaggadasapura+Chinese tops out at rating 4.0 - a 4.9 floor can't be met strictly.
    ("Kaggadasapura", ["Chinese"], 800, 4.9, True, True),
    # Both values individually valid, but this exact combination has 0 rows.
    ("Banashankari", ["Lucknowi"], 1000, 0, False, None),
    # Multiple cuisines - matches a restaurant serving either one (OR/overlap).
    ("Indiranagar", ["Chinese", "Cafe"], 2000, 0, True, False),
    # Price and rating both omitted - should return results, nothing to relax.
    ("Indiranagar", ["Chinese"], None, None, True, False),
]


def run_scenario(place, cuisines, max_price, min_rating):
    prefs = UserPreferences(
        place=place, cuisines=cuisines, max_price=max_price, min_rating=min_rating
    )
    result = get_candidates(prefs, limit=5)

    if result.candidates:
        messages = build_prompt(prefs, result)
        llm_text = get_recommendation(messages)
    else:
        llm_text = "No matching restaurants were found for your preferences."

    return prefs, result, format_response(result, llm_text)


@pytest.mark.parametrize(
    "place, cuisines, max_price, min_rating, expect_found, expect_relaxed", SCENARIOS
)
def test_scenario_grounding_and_relevance(
    place, cuisines, max_price, min_rating, expect_found, expect_relaxed
):
    prefs, result, rec = run_scenario(place, cuisines, max_price, min_rating)

    assert rec.found_any is expect_found
    if expect_relaxed is not None:
        assert rec.relaxed is expect_relaxed

    if not expect_found:
        return

    # Grounding: the LLM actually engaged with the data we gave it - at least
    # one restaurant it talks about is one we actually offered it.
    assert len(rec.matched_restaurants) > 0

    # Relevance: everything it recommended genuinely fits the request.
    for r in rec.matched_restaurants:
        assert r["place"] == prefs.place
        assert any(c in r["cuisines"] for c in prefs.cuisines)
        if not rec.relaxed:
            if prefs.max_price is not None:
                assert r["price"] <= prefs.max_price
            if prefs.min_rating is not None:
                assert r["rating"] >= prefs.min_rating

    # Every matched restaurant must actually be one of the candidates we sent -
    # not something the model invented that happens to share a name pattern.
    candidate_ids = {c["id"] for c in result.candidates}
    for r in rec.matched_restaurants:
        assert r["id"] in candidate_ids
