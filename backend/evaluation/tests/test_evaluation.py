"""
Chat Pipeline Evaluation.

Runs the live retrieval + generation pipeline (app.retrieval.hybrid ->
app.chat.prompt_builder -> Groq -> app.chat.response_formatter) across a
diverse set of real place+cuisine combinations pulled from the live data -
different areas, cuisines, price tiers, and rating thresholds - plus
deliberate edge cases (a combo that forces relaxation, and a combo with
zero matches). For each scenario we check the two things that matter most
for an LLM-backed recommendation service:

1. Grounding - the LLM only talks about restaurants that were actually in
   the candidate shortlist it was given (no hallucination).
2. Relevance - what it recommends actually satisfies the requested place,
   cuisine, and (when not relaxed) price/rating bounds.

This complements the per-domain unit/integration tests elsewhere in app/ by
testing the pipeline as a whole across varied real inputs, rather than one
hardcoded case per module. Query understanding's own NL-parsing accuracy is
tested separately (app/query_understanding/tests) - these scenarios build
HybridFilters directly and skip that stage, to keep the focus on
grounding/relevance rather than intent extraction.
"""

import pytest

from app.chat.prompt_builder import build_chat_prompt
from app.chat.response_formatter import format_chat_reply
from app.llm.groq_client import get_recommendation
from app.query_understanding.understanding import QueryUnderstanding
from app.retrieval.hybrid import HybridFilters, get_hybrid_candidates

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
    filters = HybridFilters(place=place, cuisines=cuisines, max_price=max_price, min_rating=min_rating)
    result = get_hybrid_candidates(filters, vibe_query=None, limit=5)

    understanding = QueryUnderstanding(intent="search", place=place, cuisines=cuisines)
    user_message = f"Suggest a {', '.join(cuisines)} restaurant in {place}"

    if result.candidates:
        prompt = build_chat_prompt(
            user_message, understanding, result.candidates, result.relaxation_note(), [], {}
        )
        llm_text = get_recommendation(prompt)
    else:
        llm_text = "No matching restaurants were found for your preferences."

    reply = format_chat_reply(result.candidates, llm_text)
    return filters, result, reply


@pytest.mark.parametrize(
    "place, cuisines, max_price, min_rating, expect_found, expect_relaxed", SCENARIOS
)
def test_scenario_grounding_and_relevance(
    place, cuisines, max_price, min_rating, expect_found, expect_relaxed
):
    filters, result, reply = run_scenario(place, cuisines, max_price, min_rating)

    found_any = len(result.candidates) > 0
    assert found_any is expect_found
    if expect_relaxed is not None:
        assert result.relaxed is expect_relaxed

    if not expect_found:
        return

    # Grounding: the LLM actually engaged with the data we gave it - at least
    # one restaurant it talks about is one we actually offered it.
    assert len(reply.matched_restaurants) > 0

    # Relevance: everything it recommended genuinely fits the request.
    for r in reply.matched_restaurants:
        assert r.place == filters.place
        assert any(c in r.cuisines for c in filters.cuisines)
        if not result.relaxed:
            if filters.max_price is not None:
                assert r.price <= filters.max_price
            if filters.min_rating is not None:
                assert r.rating >= filters.min_rating

    # Every matched restaurant must actually be one of the candidates we sent -
    # not something the model invented that happens to share a name pattern.
    candidate_ids = {c.id for c in result.candidates}
    for r in reply.matched_restaurants:
        assert r.id in candidate_ids
