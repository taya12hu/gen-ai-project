"""
Tests for Phase 8 - Response Formatting & Output.

Unit tests use fake RetrievalResult + canned LLM text (no DB/LLM needed).
One integration test runs the full real pipeline (Phase 4-8) end to end.
"""

import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE_DIR.parent / "phase4_preference_input"))
sys.path.insert(0, str(PHASE_DIR.parent / "phase5_retrieval_engine"))
sys.path.insert(0, str(PHASE_DIR.parent / "phase6_prompt_construction"))
sys.path.insert(0, str(PHASE_DIR.parent / "phase7_llm_engine"))

from retrieval import RetrievalResult  # noqa: E402
from response_formatter import format_response, format_for_display  # noqa: E402

CANDIDATES = [
    {
        "id": 1,
        "name": "Pot-O-Noodles",
        "place": "Indiranagar",
        "city": "Indiranagar",
        "cuisines": ["Chinese", "Asian", "Japanese"],
        "price": 800.0,
        "rating": 4.5,
        "rest_type": "Casual Dining",
        "votes": 60,
    },
    {
        "id": 2,
        "name": "Mainland China",
        "place": "Indiranagar",
        "city": "Indiranagar",
        "cuisines": ["Chinese"],
        "price": 1700.0,
        "rating": 4.4,
        "rest_type": "Fine Dining",
        "votes": 1507,
    },
]


# --- format_response -----------------------------------------------------


def test_matches_restaurant_mentioned_in_llm_text():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    llm_text = "I recommend Pot-O-Noodles because it fits your budget and rating."

    rec = format_response(result, llm_text)

    assert len(rec.matched_restaurants) == 1
    assert rec.matched_restaurants[0]["name"] == "Pot-O-Noodles"


def test_matches_multiple_restaurants_mentioned_in_llm_text():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    llm_text = "Both Pot-O-Noodles and Mainland China are great picks."

    rec = format_response(result, llm_text)

    names = {r["name"] for r in rec.matched_restaurants}
    assert names == {"Pot-O-Noodles", "Mainland China"}


def test_no_match_when_llm_text_mentions_no_candidate_name():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    llm_text = "Unfortunately none of these quite work for you."

    rec = format_response(result, llm_text)

    assert rec.matched_restaurants == []
    assert rec.explanation == llm_text


def test_found_any_false_when_no_candidates():
    result = RetrievalResult(candidates=[], relaxed=True)
    rec = format_response(result, "No matches found.")

    assert rec.found_any is False
    assert rec.matched_restaurants == []


def test_found_any_true_when_candidates_exist():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    rec = format_response(result, "Pot-O-Noodles is a great pick.")

    assert rec.found_any is True


def test_relaxed_flag_passed_through():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=True)
    rec = format_response(result, "Pot-O-Noodles is close to what you wanted.")

    assert rec.relaxed is True


def test_explanation_is_stripped():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    rec = format_response(result, "  Pot-O-Noodles is great.  \n")

    assert rec.explanation == "Pot-O-Noodles is great."


# --- format_for_display ---------------------------------------------------


def test_display_shows_fallback_message_when_nothing_found():
    result = RetrievalResult(candidates=[], relaxed=True)
    rec = format_response(result, "irrelevant")

    display = format_for_display(rec)

    assert "No restaurants matched" in display


def test_display_includes_restaurant_facts_and_explanation():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    rec = format_response(result, "Pot-O-Noodles is the best fit for your Chinese craving.")

    display = format_for_display(rec)

    assert "Pot-O-Noodles" in display
    assert "Indiranagar" in display
    assert "Chinese" in display
    assert "800" in display
    assert "4.5" in display
    assert "60 votes" in display
    assert "Pot-O-Noodles is the best fit for your Chinese craving." in display


def test_display_includes_relaxed_note_when_relaxed():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=True)
    rec = format_response(result, "Pot-O-Noodles is the closest match.")

    display = format_for_display(rec)

    assert "closest matches" in display


def test_display_omits_relaxed_note_when_not_relaxed():
    result = RetrievalResult(candidates=CANDIDATES, relaxed=False)
    rec = format_response(result, "Pot-O-Noodles is a great fit.")

    display = format_for_display(rec)

    assert "closest matches" not in display


# --- Integration: full real pipeline ---------------------------------------


def test_full_pipeline_end_to_end():
    from preferences import UserPreferences
    from retrieval import get_candidates
    from prompt_builder import build_prompt
    from llm_client import get_recommendation

    prefs = UserPreferences(place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0)
    result = get_candidates(prefs, limit=5)
    messages = build_prompt(prefs, result)
    llm_text = get_recommendation(messages)

    rec = format_response(result, llm_text)
    display = format_for_display(rec)

    assert rec.found_any is True
    assert len(display) > 0
    if rec.matched_restaurants:
        assert rec.matched_restaurants[0]["name"] in display
