"""
Tests for Phase 6 - Prompt Construction.

Unit tests build prompts from fake UserPreferences/RetrievalResult (no DB
needed). One integration test runs build_prompt on real output from
Phase 5's get_candidates() against the live database.
"""

import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE_DIR.parent / "phase4_preference_input"))
sys.path.insert(0, str(PHASE_DIR.parent / "phase5_retrieval_engine"))

from preferences import UserPreferences  # noqa: E402
from retrieval import RetrievalResult  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402

SAMPLE_CANDIDATES = [
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


def make_prefs(**overrides):
    defaults = dict(place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0)
    defaults.update(overrides)
    return UserPreferences(**defaults)


def test_build_prompt_returns_system_and_user_messages():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    messages = build_prompt(make_prefs(), result)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt_constrains_to_candidate_list_only():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    messages = build_prompt(make_prefs(), result)

    system_content = messages[0]["content"]
    assert "ONLY" in system_content
    assert "candidate list" in system_content


def test_user_message_includes_preferences():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    prefs = make_prefs(place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0)
    messages = build_prompt(prefs, result)

    user_content = messages[1]["content"]
    assert "Indiranagar" in user_content
    assert "Chinese" in user_content
    assert "800" in user_content
    assert "4.0" in user_content


def test_user_message_includes_multiple_cuisines():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    prefs = make_prefs(cuisines=["Chinese", "Cafe"])
    messages = build_prompt(prefs, result)

    assert "Chinese, Cafe" in messages[1]["content"]


def test_user_message_omits_price_and_rating_lines_when_not_given():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    prefs = make_prefs(max_price=None, min_rating=None)
    messages = build_prompt(prefs, result)

    user_content = messages[1]["content"]
    assert "Budget" not in user_content
    assert "Minimum rating" not in user_content


def test_user_message_includes_all_candidates_numbered_in_order():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    messages = build_prompt(make_prefs(), result)

    user_content = messages[1]["content"]
    idx_pot = user_content.index("1. Pot-O-Noodles")
    idx_mainland = user_content.index("2. Mainland China")
    assert idx_pot < idx_mainland
    assert "1700" in user_content
    assert "60 votes" in user_content


def test_empty_candidates_produces_placeholder_text():
    result = RetrievalResult(candidates=[], relaxed=True)
    messages = build_prompt(make_prefs(), result)

    user_content = messages[1]["content"]
    assert "no candidates found" in user_content


def test_relaxed_true_adds_relaxation_note():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=True)
    messages = build_prompt(make_prefs(), result)

    assert "relaxed" in messages[1]["content"].lower()


def test_relaxed_false_has_no_relaxation_note():
    result = RetrievalResult(candidates=SAMPLE_CANDIDATES, relaxed=False)
    messages = build_prompt(make_prefs(), result)

    assert "relaxed to still show" not in messages[1]["content"]


# --- Integration: real candidates from Phase 5 -----------------------------


def test_build_prompt_with_real_retrieval_result():
    from retrieval import get_candidates

    prefs = make_prefs(place="Indiranagar", cuisines=["Chinese"], max_price=800, min_rating=4.0)
    result = get_candidates(prefs, limit=5)

    messages = build_prompt(prefs, result)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Indiranagar" in messages[1]["content"]
    if result.candidates:
        assert result.candidates[0]["name"] in messages[1]["content"]
