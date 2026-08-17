"""
Tests for Query Understanding.

Prompt construction and the graceful-degradation path are pure/unit (no
network). The rest are real Groq integration tests (same pattern as the
LLM client's) confirming the model actually extracts the right structure -
intent, known-place/cuisine resolution, vibe text, reference resolution,
and durable preference statements - from real chat-shaped messages.
"""

from app.query_understanding import understanding as qu

KNOWN_PLACES = {"Indiranagar", "Koramangala", "Whitefield"}
KNOWN_CUISINES = {"Chinese", "North Indian", "South Indian", "Cafe", "Desserts"}


# --- prompt construction (no network) ---------------------------------------


def test_build_messages_includes_known_values_and_latest_message():
    messages = qu.build_messages("suggest a quiet place", [], KNOWN_PLACES, KNOWN_CUISINES)
    assert messages[0]["role"] == "system"
    assert "Indiranagar" in messages[0]["content"]
    assert "quiet place" in messages[1]["content"]


def test_build_messages_includes_history():
    history = [{"role": "user", "content": "I'm vegetarian"}, {"role": "assistant", "content": "Got it!"}]
    messages = qu.build_messages("suggest something", history, KNOWN_PLACES, KNOWN_CUISINES)
    assert "I'm vegetarian" in messages[1]["content"]
    assert "Got it!" in messages[1]["content"]


# --- graceful degradation on bad JSON (no network) --------------------------


def test_resolve_known_values_drops_unrecognized_place_and_cuisine():
    understanding = qu.QueryUnderstanding(place="Atlantis", cuisines=["Klingon", "Chinese"])
    resolved = qu._resolve_known_values(understanding, KNOWN_PLACES, KNOWN_CUISINES)
    assert resolved.place is None
    assert resolved.cuisines == ["Chinese"]


def test_resolve_known_values_is_case_insensitive():
    understanding = qu.QueryUnderstanding(place="indiranagar", cuisines=["chinese"])
    resolved = qu._resolve_known_values(understanding, KNOWN_PLACES, KNOWN_CUISINES)
    assert resolved.place == "Indiranagar"
    assert resolved.cuisines == ["Chinese"]


def test_clamp_relevant_preference_keys_drops_keys_not_in_input():
    """A hallucinated/malformed key here would otherwise flow straight into
    which stored preference gets applied to retrieval (see
    app.chat.service._effective_vibe_query) - it must never survive."""
    understanding = qu.QueryUnderstanding(relevant_preference_keys=["dietary", "made_up_key"])
    clamped = qu._clamp_relevant_preference_keys(understanding, {"dietary": "vegetarian"})
    assert clamped.relevant_preference_keys == ["dietary"]


def test_clamp_relevant_preference_keys_keeps_valid_keys_unchanged():
    understanding = qu.QueryUnderstanding(relevant_preference_keys=["dietary"])
    clamped = qu._clamp_relevant_preference_keys(understanding, {"dietary": "vegetarian", "ambience": "quiet"})
    assert clamped.relevant_preference_keys == ["dietary"]


# --- real Groq integration ---------------------------------------------------


def test_search_intent_extracts_place_cuisine_and_budget():
    result = qu.understand_query(
        "Suggest a good Chinese restaurant in Indiranagar under 800 rupees",
        [],
        KNOWN_PLACES,
        KNOWN_CUISINES,
    )
    assert result.intent == "search"
    assert result.place == "Indiranagar"
    assert "Chinese" in result.cuisines
    assert result.max_price is not None and result.max_price <= 900


def test_vibe_query_captures_qualitative_request():
    result = qu.understand_query(
        "Where should I go for a quiet date?", [], KNOWN_PLACES, KNOWN_CUISINES
    )
    assert result.intent == "search"
    assert result.vibe_query is not None
    assert "quiet" in result.vibe_query.lower() or "date" in result.vibe_query.lower()


def test_preference_statement_extracts_durable_facts_without_forcing_search():
    result = qu.understand_query(
        "Just so you know, I'm vegetarian and I usually prefer quiet places.",
        [],
        KNOWN_PLACES,
        KNOWN_CUISINES,
    )
    assert result.intent == "preference_statement"
    assert any("veget" in v.lower() for v in result.new_preferences.values())


def test_followup_question_refers_to_previous_restaurant():
    history = [
        {"role": "user", "content": "Suggest something in Indiranagar"},
        {"role": "assistant", "content": "I'd recommend Chinese Palace - great food and rating."},
    ]
    result = qu.understand_query(
        "What do people say about its ambience?", history, KNOWN_PLACES, KNOWN_CUISINES
    )
    assert result.intent == "followup_question"
    assert result.refers_to_previous_restaurant is True


def test_chitchat_intent_for_greeting():
    result = qu.understand_query("hey, how's it going?", [], KNOWN_PLACES, KNOWN_CUISINES)
    assert result.intent == "chitchat"


def test_relevant_preference_keys_includes_clearly_applicable_dietary_preference():
    result = qu.understand_query(
        "Suggest a good restaurant for dinner",
        [],
        KNOWN_PLACES,
        KNOWN_CUISINES,
        preferences={"dietary": "vegetarian"},
    )
    assert "dietary" in result.relevant_preference_keys


def test_relevant_preference_keys_excludes_unrelated_stored_preference():
    """Reproduces the real bug this feature fixes: a mood/ambience preference
    from an earlier conversation should not silently narrow an unrelated
    request just because it exists in storage."""
    result = qu.understand_query(
        "Suggest something for a quick solo lunch break",
        [],
        KNOWN_PLACES,
        KNOWN_CUISINES,
        preferences={"ambience": "quiet, romantic"},
    )
    assert "ambience" not in result.relevant_preference_keys
