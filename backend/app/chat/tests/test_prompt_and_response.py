"""
Tests for Chat Prompt Construction & Response Formatting.

Pure/unit - no DB, no network. Uses lightweight fake candidate/snippet
objects (duck-typed the same shape as the retrieval engine's
RestaurantCandidate/ReviewSnippet) so this module is testable in isolation
from retrieval.
"""

from dataclasses import dataclass, field

from app.chat.prompt_builder import build_chat_prompt
from app.chat.response_formatter import format_chat_reply
from app.query_understanding.understanding import QueryUnderstanding


@dataclass
class FakeSnippet:
    text: str
    rating: float | None
    similarity: float = 1.0
    id: int = 1


@dataclass
class FakeCandidate:
    id: int
    name: str
    place: str = "Indiranagar"
    city: str = "Indiranagar"
    cuisines: list[str] = field(default_factory=lambda: ["Chinese"])
    price: float = 500.0
    rating: float = 4.5
    rest_type: str | None = "Casual Dining"
    votes: int = 100
    review_snippets: list = field(default_factory=list)


def make_candidate(name="Zzyzx Noodle Palace", snippets=None):
    return FakeCandidate(id=1, name=name, review_snippets=snippets or [])


# --- chat_prompt_builder --------------------------------------------------


def test_system_prompt_is_first_message():
    understanding = QueryUnderstanding(intent="search")
    messages = build_chat_prompt("suggest something", understanding, [], False, [], {})
    assert messages[0]["role"] == "system"
    assert "recommend" in messages[0]["content"].lower()


def test_prior_turns_pass_through_as_native_roles():
    understanding = QueryUnderstanding(intent="search")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]
    messages = build_chat_prompt("suggest something", understanding, [], False, history, {})

    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello!"}


def test_candidate_block_includes_name_and_facts():
    understanding = QueryUnderstanding(intent="search")
    candidate = make_candidate()
    messages = build_chat_prompt("suggest something", understanding, [candidate], False, [], {})

    final = messages[-1]["content"]
    assert "Zzyzx Noodle Palace" in final
    assert "Chinese" in final
    assert "Rs 500" in final
    assert "4.5" in final


def test_review_snippets_included_when_present():
    understanding = QueryUnderstanding(intent="search")
    candidate = make_candidate(snippets=[FakeSnippet(text="Great quiet ambience", rating=4.0)])
    messages = build_chat_prompt("quiet place?", understanding, [candidate], False, [], {})

    assert "Great quiet ambience" in messages[-1]["content"]


def test_relaxed_note_included_when_relaxed():
    understanding = QueryUnderstanding(intent="search")
    candidate = make_candidate()
    messages = build_chat_prompt("suggest something", understanding, [candidate], True, [], {})
    assert "relaxed" in messages[-1]["content"].lower()


def test_no_candidates_notes_none_matched_for_search_intent():
    understanding = QueryUnderstanding(intent="search")
    messages = build_chat_prompt("suggest something", understanding, [], False, [], {})
    assert "none matched" in messages[-1]["content"].lower()


def test_preferences_included_when_present():
    understanding = QueryUnderstanding(intent="search")
    messages = build_chat_prompt(
        "suggest something", understanding, [], False, [], {"dietary": "vegetarian"}
    )
    assert "vegetarian" in messages[-1]["content"].lower()


def test_referenced_restaurant_used_instead_of_candidate_list():
    understanding = QueryUnderstanding(intent="followup_question", refers_to_previous_restaurant=True)
    referenced = make_candidate(name="The Only Place")
    messages = build_chat_prompt(
        "what about its ambience?", understanding, [], False, [], {}, referenced_restaurant=referenced
    )
    assert "Restaurant being discussed" in messages[-1]["content"]
    assert "The Only Place" in messages[-1]["content"]


def test_chitchat_with_no_candidates_omits_none_matched_note():
    understanding = QueryUnderstanding(intent="chitchat")
    messages = build_chat_prompt("hi there", understanding, [], False, [], {})
    assert "none matched" not in messages[-1]["content"].lower()


# --- chat_response_formatter ----------------------------------------------


def test_format_chat_reply_matches_mentioned_restaurant():
    candidate = make_candidate(name="Zzyzx Noodle Palace")
    reply = format_chat_reply([candidate], "You should try Zzyzx Noodle Palace, it's great!")

    assert reply.mentioned_restaurant_ids == [1]
    assert len(reply.matched_restaurants) == 1
    assert reply.matched_restaurants[0].name == "Zzyzx Noodle Palace"


def test_format_chat_reply_excludes_unmentioned_candidates():
    mentioned = make_candidate(name="Mentioned Place")
    mentioned.id = 1
    unmentioned = make_candidate(name="Unmentioned Place")
    unmentioned.id = 2

    reply = format_chat_reply([mentioned, unmentioned], "Try Mentioned Place!")

    assert reply.mentioned_restaurant_ids == [1]


def test_format_chat_reply_carries_review_snippets_through():
    candidate = make_candidate(
        name="Zzyzx Noodle Palace", snippets=[FakeSnippet(text="cozy and quiet", rating=4.5)]
    )
    reply = format_chat_reply([candidate], "Zzyzx Noodle Palace is a good pick.")

    assert reply.matched_restaurants[0].review_snippets[0].text == "cozy and quiet"


def test_format_chat_reply_with_no_matches_returns_empty_lists():
    candidate = make_candidate(name="Some Place")
    reply = format_chat_reply([candidate], "I'm not sure what to recommend today.")

    assert reply.mentioned_restaurant_ids == []
    assert reply.matched_restaurants == []
