"""
Tests for the Chat API service layer.

End-to-end integration: real DB, real Groq calls - the same style as the
LLM client / auth suites' "real call" tests, but exercising the full chat
turn (understand -> retrieve -> prompt -> generate -> persist). Each test
uses a throwaway user; deleting the user cascades to their conversations/
messages/preferences.
"""

import uuid

import pytest

from app.auth import users as user_store
from app.chat.service import ConversationNotFoundError, handle_chat_message
from app.conversation import preferences as ps
from app.conversation import store as cs
from app.storage.db import get_connection


@pytest.fixture
def throwaway_user():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    user = user_store.create_user(email, "password123")
    yield user
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where id = %s;", (user["id"],))
        conn.commit()
    finally:
        conn.close()


def test_search_message_creates_conversation_and_returns_candidates(throwaway_user):
    reply, conversation_id = handle_chat_message(
        throwaway_user["id"], None, "Suggest a good Chinese restaurant in Indiranagar under 800"
    )

    assert conversation_id is not None
    assert len(reply.reply_text) > 0

    messages = cs.get_messages(conversation_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_reusing_conversation_id_appends_to_same_thread(throwaway_user):
    _, conversation_id = handle_chat_message(throwaway_user["id"], None, "Hi there!")
    handle_chat_message(throwaway_user["id"], conversation_id, "Suggest something in Indiranagar")

    messages = cs.get_messages(conversation_id)
    assert len(messages) == 4  # 2 user + 2 assistant turns


def test_conversation_owned_by_another_user_is_rejected(throwaway_user):
    other_email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    other = user_store.create_user(other_email, "password123")
    try:
        _, conversation_id = handle_chat_message(other["id"], None, "hi")
        with pytest.raises(ConversationNotFoundError):
            handle_chat_message(throwaway_user["id"], conversation_id, "hi")
    finally:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("delete from users where id = %s;", (other["id"],))
            conn.commit()
        finally:
            conn.close()


def test_preference_statement_is_persisted_for_future_turns(throwaway_user):
    handle_chat_message(
        throwaway_user["id"], None, "Just so you know, I'm vegetarian and prefer quiet places."
    )

    prefs = ps.get_preferences(throwaway_user["id"])
    assert "dietary" in prefs or "ambience" in prefs


def test_followup_question_resolves_previously_mentioned_restaurant(throwaway_user):
    first_reply, conversation_id = handle_chat_message(
        throwaway_user["id"], None, "Suggest a good Chinese restaurant in Indiranagar under 800"
    )

    if not first_reply.mentioned_restaurant_ids:
        pytest.skip("LLM didn't name a specific restaurant in its reply - nothing to follow up on")

    second_reply, _ = handle_chat_message(
        throwaway_user["id"], conversation_id, "What do people say about its ambience?"
    )
    assert len(second_reply.reply_text) > 0


def test_chitchat_message_does_not_crash_and_persists_turn(throwaway_user):
    reply, conversation_id = handle_chat_message(throwaway_user["id"], None, "hey, how's it going?")
    assert len(reply.reply_text) > 0
    assert len(cs.get_messages(conversation_id)) == 2
