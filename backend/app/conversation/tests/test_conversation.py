"""
Tests for Conversation & Personalization Storage.

Runs against the live database, using a throwaway user per test (the auth
domain's user store) so the suite is safely re-runnable. Deleting the user
cascades to their conversations/messages/preferences (FK ON DELETE CASCADE
in schema.sql), so a single cleanup step is enough.
"""

import uuid

import pytest

from app.auth import users as user_store
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


# --- conversations / messages ----------------------------------------------


def test_create_conversation_belongs_to_user(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    assert conversation["user_id"] == throwaway_user["id"]


def test_get_conversation_succeeds_for_owner(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    fetched = cs.get_conversation(conversation["id"], throwaway_user["id"])
    assert fetched["id"] == conversation["id"]


def test_get_conversation_raises_for_non_owner(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    with pytest.raises(cs.ConversationNotFoundError):
        cs.get_conversation(conversation["id"], throwaway_user["id"] + 999999)


def test_get_conversation_raises_for_unknown_id(throwaway_user):
    with pytest.raises(cs.ConversationNotFoundError):
        cs.get_conversation(-1, throwaway_user["id"])


def test_list_conversations_orders_most_recent_first(throwaway_user):
    first = cs.create_conversation(throwaway_user["id"])
    second = cs.create_conversation(throwaway_user["id"])

    listed = cs.list_conversations(throwaway_user["id"])
    ids = [c["id"] for c in listed]
    assert ids.index(second["id"]) < ids.index(first["id"])


def test_add_message_and_get_messages_round_trip(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    cs.add_message(conversation["id"], "user", "hello")
    cs.add_message(conversation["id"], "assistant", "hi there", mentioned_restaurant_ids=[1, 2])

    messages = cs.get_messages(conversation["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello"
    assert messages[1]["mentioned_restaurant_ids"] == [1, 2]


def test_get_messages_respects_limit_and_keeps_chronological_order(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    for i in range(5):
        cs.add_message(conversation["id"], "user", f"message {i}")

    messages = cs.get_messages(conversation["id"], limit=3)
    assert [m["content"] for m in messages] == ["message 2", "message 3", "message 4"]


def test_get_last_mentioned_restaurant_ids_returns_most_recent(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    cs.add_message(conversation["id"], "user", "suggest something")
    cs.add_message(conversation["id"], "assistant", "try A", mentioned_restaurant_ids=[10])
    cs.add_message(conversation["id"], "user", "what about ambience")
    cs.add_message(conversation["id"], "assistant", "no restaurant here", mentioned_restaurant_ids=None)

    assert cs.get_last_mentioned_restaurant_ids(conversation["id"]) == [10]


def test_get_last_mentioned_restaurant_ids_empty_when_none_mentioned(throwaway_user):
    conversation = cs.create_conversation(throwaway_user["id"])
    cs.add_message(conversation["id"], "user", "hi")
    assert cs.get_last_mentioned_restaurant_ids(conversation["id"]) == []


# --- preferences ------------------------------------------------------------


def test_get_preferences_empty_for_new_user(throwaway_user):
    assert ps.get_preferences(throwaway_user["id"]) == {}


def test_upsert_preferences_inserts_new_facts(throwaway_user):
    ps.upsert_preferences(throwaway_user["id"], {"dietary": "vegetarian", "ambience": "quiet"})
    prefs = ps.get_preferences(throwaway_user["id"])
    assert prefs == {"dietary": "vegetarian", "ambience": "quiet"}


def test_upsert_preferences_overwrites_existing_key(throwaway_user):
    ps.upsert_preferences(throwaway_user["id"], {"dietary": "vegetarian"})
    ps.upsert_preferences(throwaway_user["id"], {"dietary": "vegan"})

    prefs = ps.get_preferences(throwaway_user["id"])
    assert prefs["dietary"] == "vegan"


def test_upsert_preferences_with_empty_dict_is_noop(throwaway_user):
    ps.upsert_preferences(throwaway_user["id"], {"dietary": "vegetarian"})
    ps.upsert_preferences(throwaway_user["id"], {})

    assert ps.get_preferences(throwaway_user["id"]) == {"dietary": "vegetarian"}


def test_preferences_are_isolated_per_user(throwaway_user):
    other_email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    other = user_store.create_user(other_email, "password123")
    try:
        ps.upsert_preferences(throwaway_user["id"], {"dietary": "vegetarian"})
        assert ps.get_preferences(other["id"]) == {}
    finally:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("delete from users where id = %s;", (other["id"],))
            conn.commit()
        finally:
            conn.close()
