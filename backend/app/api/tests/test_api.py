"""
Tests for the Application / Interface Layer (backend API).

Uses FastAPI's TestClient against the real app, which in turn hits the
live database, Groq API, and the auth domain's logic - this exercises the
full wired-together pipeline through the actual HTTP interface.
"""

import os
import uuid

import psycopg2
import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api.main import app
from app.auth import password_reset
from app.conversation import filters
from app.storage.db import get_connection

client = TestClient(app)


@pytest.fixture
def auth_headers():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post(
        "/auth/register", json={"email": email, "password": "password123", "display_name": "Test User"}
    )
    token = response.json()["access_token"]

    yield {"Authorization": f"Bearer {token}"}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


# --- Health --------------------------------------------------------------------


def test_health_check_reports_database_reachability():
    """The platform uses this as its healthCheckPath, so it has to be able to
    fail. It previously returned {"status": "ok"} unconditionally, which meant
    an instance with an unreachable database kept being sent traffic it could
    only fail."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_check_reports_503_when_the_database_is_unreachable(monkeypatch):
    def boom():
        raise psycopg2.OperationalError("connection refused")

    monkeypatch.setattr(main, "get_connection", boom)

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


# --- Auth: register / login / me ----------------------------------------------


def test_register_creates_user_and_returns_token():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post("/auth/register", json={"email": email, "password": "password123"})

    assert response.status_code == 201
    data = response.json()
    assert data["user"]["email"] == email
    assert len(data["access_token"]) > 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


def test_register_duplicate_email_returns_409(auth_headers):
    # auth_headers fixture already registered a user; grab its email via /auth/me
    me = client.get("/auth/me", headers=auth_headers).json()

    response = client.post("/auth/register", json={"email": me["email"], "password": "whatever123"})
    assert response.status_code == 409


def test_register_short_password_returns_422():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post("/auth/register", json={"email": email, "password": "short"})
    assert response.status_code == 422


def test_register_invalid_email_returns_422():
    response = client.post("/auth/register", json={"email": "not-an-email", "password": "password123"})
    assert response.status_code == 422


def test_login_succeeds_with_correct_credentials():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "password123"})

    response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    assert response.json()["user"]["email"] == email

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


def test_login_fails_with_wrong_password(auth_headers):
    me = client.get("/auth/me", headers=auth_headers).json()

    response = client.post("/auth/login", json={"email": me["email"], "password": "wrong-password"})
    assert response.status_code == 401


def test_login_fails_for_unregistered_email():
    response = client.post(
        "/auth/login", json={"email": "nobody-registered@example.com", "password": "whatever123"}
    )
    assert response.status_code == 401


def test_me_returns_current_user(auth_headers):
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert "email" in response.json()


def test_me_without_token_returns_401():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_with_invalid_token_returns_401():
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


# --- Auth: forgot / reset password ---------------------------------------------

RESEND_CONFIGURED = bool(os.environ.get("RESEND_API_KEY"))


def test_forgot_password_for_unknown_email_returns_generic_message():
    # No RESEND_API_KEY needed here - an unregistered email returns before
    # the email service would ever be called (see forgot_password in api.py).
    response = client.post("/auth/forgot-password", json={"email": "nobody-registered@example.com"})
    assert response.status_code == 200
    assert "If an account with that email exists" in response.json()["message"]


@pytest.mark.skipif(not RESEND_CONFIGURED, reason="RESEND_API_KEY not configured in this environment")
def test_forgot_password_for_known_email_creates_a_reset_token(auth_headers):
    me = client.get("/auth/me", headers=auth_headers).json()

    response = client.post("/auth/forgot-password", json={"email": me["email"]})
    assert response.status_code == 200
    assert "If an account with that email exists" in response.json()["message"]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from password_reset_tokens where user_id = %s and used_at is null;",
                (me["id"],),
            )
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def test_reset_password_with_invalid_token_returns_400():
    response = client.post(
        "/auth/reset-password", json={"token": "not-a-real-token", "new_password": "newpassword123"}
    )
    assert response.status_code == 400


def test_reset_password_updates_password_and_allows_login(auth_headers):
    # Issues the token directly (bypassing /auth/forgot-password) so this test
    # doesn't depend on RESEND_API_KEY being configured - it's exercising the
    # reset endpoint itself, not email delivery.
    me = client.get("/auth/me", headers=auth_headers).json()
    token = password_reset.create_reset_token(me["id"])

    response = client.post(
        "/auth/reset-password", json={"token": token, "new_password": "brand-new-password123"}
    )
    assert response.status_code == 200

    login_response = client.post(
        "/auth/login", json={"email": me["email"], "password": "brand-new-password123"}
    )
    assert login_response.status_code == 200


def test_reset_password_token_cannot_be_reused(auth_headers):
    me = client.get("/auth/me", headers=auth_headers).json()
    token = password_reset.create_reset_token(me["id"])

    first = client.post("/auth/reset-password", json={"token": token, "new_password": "first-new-pass123"})
    assert first.status_code == 200

    second = client.post("/auth/reset-password", json={"token": token, "new_password": "second-new-pass123"})
    assert second.status_code == 400


def test_cors_headers_present_for_allowed_origin():
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --- Chat routes -----------------------------------------------------------------


def test_create_conversation_requires_auth():
    response = client.post("/chat/conversations")
    assert response.status_code == 401


def test_create_conversation_returns_201(auth_headers):
    response = client.post("/chat/conversations", headers=auth_headers)
    assert response.status_code == 201
    assert "id" in response.json()


def test_list_conversations_only_shows_own(auth_headers):
    created = client.post("/chat/conversations", headers=auth_headers).json()

    response = client.get("/chat/conversations", headers=auth_headers)
    assert response.status_code == 200
    ids = [c["id"] for c in response.json()]
    assert created["id"] in ids


def test_get_messages_for_unknown_conversation_returns_404(auth_headers):
    response = client.get("/chat/conversations/999999999/messages", headers=auth_headers)
    assert response.status_code == 404


def test_send_message_to_unknown_conversation_returns_404(auth_headers):
    response = client.post(
        "/chat/conversations/999999999/messages", json={"message": "hi"}, headers=auth_headers
    )
    assert response.status_code == 404


def test_send_message_without_auth_returns_401():
    response = client.post("/chat/conversations/1/messages", json={"message": "hi"})
    assert response.status_code == 401


def test_send_message_requires_nonempty_message(auth_headers):
    created = client.post("/chat/conversations", headers=auth_headers).json()
    response = client.post(
        f"/chat/conversations/{created['id']}/messages", json={"message": ""}, headers=auth_headers
    )
    assert response.status_code == 422


def test_full_chat_turn_returns_reply_and_persists_history(auth_headers):
    created = client.post("/chat/conversations", headers=auth_headers).json()

    response = client.post(
        f"/chat/conversations/{created['id']}/messages",
        json={"message": "Suggest a good Chinese restaurant in Indiranagar under 800"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == created["id"]
    assert len(data["reply"]) > 0

    history = client.get(f"/chat/conversations/{created['id']}/messages", headers=auth_headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_second_users_conversation_is_not_visible_to_first(auth_headers):
    other_email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    other_response = client.post(
        "/auth/register", json={"email": other_email, "password": "password123"}
    )
    other_headers = {"Authorization": f"Bearer {other_response.json()['access_token']}"}

    try:
        other_conversation = client.post("/chat/conversations", headers=other_headers).json()

        response = client.get(
            f"/chat/conversations/{other_conversation['id']}/messages", headers=auth_headers
        )
        assert response.status_code == 404
    finally:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("delete from users where email = %s;", (other_email,))
            conn.commit()
        finally:
            conn.close()


def test_recommend_endpoint_no_longer_exists():
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese"]},
    )
    assert response.status_code == 404


def test_options_endpoint_no_longer_exists():
    response = client.get("/options")
    assert response.status_code == 404


# --- Conversation filters -----------------------------------------------------
#
# The structured constraints in force for a conversation, exposed so the UI can
# show them as chips and remove them individually. Before they were held
# explicitly, they lived only in the transcript - invisible to the user and
# re-inferred by a model on every turn.


def test_new_conversation_starts_with_no_filters(auth_headers):
    conversation_id = client.post("/chat/conversations", headers=auth_headers).json()["id"]
    response = client.get(f"/chat/conversations/{conversation_id}/filters", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_filters_require_authentication():
    conversation_id = 1
    assert client.get(f"/chat/conversations/{conversation_id}/filters").status_code == 401
    assert client.delete(f"/chat/conversations/{conversation_id}/filters/place").status_code == 401


def test_filters_of_another_users_conversation_are_not_readable(auth_headers):
    """Same ownership rule as messages - a conversation id is not a capability."""
    other = client.post(
        "/auth/register",
        json={"email": f"test-{uuid.uuid4().hex[:12]}@example.com", "password": "password123"},
    ).json()
    other_headers = {"Authorization": f"Bearer {other['access_token']}"}
    conversation_id = client.post("/chat/conversations", headers=other_headers).json()["id"]

    response = client.get(f"/chat/conversations/{conversation_id}/filters", headers=auth_headers)
    assert response.status_code == 404


def test_clearing_an_unknown_dimension_is_rejected(auth_headers):
    """The dimension name is a shared vocabulary between model, API and UI;
    an unrecognized one must fail loudly rather than silently clearing
    nothing and leaving the user believing a constraint was removed."""
    conversation_id = client.post("/chat/conversations", headers=auth_headers).json()["id"]
    response = client.delete(
        f"/chat/conversations/{conversation_id}/filters/budget", headers=auth_headers
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "unknown_filter"


def test_clearing_a_filter_returns_what_remains(auth_headers):
    """Returns the surviving chips rather than 204, so the UI updates from an
    authoritative list instead of guessing what removal left behind."""
    conversation_id = client.post("/chat/conversations", headers=auth_headers).json()["id"]
    filters.save_state(
        conversation_id,
        filters.SearchState(place="Indiranagar", cuisines=("Chinese",), max_price=800.0),
    )

    response = client.delete(
        f"/chat/conversations/{conversation_id}/filters/place", headers=auth_headers
    )

    assert response.status_code == 200
    dimensions = {chip["dimension"] for chip in response.json()}
    assert dimensions == {"cuisines", "price"}


def test_clearing_a_filter_that_was_not_set_is_harmless(auth_headers):
    """Removing a chip the UI has already dropped shouldn't 404 - the end
    state the user wanted is the one they have."""
    conversation_id = client.post("/chat/conversations", headers=auth_headers).json()["id"]
    filters.save_state(conversation_id, filters.SearchState(place="Indiranagar"))

    response = client.delete(
        f"/chat/conversations/{conversation_id}/filters/rating", headers=auth_headers
    )

    assert response.status_code == 200
    assert [chip["dimension"] for chip in response.json()] == ["place"]


def test_chip_labels_are_written_for_display(auth_headers):
    """The backend owns the wording so a chip and the assistant's reply
    describe the same constraint the same way."""
    conversation_id = client.post("/chat/conversations", headers=auth_headers).json()["id"]
    filters.save_state(conversation_id, filters.SearchState(max_price=800.0, min_rating=4.0))

    labels = {c["dimension"]: c["label"] for c in
              client.get(f"/chat/conversations/{conversation_id}/filters", headers=auth_headers).json()}
    assert labels["price"] == "under Rs 800"
    assert labels["rating"] == "4.0+ stars"
