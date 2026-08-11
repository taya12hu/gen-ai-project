"""
Tests for Authentication.

Password hashing and JWT tests are pure/unit (no DB, no network).
user store tests run against the live `users` table, each using a unique
throwaway email and cleaning up after itself so the suite is safely
re-runnable.
"""

import uuid

import jwt
import pytest

from app.auth import password_reset, tokens as auth, users as user_store
from app.storage.db import get_connection


# --- Password hashing ---------------------------------------------------


def test_hash_password_does_not_store_plaintext():
    hashed = auth.hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2b$")  # bcrypt hash prefix


def test_verify_password_succeeds_for_correct_password():
    hashed = auth.hash_password("my-secret-password")
    assert auth.verify_password("my-secret-password", hashed) is True


def test_verify_password_fails_for_wrong_password():
    hashed = auth.hash_password("my-secret-password")
    assert auth.verify_password("wrong-password", hashed) is False


def test_same_password_hashes_differently_each_time():
    # bcrypt salts each hash, so two hashes of the same password must differ.
    h1 = auth.hash_password("same-password")
    h2 = auth.hash_password("same-password")
    assert h1 != h2
    assert auth.verify_password("same-password", h1)
    assert auth.verify_password("same-password", h2)


# --- JWT -----------------------------------------------------------------


def test_create_and_decode_access_token_roundtrip():
    token = auth.create_access_token(user_id=42, email="test@example.com")
    payload = auth.decode_access_token(token)

    assert payload["sub"] == "42"
    assert payload["email"] == "test@example.com"


def test_decode_rejects_tampered_token():
    token = auth.create_access_token(user_id=42, email="test@example.com")
    tampered = token[:-4] + "abcd"

    with pytest.raises(auth.InvalidTokenError):
        auth.decode_access_token(tampered)


def test_decode_rejects_garbage_token():
    with pytest.raises(auth.InvalidTokenError):
        auth.decode_access_token("not-a-real-token")


def test_decode_rejects_expired_token():
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    expired_payload = {
        "sub": "1",
        "email": "test@example.com",
        "iat": now - datetime.timedelta(minutes=10),
        "exp": now - datetime.timedelta(minutes=1),
    }
    expired_token = jwt.encode(expired_payload, auth.JWT_SECRET_KEY, algorithm=auth.JWT_ALGORITHM)

    with pytest.raises(auth.InvalidTokenError):
        auth.decode_access_token(expired_token)


def test_decode_rejects_token_signed_with_wrong_secret():
    forged = jwt.encode(
        {"sub": "1", "email": "test@example.com"}, "wrong-secret-key", algorithm=auth.JWT_ALGORITHM
    )
    with pytest.raises(auth.InvalidTokenError):
        auth.decode_access_token(forged)


# --- user_store (live DB) -------------------------------------------------


@pytest.fixture
def throwaway_email():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    yield email
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email.strip().lower(),))
        conn.commit()
    finally:
        conn.close()


def test_create_user_stores_hashed_not_plaintext_password(throwaway_email):
    user_store.create_user(throwaway_email, "s3cret-password", display_name="Test User")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select hashed_password from users where email = %s;",
                (throwaway_email.strip().lower(),),
            )
            stored = cur.fetchone()[0]
    finally:
        conn.close()

    assert stored != "s3cret-password"
    assert auth.verify_password("s3cret-password", stored)


def test_create_user_normalizes_email_casing_and_whitespace(throwaway_email):
    messy_email = f"  {throwaway_email.upper()}  "
    created = user_store.create_user(messy_email, "password123")
    assert created["email"] == throwaway_email.strip().lower()


def test_duplicate_email_registration_is_rejected(throwaway_email):
    user_store.create_user(throwaway_email, "password123")

    with pytest.raises(user_store.EmailAlreadyRegisteredError):
        user_store.create_user(throwaway_email, "different-password")


def test_authenticate_user_succeeds_with_correct_credentials(throwaway_email):
    user_store.create_user(throwaway_email, "correct-password")

    user = user_store.authenticate_user(throwaway_email, "correct-password")
    assert user["email"] == throwaway_email.strip().lower()


def test_authenticate_user_fails_with_wrong_password(throwaway_email):
    user_store.create_user(throwaway_email, "correct-password")

    with pytest.raises(user_store.InvalidCredentialsError):
        user_store.authenticate_user(throwaway_email, "wrong-password")


def test_authenticate_user_fails_for_unknown_email():
    with pytest.raises(user_store.InvalidCredentialsError):
        user_store.authenticate_user("nobody-registered@example.com", "whatever")


def test_get_user_by_id_returns_user(throwaway_email):
    created = user_store.create_user(throwaway_email, "password123")

    fetched = user_store.get_user_by_id(created["id"])
    assert fetched["email"] == throwaway_email.strip().lower()


def test_get_user_by_id_returns_none_for_unknown_id():
    assert user_store.get_user_by_id(-1) is None


def test_get_user_by_email_returns_user(throwaway_email):
    created = user_store.create_user(throwaway_email, "password123")

    fetched = user_store.get_user_by_email(throwaway_email)
    assert fetched["id"] == created["id"]


def test_get_user_by_email_returns_none_for_unknown_email():
    assert user_store.get_user_by_email("nobody-registered@example.com") is None


def test_set_password_updates_hash_and_old_password_stops_working(throwaway_email):
    created = user_store.create_user(throwaway_email, "old-password123")

    user_store.set_password(created["id"], "new-password456")

    with pytest.raises(user_store.InvalidCredentialsError):
        user_store.authenticate_user(throwaway_email, "old-password123")
    assert user_store.authenticate_user(throwaway_email, "new-password456")["id"] == created["id"]


# --- password_reset (live DB) ----------------------------------------------


@pytest.fixture
def registered_user(throwaway_email):
    return user_store.create_user(throwaway_email, "original-password123")


def test_create_and_consume_reset_token_roundtrip(registered_user):
    token = password_reset.create_reset_token(registered_user["id"])

    user_id = password_reset.consume_reset_token(token)
    assert user_id == registered_user["id"]


def test_consume_unknown_token_is_rejected():
    with pytest.raises(password_reset.InvalidResetTokenError):
        password_reset.consume_reset_token("not-a-real-token")


def test_consume_same_token_twice_is_rejected(registered_user):
    token = password_reset.create_reset_token(registered_user["id"])
    password_reset.consume_reset_token(token)

    with pytest.raises(password_reset.InvalidResetTokenError):
        password_reset.consume_reset_token(token)


def test_consume_expired_token_is_rejected(registered_user):
    token = password_reset.create_reset_token(registered_user["id"])

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "update password_reset_tokens set expires_at = now() - interval '1 minute' "
                "where user_id = %s;",
                (registered_user["id"],),
            )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(password_reset.InvalidResetTokenError):
        password_reset.consume_reset_token(token)


def test_requesting_a_new_token_invalidates_the_previous_one(registered_user):
    first_token = password_reset.create_reset_token(registered_user["id"])
    password_reset.create_reset_token(registered_user["id"])  # supersedes first_token

    with pytest.raises(password_reset.InvalidResetTokenError):
        password_reset.consume_reset_token(first_token)
