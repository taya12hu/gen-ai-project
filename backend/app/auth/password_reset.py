"""
Authentication - password reset tokens.

Generates, stores, and verifies one-time password reset tokens. The raw
token is only ever returned to the caller once (to be emailed to the user,
see email.py) - only its SHA-256 hash is stored, so a database leak alone
can't be used to reset anyone's password. Unlike login passwords
(low-entropy, need bcrypt's deliberate slowness to resist guessing), reset
tokens are generated with 256 bits of randomness via secrets.token_urlsafe,
so a fast hash is sufficient - brute-forcing the token itself is infeasible
regardless of hash speed.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from app.storage.db import get_connection

logger = logging.getLogger(__name__)

TOKEN_TTL_MINUTES = 30


class InvalidResetTokenError(Exception):
    """Raised when a reset token is unknown, expired, or already used."""


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_reset_token(user_id: int) -> str:
    """Issues a new reset token for user_id, invalidating any of their
    still-outstanding tokens first (only the most recently requested reset
    link should work). Returns the raw token - the only time it ever exists
    outside a hash."""
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "delete from password_reset_tokens where user_id = %s and used_at is null;",
                (user_id,),
            )
            cur.execute(
                "insert into password_reset_tokens (user_id, token_hash, expires_at) values (%s, %s, %s);",
                (user_id, token_hash, expires_at),
            )
        conn.commit()
        logger.info("Issued password reset token for user_id=%s", user_id)
    finally:
        conn.close()

    return token


def consume_reset_token(token: str) -> int:
    """Validates the token (exists, not expired, not already used) and marks
    it used so it can't be replayed. Returns the associated user_id."""
    token_hash = _hash_token(token)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, user_id, expires_at, used_at from password_reset_tokens where token_hash = %s;",
                (token_hash,),
            )
            row = cur.fetchone()
            if row is None:
                logger.info("Reset token consumption failed: unknown token")
                raise InvalidResetTokenError("Unknown reset token")

            token_id, user_id, expires_at, used_at = row
            if used_at is not None:
                logger.info("Reset token consumption failed: already used (user_id=%s)", user_id)
                raise InvalidResetTokenError("Reset token already used")
            if expires_at < datetime.now(timezone.utc):
                logger.info("Reset token consumption failed: expired (user_id=%s)", user_id)
                raise InvalidResetTokenError("Reset token expired")

            cur.execute("update password_reset_tokens set used_at = now() where id = %s;", (token_id,))
        conn.commit()
        logger.info("Password reset token consumed for user_id=%s", user_id)
        return user_id
    finally:
        conn.close()
