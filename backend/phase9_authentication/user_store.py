"""
Phase 9 - Authentication & Authorization.

Registers users, authenticates login attempts, and looks users up by id -
all backed by the `users` table (schema.sql) via the same connection helper
used by the rest of the storage layer (Phase 3).
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3_storage_indexing"))

from db import get_connection  # noqa: E402
from auth import hash_password, verify_password  # noqa: E402


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(email: str, password: str, display_name: str | None = None) -> dict:
    email = _normalize_email(email)
    hashed = hash_password(password)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "insert into users (email, hashed_password, display_name) "
                    "values (%s, %s, %s) returning id, email, display_name, created_at;",
                    (email, hashed, display_name),
                )
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise EmailAlreadyRegisteredError(f"Email already registered: {email}")
            row = cur.fetchone()
        conn.commit()
        return {"id": row[0], "email": row[1], "display_name": row[2], "created_at": row[3]}
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> dict:
    email = _normalize_email(email)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, email, hashed_password, display_name from users where email = %s;",
                (email,),
            )
            row = cur.fetchone()
        if row is None or not verify_password(password, row[2]):
            raise InvalidCredentialsError("Invalid email or password")
        return {"id": row[0], "email": row[1], "display_name": row[3]}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, email, display_name, created_at from users where id = %s;",
                (user_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"id": row[0], "email": row[1], "display_name": row[2], "created_at": row[3]}
    finally:
        conn.close()
