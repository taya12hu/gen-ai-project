"""
Authentication - user store.

Registers users, authenticates login attempts, and looks users up by id -
all backed by the `users` table (schema.sql) via the shared storage
connection helper.
"""

import logging

import psycopg2

from app.auth.tokens import hash_password, verify_password
from app.storage.db import get_connection

logger = logging.getLogger(__name__)


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
                logger.info("Registration rejected: email already registered")
                raise EmailAlreadyRegisteredError(f"Email already registered: {email}")
            row = cur.fetchone()
        conn.commit()
        logger.info("Registered new user_id=%s", row[0])
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
            logger.info("Login failed for email=%s", email)
            raise InvalidCredentialsError("Invalid email or password")
        logger.info("Login succeeded for user_id=%s", row[0])
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


def get_user_by_email(email: str) -> dict | None:
    email = _normalize_email(email)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, email, display_name, created_at from users where email = %s;",
                (email,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"id": row[0], "email": row[1], "display_name": row[2], "created_at": row[3]}
    finally:
        conn.close()


def set_password(user_id: int, new_password: str) -> None:
    hashed = hash_password(new_password)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("update users set hashed_password = %s where id = %s;", (hashed, user_id))
        conn.commit()
        logger.info("Password updated for user_id=%s", user_id)
    finally:
        conn.close()
