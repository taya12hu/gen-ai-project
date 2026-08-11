"""
Conversation & Personalization Storage.

CRUD helpers for conversations/messages, backed by the tables in schema.sql.
Ownership (a conversation belongs to exactly one user) is enforced at the
query level here, not just in the API layer, so any caller of this module
gets the same guarantee.
"""

import logging

from app.storage.db import get_connection

logger = logging.getLogger(__name__)


class ConversationNotFoundError(Exception):
    pass

# Sidebar label length - long enough to be recognizable, short enough that a
# CSS ellipsis on the sidebar row only kicks in for genuinely long messages.
TITLE_MAX_LENGTH = 60


def _derive_title(content: str) -> str:
    content = " ".join(content.split())  # collapse newlines/repeated whitespace
    if len(content) <= TITLE_MAX_LENGTH:
        return content
    return content[:TITLE_MAX_LENGTH].rstrip() + "…"


def create_conversation(user_id: int) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "insert into conversations (user_id) values (%s) returning id, user_id, title, created_at;",
                (user_id,),
            )
            row = cur.fetchone()
        conn.commit()
        logger.info("Created conversation_id=%s for user_id=%s", row[0], user_id)
        return {"id": row[0], "user_id": row[1], "title": row[2], "created_at": row[3]}
    finally:
        conn.close()


def get_conversation(conversation_id: int, user_id: int) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, user_id, title, created_at from conversations where id = %s and user_id = %s;",
                (conversation_id, user_id),
            )
            row = cur.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"No conversation {conversation_id} for this user")
        return {"id": row[0], "user_id": row[1], "title": row[2], "created_at": row[3]}
    finally:
        conn.close()


def list_conversations(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, user_id, title, created_at from conversations "
                "where user_id = %s order by created_at desc;",
                (user_id,),
            )
            rows = cur.fetchall()
        return [{"id": r[0], "user_id": r[1], "title": r[2], "created_at": r[3]} for r in rows]
    finally:
        conn.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    mentioned_restaurant_ids: list[int] | None = None,
    mentioned_review_ids: list[int] | None = None,
) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if role == "user":
                # coalesce: only the first user message ever sets it, so a
                # later message in the same conversation can't overwrite it.
                cur.execute(
                    "update conversations set title = coalesce(title, %s) where id = %s;",
                    (_derive_title(content), conversation_id),
                )
            cur.execute(
                "insert into messages "
                "(conversation_id, role, content, mentioned_restaurant_ids, mentioned_review_ids) "
                "values (%s, %s, %s, %s, %s) "
                "returning id, conversation_id, role, content, mentioned_restaurant_ids, "
                "mentioned_review_ids, created_at;",
                (conversation_id, role, content, mentioned_restaurant_ids, mentioned_review_ids),
            )
            row = cur.fetchone()
        conn.commit()
        return {
            "id": row[0],
            "conversation_id": row[1],
            "role": row[2],
            "content": row[3],
            "mentioned_restaurant_ids": row[4],
            "mentioned_review_ids": row[5],
            "created_at": row[6],
        }
    finally:
        conn.close()


def get_messages(conversation_id: int, limit: int = 50) -> list[dict]:
    """Most recent `limit` messages, oldest first (ready to feed straight into a prompt)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, conversation_id, role, content, mentioned_restaurant_ids, "
                "mentioned_review_ids, created_at "
                "from messages where conversation_id = %s "
                "order by created_at desc limit %s;",
                (conversation_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {
                "id": r[0],
                "conversation_id": r[1],
                "role": r[2],
                "content": r[3],
                "mentioned_restaurant_ids": r[4],
                "mentioned_review_ids": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_last_mentioned_restaurant_ids(conversation_id: int) -> list[int]:
    """Restaurant ids from the most recent assistant message that grounded any."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select mentioned_restaurant_ids from messages "
                "where conversation_id = %s and role = 'assistant' "
                "and mentioned_restaurant_ids is not null "
                "order by created_at desc limit 1;",
                (conversation_id,),
            )
            row = cur.fetchone()
        return list(row[0]) if row and row[0] else []
    finally:
        conn.close()
