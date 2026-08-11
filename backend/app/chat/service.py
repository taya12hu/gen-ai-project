"""
Chat API - service layer.

Orchestrates one chat turn: query understanding -> load remembered
preferences / recent history -> retrieve (either a fresh hybrid search or
a followup lookup on a previously named restaurant) -> build the prompt,
call the LLM, format the reply -> persist the turn + any new preference
facts. Kept separate from the FastAPI route (app/api/main.py) so it's
callable and testable without spinning up the app.

Split into three stages so the streaming endpoint can do its "does this
request even make sense" checks (ownership, mainly) BEFORE committing to a
streamed HTTP response - once a StreamingResponse starts, its status code
can no longer change, so a 404 has to happen earlier, not from inside the
generator that's actually streaming tokens:

- prepare_chat_turn: everything before the LLM call (may raise
  ConversationNotFoundError - always run this synchronously, not lazily).
- get_recommendation / stream_recommendation: the LLM call itself, either
  all at once or as it's generated.
- finalize_chat_turn: cross-references the finished LLM text against the
  candidates and persists the turn.

handle_chat_message composes all three for the plain non-streaming path.
"""

import logging
from dataclasses import dataclass

from app.chat.prompt_builder import build_chat_prompt
from app.chat.response_formatter import ChatReply, format_chat_reply
from app.conversation.preferences import get_preferences, upsert_preferences
from app.conversation.store import (
    ConversationNotFoundError,
    add_message,
    create_conversation,
    get_conversation,
    get_last_mentioned_restaurant_ids,
    get_messages,
)
from app.llm.groq_client import get_recommendation, stream_recommendation
from app.query_understanding.understanding import understand_query
from app.retrieval.hybrid import HybridFilters, get_hybrid_candidates, get_reviews_for_restaurant
from app.retrieval.known_values import get_known_cuisines, get_known_places

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 10
SEARCH_LIMIT = 5

# Preference keys folded into the semantic ("vibe") side of retrieval rather
# than a hard SQL filter, per the soft-default design: a stored fact should
# bias what gets surfaced, but a one-off request in the current message can
# still override it.
VIBE_PREFERENCE_KEYS = {"dietary", "ambience", "occasion", "vibe"}


def _effective_vibe_query(vibe_query: str | None, preferences: dict[str, str]) -> str | None:
    extras = [v for k, v in preferences.items() if k in VIBE_PREFERENCE_KEYS]
    parts = [p for p in [vibe_query, *extras] if p]
    return "; ".join(parts) if parts else None


@dataclass
class PreparedChatTurn:
    conversation_id: int
    message: str
    prompt: list[dict]
    candidates: list
    referenced_restaurant: object | None


def prepare_chat_turn(user_id: int, conversation_id: int | None, message: str) -> PreparedChatTurn:
    """Everything needed before the LLM call: resolves/validates the
    conversation, runs query understanding, retrieves candidates, and builds
    the prompt. Raises ConversationNotFoundError if conversation_id is given
    but doesn't belong to user_id - always run synchronously (not from
    inside a generator) so that error can still become a normal 404."""
    if conversation_id is None:
        conversation = create_conversation(user_id)
        conversation_id = conversation["id"]
    else:
        get_conversation(conversation_id, user_id)  # raises ConversationNotFoundError if not owned

    logger.info("Preparing chat turn for user_id=%s conversation_id=%s", user_id, conversation_id)
    recent_messages = get_messages(conversation_id, limit=HISTORY_LIMIT)
    preferences = get_preferences(user_id)

    known_places = get_known_places()
    known_cuisines = get_known_cuisines()
    understanding = understand_query(message, recent_messages, known_places, known_cuisines)

    if understanding.new_preferences:
        upsert_preferences(user_id, understanding.new_preferences)
        preferences = {**preferences, **understanding.new_preferences}

    candidates: list = []
    referenced_restaurant = None
    relaxed = False

    if understanding.intent == "followup_question" and understanding.refers_to_previous_restaurant:
        prior_ids = get_last_mentioned_restaurant_ids(conversation_id)
        if prior_ids:
            referenced_restaurant = get_reviews_for_restaurant(
                prior_ids[0], _effective_vibe_query(understanding.vibe_query, preferences)
            )
        else:
            understanding.intent = "search"  # nothing to refer back to - fall through to a fresh search

    if understanding.intent == "search" or (
        understanding.intent == "followup_question" and referenced_restaurant is None
    ):
        filters = HybridFilters(
            place=understanding.place,
            cuisines=understanding.cuisines,
            max_price=understanding.max_price,
            min_rating=understanding.min_rating,
        )
        result = get_hybrid_candidates(
            filters, _effective_vibe_query(understanding.vibe_query, preferences), limit=SEARCH_LIMIT
        )
        candidates = result.candidates
        relaxed = result.relaxed

    prompt = build_chat_prompt(
        message, understanding, candidates, relaxed, recent_messages, preferences, referenced_restaurant
    )

    return PreparedChatTurn(
        conversation_id=conversation_id,
        message=message,
        prompt=prompt,
        candidates=candidates,
        referenced_restaurant=referenced_restaurant,
    )


def finalize_chat_turn(prepared: PreparedChatTurn, llm_text: str) -> ChatReply:
    """Cross-references the finished LLM text against the candidates and
    persists both sides of the turn. Needs the LLM's text in full -
    restaurant-name matching can't happen on a partial reply - so this only
    runs once a streamed reply has finished, too."""
    matching_pool = (
        [prepared.referenced_restaurant] if prepared.referenced_restaurant is not None else prepared.candidates
    )
    reply = format_chat_reply(matching_pool, llm_text, force_match=prepared.referenced_restaurant is not None)

    add_message(prepared.conversation_id, "user", prepared.message)
    add_message(
        prepared.conversation_id,
        "assistant",
        reply.reply_text,
        mentioned_restaurant_ids=reply.mentioned_restaurant_ids or None,
        mentioned_review_ids=reply.mentioned_review_ids or None,
    )
    logger.info("Chat turn persisted for conversation_id=%s", prepared.conversation_id)

    return reply


def handle_chat_message(user_id: int, conversation_id: int | None, message: str) -> tuple[ChatReply, int]:
    prepared = prepare_chat_turn(user_id, conversation_id, message)
    llm_text = get_recommendation(prepared.prompt)
    reply = finalize_chat_turn(prepared, llm_text)
    return reply, prepared.conversation_id


def stream_chat_message_tokens(prepared: PreparedChatTurn):
    """Yields the LLM's reply text as it's generated. Pure token streaming
    only - accumulating the full text and finalizing (matching + persistence)
    is the caller's job, since that has to happen once, after the last
    token, not per-chunk."""
    yield from stream_recommendation(prepared.prompt)


__all__ = [
    "handle_chat_message",
    "prepare_chat_turn",
    "stream_chat_message_tokens",
    "finalize_chat_turn",
    "ConversationNotFoundError",
]
