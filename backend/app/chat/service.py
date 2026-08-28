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
from app.conversation.preferences import PREFERENCE_KEYS, get_preferences, upsert_preferences
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

# Preference keys eligible to fold into the semantic ("vibe") side of
# retrieval rather than a hard SQL filter, per the soft-default design: a
# stored fact should bias what gets surfaced, but a one-off request in the
# current message can still override it.
#
# This is the same closed vocabulary the storage layer enforces (see
# app.conversation.preferences.PREFERENCE_KEYS) rather than a second list
# maintained alongside it - when the two drifted apart, a fact stored under a
# key this module didn't recognize was saved, shown to the user, and then
# never applied to anything.
#
# Being *eligible* doesn't mean a given preference is applied to every turn,
# though - query understanding (see relevant_preference_keys below) decides
# per-message whether a specific stored fact is actually relevant to what
# was just asked. Unconditionally gluing every remembered preference onto
# every search used to silently skew unrelated requests (e.g. a stored
# "healthier" preference narrowing an unrelated "late-night meal" search)
# with no way for the user to know why.
VIBE_PREFERENCE_KEYS = frozenset(PREFERENCE_KEYS)


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
    new_preferences: dict[str, str]


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
    stored_preferences = {k: v for k, v in get_preferences(user_id).items() if k in VIBE_PREFERENCE_KEYS}

    known_places = get_known_places()
    known_cuisines = get_known_cuisines()
    understanding = understand_query(message, recent_messages, known_places, known_cuisines, stored_preferences)

    # upsert returns what was actually persisted after key normalization,
    # which can differ from what the model proposed - that's what the user
    # should be told was remembered, not the model's raw suggestion.
    new_preferences = upsert_preferences(user_id, understanding.new_preferences)

    # Two sources, combined:
    #
    # - Stored facts, but only the subset query understanding judged relevant
    #   to *this* message. A preference that exists but wasn't judged relevant
    #   this turn is simply not in play; it can still surface on a later turn
    #   where it genuinely applies.
    # - Facts stated in this very message, applied unconditionally. They don't
    #   go through the relevance gate: that gate exists to stop an unrelated
    #   fact from an older conversation leaking in, and something the user
    #   just said out loud in the message being answered is by definition not
    #   that. (This previously merged into a variable nothing read again, so
    #   "I'm vegetarian - where should I eat?" remembered the preference and
    #   then ignored it for the very request that stated it.)
    applied_preferences = {
        **{k: v for k, v in stored_preferences.items() if k in understanding.relevant_preference_keys},
        **new_preferences,
    }

    candidates: list = []
    referenced_restaurant = None
    relaxation_note: str | None = None

    if understanding.intent == "followup_question" and understanding.refers_to_previous_restaurant:
        prior_ids = get_last_mentioned_restaurant_ids(conversation_id)
        if prior_ids:
            referenced_restaurant = get_reviews_for_restaurant(
                prior_ids[0], _effective_vibe_query(understanding.vibe_query, applied_preferences)
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
            filters, _effective_vibe_query(understanding.vibe_query, applied_preferences), limit=SEARCH_LIMIT
        )
        candidates = result.candidates
        relaxation_note = result.relaxation_note()

    prompt = build_chat_prompt(
        message,
        understanding,
        candidates,
        relaxation_note,
        recent_messages,
        applied_preferences,
        referenced_restaurant,
    )

    return PreparedChatTurn(
        conversation_id=conversation_id,
        message=message,
        prompt=prompt,
        candidates=candidates,
        referenced_restaurant=referenced_restaurant,
        new_preferences=new_preferences,
    )


def finalize_chat_turn(prepared: PreparedChatTurn, llm_text: str) -> ChatReply:
    """Cross-references the finished LLM text against the candidates and
    persists both sides of the turn. Needs the LLM's text in full -
    restaurant-name matching can't happen on a partial reply - so this only
    runs once a streamed reply has finished, too."""
    matching_pool = (
        [prepared.referenced_restaurant] if prepared.referenced_restaurant is not None else prepared.candidates
    )
    reply = format_chat_reply(
        matching_pool,
        llm_text,
        force_match=prepared.referenced_restaurant is not None,
        new_preferences=prepared.new_preferences,
    )

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
