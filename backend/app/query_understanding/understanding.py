"""
Query Understanding.

Turns one raw chat message (plus recent conversation context) into a
structured QueryUnderstanding: hard filters (place/cuisines/price/rating)
that go straight to SQL, a free-text "vibe" query for semantic search over
review text, whether the message refers back to a restaurant already named
in this conversation, and any durable preference statements ("I'm
vegetarian") to persist via app.conversation.preferences.

This is a single LLM call with JSON-mode output - deliberately separate from
the final recommendation call (app.chat) because the two have very
different jobs: this one extracts structure from *one message*, that one
writes a grounded natural-language reply from a *candidate shortlist*.
Mixing them would mean re-parsing the model's own prose to figure out what
it was asked, which is exactly the kind of hallucination surface the rest
of this pipeline avoids by keeping retrieval and generation separate.
"""

import json
import logging
import os
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from app.llm.gemini_client import get_json_completion_gemini

DEFAULT_MODEL = "openai/gpt-oss-120b"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the query understanding stage of a restaurant recommendation chat
assistant. Given the latest user message and recent conversation history,
extract a JSON object describing what the user wants. Output ONLY valid
JSON matching this shape, no other text:

{
  "intent": "search" | "followup_question" | "preference_statement" | "chitchat",
  "place": string or null,
  "cuisines": array of strings (empty if none mentioned),
  "max_price": number or null (budget for two people, in rupees),
  "min_rating": number or null (0-5),
  "vibe_query": string or null (qualitative description to search reviews
      for - e.g. "quiet", "good for a date", "great ambience" - null if the
      message is purely structured/factual with no qualitative aspect),
  "refers_to_previous_restaurant": boolean (true if this message is about a
      restaurant already named earlier in the conversation, e.g. "its
      ambience", "is it good for kids", "what about that place"),
  "new_preferences": object mapping short preference keys to values, only
      for facts the user is stating about themselves that should be
      remembered for future conversations too (e.g. {"dietary": "vegetarian"},
      {"ambience": "quiet"}). Do NOT include one-off requests for the
      current message here - only durable statements about the user
      ("I'm vegetarian", "I usually prefer quiet places").
  "relevant_preference_keys": array of strings - which of the "Remembered
      preferences" keys below (if any) are actually relevant to interpreting
      or answering THIS specific message. Only include a key if it should
      genuinely influence this request, not just because it exists. A
      dietary restriction is usually relevant to most food requests; a mood/
      occasion preference from an earlier conversation is often NOT relevant
      to an unrelated request (e.g. a past "quiet" preference isn't relevant
      to "suggest something for a quick lunch" unless this message is also
      about mood/ambience). Empty array if none apply, or if no preferences
      are listed below.
}

intent guide:
- "search": asking for a restaurant recommendation (even vague, e.g. "suggest something good").
- "followup_question": asking about a specific restaurant already discussed (ambience, reviews, etc).
- "preference_statement": only/mainly stating a lasting preference, not asking for a recommendation right now.
- "chitchat": greeting or anything unrelated to restaurants.

Known places in the dataset: __KNOWN_PLACES__
Known cuisines in the dataset: __KNOWN_CUISINES__
Only fill "place"/"cuisines" with values that reasonably match the known
lists above (case-insensitive match is fine) - if the user's wording
doesn't clearly match anything known, leave it null/empty rather than
guessing.

Remembered preferences for this user (previously stated, not necessarily
related to this message):
__PREFERENCES__
""".strip()


class QueryUnderstanding(BaseModel):
    intent: Literal["search", "followup_question", "preference_statement", "chitchat"] = "search"
    place: str | None = None
    cuisines: list[str] = Field(default_factory=list)
    max_price: float | None = None
    min_rating: float | None = None
    vibe_query: str | None = None
    refers_to_previous_restaurant: bool = False
    new_preferences: dict[str, str] = Field(default_factory=dict)
    relevant_preference_keys: list[str] = Field(default_factory=list)


def _get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _resolve_known_values(understanding: QueryUnderstanding, known_places: set[str], known_cuisines: set[str]) -> QueryUnderstanding:
    """Snap freeform place/cuisine text to canonical DB casing; drop anything unrecognized."""
    place_lookup = {p.lower(): p for p in known_places}
    cuisine_lookup = {c.lower(): c for c in known_cuisines}

    canonical_place = place_lookup.get(understanding.place.lower()) if understanding.place else None
    canonical_cuisines = [
        cuisine_lookup[c.lower()] for c in understanding.cuisines if c.lower() in cuisine_lookup
    ]

    return understanding.model_copy(update={"place": canonical_place, "cuisines": canonical_cuisines})


def build_messages(
    message: str,
    recent_messages: list[dict],
    known_places: set[str],
    known_cuisines: set[str],
    preferences: dict[str, str] | None = None,
) -> list[dict]:
    preferences = preferences or {}
    pref_lines = "\n".join(f"- {k}: {v}" for k, v in preferences.items()) if preferences else "(none stored yet)"
    system = (
        SYSTEM_PROMPT.replace("__KNOWN_PLACES__", ", ".join(sorted(known_places)))
        .replace("__KNOWN_CUISINES__", ", ".join(sorted(known_cuisines)))
        .replace("__PREFERENCES__", pref_lines)
    )

    history_lines = [f"{m['role']}: {m['content']}" for m in recent_messages]
    history_block = "\n".join(history_lines) if history_lines else "(no prior messages)"

    user_content = (
        f"Recent conversation:\n{history_block}\n\n"
        f"Latest user message: {message}\n\n"
        "Extract the JSON object described above for the latest user message."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _clamp_relevant_preference_keys(understanding: QueryUnderstanding, preferences: dict[str, str]) -> QueryUnderstanding:
    """Drop any key the model claimed as relevant that isn't actually one we
    gave it - a hallucinated or malformed key here would otherwise flow
    straight into which stored preference gets applied to retrieval."""
    valid = [k for k in understanding.relevant_preference_keys if k in preferences]
    if valid == understanding.relevant_preference_keys:
        return understanding
    return understanding.model_copy(update={"relevant_preference_keys": valid})


def understand_query(
    message: str,
    recent_messages: list[dict],
    known_places: set[str],
    known_cuisines: set[str],
    preferences: dict[str, str] | None = None,
    model: str | None = None,
) -> QueryUnderstanding:
    preferences = preferences or {}
    client = _get_client()
    resolved_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    messages = build_messages(message, recent_messages, known_places, known_cuisines, preferences)

    try:
        completion = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
    except Exception as exc:
        logger.warning("Groq call failed (%s); falling back to Gemini", exc)
        raw = get_json_completion_gemini(messages)

    try:
        understanding = QueryUnderstanding.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        # Degrade gracefully: treat the whole message as a vibe-only search
        # rather than failing the request outright.
        logger.warning("Query understanding returned invalid JSON; degrading to vibe-only search")
        understanding = QueryUnderstanding(intent="search", vibe_query=message)

    resolved = _resolve_known_values(understanding, known_places, known_cuisines)
    resolved = _clamp_relevant_preference_keys(resolved, preferences)
    logger.info(
        "Query understood: intent=%s place=%s cuisines=%s vibe_query=%r refers_to_previous=%s relevant_preferences=%s",
        resolved.intent, resolved.place, resolved.cuisines, resolved.vibe_query,
        resolved.refers_to_previous_restaurant, resolved.relevant_preference_keys,
    )
    return resolved
