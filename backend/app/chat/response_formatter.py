"""
Chat Response Formatting.

Turns the LLM's raw reply text into a ChatReply: which candidates it
actually talked about (name-matched against the candidate list, so the
facts shown to the user are DB-sourced, not restated by the LLM), which
review snippets support that, and the restaurant/review ids to persist on
the assistant's message - the restaurant ids are what lets a later "what
about its ambience?" resolve back to a concrete restaurant instead of
re-parsing prose; the review ids are what lets a reloaded conversation
replay the exact same snippets shown live instead of re-selecting a fresh
"top rated" set that may not match (e.g. a structured-only query that
attached no snippets at all should stay snippet-less on reload too).
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReviewSnippetOut:
    id: int
    text: str
    rating: float | None


@dataclass
class MatchedRestaurant:
    id: int
    name: str
    place: str
    city: str
    cuisines: list[str]
    price: float
    rating: float
    rest_type: str | None
    votes: int
    review_snippets: list[ReviewSnippetOut] = field(default_factory=list)


@dataclass
class ChatReply:
    reply_text: str
    matched_restaurants: list[MatchedRestaurant]
    mentioned_restaurant_ids: list[int]
    mentioned_review_ids: list[int]


def _to_matched(candidate) -> MatchedRestaurant:
    return MatchedRestaurant(
        id=candidate.id,
        name=candidate.name,
        place=candidate.place,
        city=candidate.city,
        cuisines=candidate.cuisines,
        price=candidate.price,
        rating=candidate.rating,
        rest_type=candidate.rest_type,
        votes=candidate.votes,
        review_snippets=[ReviewSnippetOut(id=s.id, text=s.text, rating=s.rating) for s in candidate.review_snippets],
    )


def _name_mentioned(name: str, llm_text: str) -> bool:
    # Case-insensitive and markdown-stripped so "**Toit**" or a lowercase
    # mention still grounds the reply - the LLM isn't required to reproduce
    # the name byte-for-byte for its reviews to show.
    normalized_text = llm_text.replace("*", "").replace("_", "").lower()
    return name.lower() in normalized_text


def format_chat_reply(candidates: list, llm_text: str, force_match: bool = False) -> ChatReply:
    llm_text = llm_text.strip()
    # force_match: the followup-question path already resolved to exactly
    # one restaurant before the LLM call (see service.py) - the model isn't
    # asked to restate its name (e.g. "what about its ambience?" answers
    # without naming it), so name-matching would wrongly drop its reviews.
    matched = candidates if force_match else [c for c in candidates if _name_mentioned(c.name, llm_text)]
    logger.info("Chat reply grounded %d/%d candidate(s)", len(matched), len(candidates))

    matched_out = [_to_matched(c) for c in matched]
    return ChatReply(
        reply_text=llm_text,
        matched_restaurants=matched_out,
        mentioned_restaurant_ids=[c.id for c in matched],
        mentioned_review_ids=[s.id for r in matched_out for s in r.review_snippets],
    )
