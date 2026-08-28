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
import re
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
    # Preference facts newly captured *this* turn (see
    # app.chat.service.prepare_chat_turn) - surfaced separately from
    # "Preferences applied to this search" so the frontend can tell the user
    # what was just remembered, instead of that happening silently.
    new_preferences: dict[str, str] = field(default_factory=dict)


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


# Separators after which a dataset name usually carries a branch, outlet or
# descriptor the model won't reproduce: "Truffles - Ice & Spice", "Empire
# Restaurant (Indiranagar)", "Toit, 100 Feet Road".
_NAME_SEPARATORS = re.compile(r"\s+[-–—|]\s+|\s*[(,]")

# A leading segment this weak isn't distinctive enough to ground a reply on
# its own - "The Bar" or "Cafe" would match incidental prose. A single token
# has to be long enough to be a real name rather than a common noun.
_MIN_CORE_TOKEN_LENGTH = 5


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens. Markdown emphasis, punctuation and
    ampersands all fall out as separators, so "**Toit**", "Toit," and "toit"
    all tokenize identically."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    """Whole-token subsequence match.

    Token-based rather than substring: a plain `"bar" in text` also matches
    inside "barbecue" and "barista", which quietly attached review evidence to
    restaurants the reply never discussed.
    """
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return True
    return False


def _core_name(name: str) -> str:
    """The distinctive leading part of a dataset name, before any branch or
    descriptor suffix."""
    return _NAME_SEPARATORS.split(name, maxsplit=1)[0].strip()


def _is_distinctive(tokens: list[str]) -> bool:
    return len(tokens) >= 2 or (len(tokens) == 1 and len(tokens[0]) >= _MIN_CORE_TOKEN_LENGTH)


def _name_mentioned(name: str, llm_text: str) -> bool:
    """Whether the reply actually names this restaurant.

    Two passes, because dataset names and conversational prose disagree in
    both directions. The full name is tried first; failing that, the core name
    is tried, so a reply saying "Truffles" still grounds the row stored as
    "Truffles - Ice & Spice". The distinctiveness guard keeps that second pass
    from turning a generically-named row into a match on ordinary prose.
    """
    text_tokens = _tokens(llm_text)
    if _contains_token_sequence(text_tokens, _tokens(name)):
        return True

    core_tokens = _tokens(_core_name(name))
    if core_tokens == _tokens(name) or not _is_distinctive(core_tokens):
        return False
    return _contains_token_sequence(text_tokens, core_tokens)


def format_chat_reply(
    candidates: list, llm_text: str, force_match: bool = False, new_preferences: dict[str, str] | None = None
) -> ChatReply:
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
        new_preferences=new_preferences or {},
    )
