"""
Conversation search state.

The structured constraints in play for a conversation - place, cuisines,
budget, minimum rating - held explicitly, per conversation, and readable by
the user.

Before this they were not held anywhere. `HybridFilters` was rebuilt from
scratch every turn, and a budget survived only because query understanding
re-read "under Rs 800" out of the last ten messages and chose to re-emit it.
That worked well enough to feel natural - "what about somewhere cheaper?"
needed no special handling - but it made the constraints *invisible* and
*unpredictable*: whether one carried over was a model judgement, nothing in
the interface showed which were active, and a search returning nothing gave no
hint that a ceiling from four turns ago was the reason.

The state is a merge, not a replacement, and the three cases are distinguished
explicitly rather than inferred:

- a value the model extracted from this message SETS that dimension,
- a dimension the model reports the user dropped ("anywhere", "any budget")
  CLEARS it,
- and silence LEAVES IT ALONE, which is what makes a constraint persist across
  turns at all.

The middle case is the one that has to be asked for rather than guessed:
without it, "somewhere in Indiranagar" followed by "actually anywhere" would
leave Indiranagar pinned forever, since both turns look identical to a schema
that only reports what was mentioned.

Kept deliberately separate from `user_preferences`. Preferences are durable
facts about a person and outlive the conversation; this is the state of one
search and dies with the thread.
"""

import json
import logging
from dataclasses import dataclass, field, replace

from app.storage.db import get_connection

logger = logging.getLogger(__name__)

# The dimensions a user can set or clear. These names are what query
# understanding is asked to emit in `cleared_filters`, and what the frontend
# addresses when removing a chip, so they are a small shared vocabulary rather
# than an internal detail.
FILTER_DIMENSIONS: tuple[str, ...] = ("place", "cuisines", "price", "rating")


@dataclass(frozen=True)
class SearchState:
    place: str | None = None
    cuisines: tuple[str, ...] = ()
    max_price: float | None = None
    min_rating: float | None = None

    def is_empty(self) -> bool:
        return not (self.place or self.cuisines or self.max_price is not None or self.min_rating is not None)

    def to_json(self) -> dict:
        return {
            "place": self.place,
            "cuisines": list(self.cuisines),
            "max_price": self.max_price,
            "min_rating": self.min_rating,
        }

    @classmethod
    def from_json(cls, raw: dict | None) -> "SearchState":
        if not raw:
            return cls()
        return cls(
            place=raw.get("place"),
            cuisines=tuple(raw.get("cuisines") or ()),
            max_price=raw.get("max_price"),
            min_rating=raw.get("min_rating"),
        )

    def cleared(self, dimension: str) -> "SearchState":
        """This state with one dimension dropped."""
        if dimension == "place":
            return replace(self, place=None)
        if dimension == "cuisines":
            return replace(self, cuisines=())
        if dimension == "price":
            return replace(self, max_price=None)
        if dimension == "rating":
            return replace(self, min_rating=None)
        raise ValueError(f"Unknown filter dimension: {dimension!r}")

    def as_chips(self) -> list[dict]:
        """The active constraints, labelled for display.

        The label is written here rather than in the frontend because it has to
        match what the user actually said - "under Rs 800", not "max_price:
        800" - and the same wording is what the reply refers to.
        """
        chips: list[dict] = []
        if self.place:
            chips.append({"dimension": "place", "label": self.place})
        if self.cuisines:
            chips.append({"dimension": "cuisines", "label": ", ".join(self.cuisines)})
        if self.max_price is not None:
            chips.append({"dimension": "price", "label": f"under Rs {self.max_price:.0f}"})
        if self.min_rating is not None:
            chips.append({"dimension": "rating", "label": f"{self.min_rating}+ stars"})
        return chips


def merge(state: SearchState, understanding, cleared: list[str] | None = None) -> SearchState:
    """Applies one turn's understanding to the running state.

    Set / clear / leave-alone, per dimension. `cleared` names dimensions the
    user asked to drop; anything the understanding extracted overrides a clear
    for the same dimension, since "anywhere, but make it Chinese" mentions both
    and the mention is the more specific instruction.
    """
    cleared = [d for d in (cleared or []) if d in FILTER_DIMENSIONS]

    place = understanding.place if understanding.place else (None if "place" in cleared else state.place)
    cuisines = (
        tuple(understanding.cuisines)
        if understanding.cuisines
        else (() if "cuisines" in cleared else state.cuisines)
    )
    max_price = (
        understanding.max_price
        if understanding.max_price is not None
        else (None if "price" in cleared else state.max_price)
    )
    min_rating = (
        understanding.min_rating
        if understanding.min_rating is not None
        else (None if "rating" in cleared else state.min_rating)
    )
    return SearchState(place=place, cuisines=cuisines, max_price=max_price, min_rating=min_rating)


def get_state(conversation_id: int) -> SearchState:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select filters from conversations where id = %s;", (conversation_id,))
            row = cur.fetchone()
        return SearchState.from_json(row[0] if row else None)
    finally:
        conn.close()


def save_state(conversation_id: int, state: SearchState) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "update conversations set filters = %s where id = %s;",
                (json.dumps(state.to_json()), conversation_id),
            )
        conn.commit()
    finally:
        conn.close()


def clear_dimension(conversation_id: int, dimension: str) -> SearchState:
    """Drops one constraint and returns what's left.

    Returns the resulting state rather than a bare success flag so the caller
    can hand the frontend the new chip set without a second round trip - a
    removed chip should disappear from a list that is already correct, not
    trigger a refetch.
    """
    if dimension not in FILTER_DIMENSIONS:
        raise ValueError(f"Unknown filter dimension: {dimension!r}")

    state = get_state(conversation_id).cleared(dimension)
    save_state(conversation_id, state)
    logger.info("Cleared filter %s for conversation_id=%s", dimension, conversation_id)
    return state
