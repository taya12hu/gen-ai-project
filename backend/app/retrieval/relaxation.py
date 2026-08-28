"""
Hybrid Retrieval Engine - constraint relaxation ladder.

When strict filters match nothing, the retriever loosens the *soft* numeric
constraints (budget and minimum rating) rather than returning an empty list.
Place and cuisine are never relaxed: "Chinese in Indiranagar" has no useful
weaker form - a Thai place in Whitefield is not a partial answer to it -
whereas "under Rs500" genuinely does ("here are some at Rs700").

This used to be a single `enforce_price_rating: bool`, which dropped *both*
constraints at once and could only report "constraints were relaxed" with no
detail. That produced answers like Rs1800 / 3.2-star results for a "under
Rs500, 4.5-star" request, with no way for the reply to explain which of the
two had actually moved or by how much.

The ladder here widens one dimension at a time, by a bounded amount, and
records exactly what it did (see AppliedRelaxation.describe) so the prompt
can tell the user the truth: "nothing under Rs500 - here are some up to
Rs750". Steps are only generated for constraints the user actually set, so a
query with no budget never pays for a budget-widening round trip.
"""

from dataclasses import dataclass

# Multiplied into the budget at each widening step. 1.5x then 2.25x keeps the
# suggestion recognisably close to what was asked for - a 10x budget "match"
# is not a useful answer to a price-sensitive question.
PRICE_WIDEN_FACTOR = 1.5
# Subtracted from the minimum rating at each widening step. 0.5 is one full
# half-star, the granularity users actually think in.
RATING_WIDEN_STEP = 0.5
# Ratings below this are not worth surfacing even as a relaxed match.
RATING_FLOOR = 3.0

# Every rung is one indexed query, so the ladder's length is a latency budget.
# A full price x rating grid reaches ~20 combinations, which is both slow to
# walk and pointless at the far end: by then the "match" is so far from what
# was asked that it isn't an answer to the question. Intermediate rungs are
# dropped to fit; the strictest and the fully-relaxed rung are always kept, so
# behaviour at both ends is unchanged.
MAX_LADDER_STEPS = 8


@dataclass(frozen=True)
class AppliedRelaxation:
    """The price/rating constraints actually used for a query, alongside what
    the user originally asked for. `None` on a used_* field means that
    constraint was dropped entirely for this attempt."""

    used_max_price: float | None
    used_min_rating: float | None
    requested_max_price: float | None
    requested_min_rating: float | None

    @property
    def price_relaxed(self) -> bool:
        return self.requested_max_price is not None and self.used_max_price != self.requested_max_price

    @property
    def rating_relaxed(self) -> bool:
        return self.requested_min_rating is not None and self.used_min_rating != self.requested_min_rating

    @property
    def is_relaxed(self) -> bool:
        return self.price_relaxed or self.rating_relaxed

    def describe(self) -> str | None:
        """A plain sentence for the prompt, naming what moved and to what.
        None when nothing was relaxed, so callers can skip the note entirely."""
        if not self.is_relaxed:
            return None

        parts: list[str] = []
        if self.price_relaxed:
            if self.used_max_price is None:
                parts.append(f"no budget limit (asked for under Rs {self.requested_max_price:.0f})")
            else:
                parts.append(
                    f"budget widened to Rs {self.used_max_price:.0f} "
                    f"(asked for under Rs {self.requested_max_price:.0f})"
                )
        if self.rating_relaxed:
            if self.used_min_rating is None:
                parts.append(f"no rating floor (asked for {self.requested_min_rating}+)")
            else:
                parts.append(
                    f"minimum rating lowered to {self.used_min_rating:.1f} "
                    f"(asked for {self.requested_min_rating}+)"
                )
        return "Nothing matched the original constraints exactly, so: " + "; ".join(parts) + "."


def _widened_prices(requested: float | None) -> list[float | None]:
    """Successively weaker budgets, ending in 'no budget at all'."""
    if requested is None:
        return [None]
    return [requested, requested * PRICE_WIDEN_FACTOR, requested * PRICE_WIDEN_FACTOR**2, None]


def _widened_ratings(requested: float | None) -> list[float | None]:
    """Successively weaker rating floors, stopping at RATING_FLOOR before
    dropping the constraint entirely."""
    if requested is None:
        return [None]
    steps: list[float | None] = [requested]
    nxt = requested - RATING_WIDEN_STEP
    while nxt >= RATING_FLOOR:
        steps.append(nxt)
        nxt -= RATING_WIDEN_STEP
    steps.append(None)
    return steps


def relaxation_ladder(max_price: float | None, min_rating: float | None) -> list[AppliedRelaxation]:
    """Attempts to try in order, strictest first.

    Widening is interleaved rather than exhausting one dimension before
    touching the other: a user who asked for both a budget and a rating cares
    about both, so 'slightly over budget at the requested rating' and 'on
    budget slightly below the rating' should both be tried before either
    constraint is stretched far. Duplicate combinations (which arise whenever
    one of the two ladders is shorter) are dropped, so no attempt runs twice.
    """
    prices = _widened_prices(max_price)
    ratings = _widened_ratings(min_rating)

    seen: set[tuple[float | None, float | None]] = set()
    ladder: list[AppliedRelaxation] = []
    # Diagonal (by total distance from strict) rather than nested loops, so
    # the second attempt is one step off in one dimension - never two steps
    # off in one while the other is untouched.
    for total in range(len(prices) + len(ratings) - 1):
        for pi in range(min(total, len(prices) - 1), -1, -1):
            ri = total - pi
            if ri >= len(ratings):
                continue
            key = (prices[pi], ratings[ri])
            if key in seen:
                continue
            seen.add(key)
            ladder.append(
                AppliedRelaxation(
                    used_max_price=prices[pi],
                    used_min_rating=ratings[ri],
                    requested_max_price=max_price,
                    requested_min_rating=min_rating,
                )
            )
    return _cap(ladder)


def _cap(ladder: list[AppliedRelaxation]) -> list[AppliedRelaxation]:
    """Thins an over-long ladder, always keeping the first and last rungs.

    The first is the user's actual request and the last is the guaranteed-
    terminating "no numeric constraints" rung; everything between is a
    convenience, so intermediate rungs are sampled evenly rather than
    truncated - truncating would drop the fully-relaxed rung and could leave
    a query with no matching rung at all.
    """
    if len(ladder) <= MAX_LADDER_STEPS:
        return ladder

    middle = ladder[1:-1]
    keep = MAX_LADDER_STEPS - 2
    stride = len(middle) / keep
    sampled = [middle[int(i * stride)] for i in range(keep)]
    return [ladder[0], *sampled, ladder[-1]]
