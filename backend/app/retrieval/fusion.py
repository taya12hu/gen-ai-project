"""
Hybrid Retrieval Engine - rank fusion.

The two halves of a hybrid query produce two independent orderings of the
same restaurants: a structured one (rating desc, votes desc - "is this place
any good?") and a semantic one (how well its reviews match the vibe text -
"is this place the *kind* of place you asked for?"). Something has to
combine them into one list.

The previous version didn't combine them at all. It ordered restaurants by
where their single best-matching review happened to land in the pgvector
results, which had two consequences worth naming:

- One fluky review beat sustained evidence. A restaurant with a single
  0.71-similarity review outranked one with three consistent 0.68s, even
  though the second is far better supported.
- The structured signal was discarded entirely the moment a vibe query was
  present. A 4.6-star place and a 3.1-star place ranked identically if their
  best review scored the same.

Two pieces fix that:

1. `semantic_score` sums a restaurant's top-K review similarities instead of
   taking the max. Summing (not averaging) is deliberate: it rewards a
   restaurant that has *several* supporting reviews, while the top-K cap
   stops a restaurant with 15 mediocre reviews from beating one with 3
   excellent ones purely on volume.

2. `reciprocal_rank_fusion` merges the two orderings by rank rather than by
   score. Cosine similarities (~0.3-0.8, tightly clustered) and star ratings
   (1-5) live on incompatible scales, so any weighted sum of the raw numbers
   would need arbitrary normalisation constants that drift with the dataset.
   RRF only looks at position, which sidesteps that entirely and is the
   standard approach for exactly this problem.
"""

from dataclasses import dataclass

# Reviews per restaurant that contribute to its semantic score. 3 matches
# SNIPPETS_PER_RESTAURANT - the evidence the user is actually shown is the
# evidence the ranking is based on, which keeps the reply's justification
# honest.
SEMANTIC_TOP_K = 3

# RRF's smoothing constant, from Cormack et al. (2009). Larger values flatten
# the contribution of top ranks, so a single first-place finish can't dominate
# the fused order on its own; 60 is the published default and behaves well
# without dataset-specific tuning.
RRF_K = 60


@dataclass(frozen=True)
class FusedRank:
    restaurant_id: int
    score: float
    structured_rank: int | None
    semantic_rank: int | None


def semantic_score(similarities: list[float], top_k: int = SEMANTIC_TOP_K) -> float:
    """Sum of a restaurant's strongest `top_k` review similarities.

    Sum rather than max, so three supporting reviews beat one; capped at
    top_k, so volume alone can't win.
    """
    if not similarities:
        return 0.0
    return sum(sorted(similarities, reverse=True)[:top_k])


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K, weights: list[float] | None = None
) -> list[FusedRank]:
    """Merges several orderings of restaurant ids into one.

    Each ranking contributes `weight / (k + rank)` for the ids it contains,
    where rank is 1-based. An id missing from a ranking simply contributes
    nothing from it - which is the behaviour we want when semantic search
    surfaced a restaurant the structured ordering never reached, or vice
    versa.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must be the same length as rankings")

    scores: dict[int, float] = {}
    positions: dict[int, list[int | None]] = {}

    for ranking_index, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, restaurant_id in enumerate(ranking, start=1):
            scores[restaurant_id] = scores.get(restaurant_id, 0.0) + weight / (k + rank)
            positions.setdefault(restaurant_id, [None] * len(rankings))[ranking_index] = rank

    fused = [
        FusedRank(
            restaurant_id=restaurant_id,
            score=score,
            structured_rank=positions[restaurant_id][0] if len(rankings) > 0 else None,
            semantic_rank=positions[restaurant_id][1] if len(rankings) > 1 else None,
        )
        for restaurant_id, score in scores.items()
    ]
    # Tie-break on semantic rank then id, so a fused ordering is stable across
    # runs instead of depending on dict iteration order.
    fused.sort(
        key=lambda f: (
            -f.score,
            f.semantic_rank if f.semantic_rank is not None else 10**9,
            f.restaurant_id,
        )
    )
    return fused
