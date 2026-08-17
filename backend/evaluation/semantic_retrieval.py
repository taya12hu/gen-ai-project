"""
Semantic ("vibe") retrieval evaluation.

Calls the actual production retrieval entry point, get_hybrid_candidates(),
with no structured filters (place/cuisines/price/rating all unset) and the
query text as vibe_query - the same path a vibe-only chat message (e.g. "I
want a quiet place") hits in production: app.chat.service.prepare_chat_turn
builds an all-empty HybridFilters() whenever query understanding extracts no
place/cuisine/price/rating, and passes the extracted vibe_query straight into
get_hybrid_candidates(). semantic_search_reviews() itself isn't called
directly here because it returns raw (review, similarity) rows, not the
restaurant-ranked, deduped, snippet-attached output the chat pipeline (and
therefore a real user) actually sees - evaluating that raw function would
score something no user ever gets shown.

Retrieval only - no Groq/Gemini call - so this stays fast and isolates
retrieval quality from generation quality, per the ground truth being an
independent check on the retriever, not on the LLM.

SemanticQueryResult carries the full retrieved RestaurantCandidate objects
(not just their IDs) so run_eval.py can hand their review_snippets to
evaluation.judge.judge_retrieval_relevance as evidence for a second,
ground-truth-independent relevance check - that judging happens in
run_eval.py, not here, to keep this module's "no LLM call" property intact.
"""

import time
from dataclasses import dataclass

from app.retrieval.hybrid import HybridFilters, get_hybrid_candidates
from evaluation.ground_truth import SEMANTIC_GROUND_TRUTH

# Matches SEARCH_LIMIT in app.chat.service and the limit used by the existing
# 6-scenario suite's run_scenario, so results are comparable across suites.
RETRIEVAL_K = 5


@dataclass
class SemanticQueryResult:
    query: str
    ground_truth_ids: list[int]
    retrieved_ids: list[int]
    retrieved_candidates: list  # RestaurantCandidate objects - evidence for judge_retrieval_relevance
    recall_at_k: float
    precision_at_k: float
    latency_seconds: float
    judged_relevant: list[bool | None] | None = None  # filled in by run_eval.py after judging


def _recall_at_k(retrieved_ids: list[int], ground_truth_ids: list[int]) -> float:
    """Fraction of the ground-truth set that showed up in the top-k."""
    if not ground_truth_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(ground_truth_ids))
    return hits / len(ground_truth_ids)


def _precision_at_k(retrieved_ids: list[int], ground_truth_ids: list[int], k: int) -> float:
    """Fraction of the top-k that are in the ground-truth set. Still a strict
    ID-match metric even with the widened ~11-15-ID ground-truth sets - see
    judge_retrieval_relevance (evaluation/judge.py) for a second, ID-list-
    independent relevance signal that doesn't share this metric's blind spot
    (a genuinely good result that simply isn't one of the approved IDs)."""
    if not retrieved_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(ground_truth_ids))
    return hits / min(len(retrieved_ids), k)


def run_semantic_query(query: str, ground_truth_ids: list[int], k: int = RETRIEVAL_K) -> SemanticQueryResult:
    start = time.monotonic()
    result = get_hybrid_candidates(HybridFilters(), vibe_query=query, limit=k)
    latency = time.monotonic() - start

    retrieved_ids = [c.id for c in result.candidates]
    return SemanticQueryResult(
        query=query,
        ground_truth_ids=ground_truth_ids,
        retrieved_ids=retrieved_ids,
        retrieved_candidates=result.candidates,
        recall_at_k=_recall_at_k(retrieved_ids, ground_truth_ids),
        precision_at_k=_precision_at_k(retrieved_ids, ground_truth_ids, k=k),
        latency_seconds=latency,
    )


def run_all_semantic_scenarios(k: int = RETRIEVAL_K) -> list[SemanticQueryResult]:
    return [run_semantic_query(query, ground_truth_ids, k=k) for query, ground_truth_ids in SEMANTIC_GROUND_TRUTH.items()]
