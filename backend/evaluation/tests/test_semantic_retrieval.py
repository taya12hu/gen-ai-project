"""
Semantic retrieval evaluation (structural/regression test).

Runs the approved ground-truth queries (evaluation/ground_truth.py) through
the production retrieval path (evaluation/semantic_retrieval.py ->
get_hybrid_candidates) and checks the pipeline behaves structurally - it
returns at most k candidates and produces valid recall/precision numbers -
rather than asserting a specific recall/precision bar.

This is deliberately NOT a quality gate: on the current data, manual spot
checks (see EVALUATION.md) show recall@5 legitimately near 0 for several
queries, not because retrieval is broken (the retrieved restaurants are
often plausible matches too - e.g. other genuinely quiet restaurants for
"quiet place away from crowds") but because the hand-picked ground truth is
a small, strict sample of one valid answer set out of a much larger pool of
"good enough" restaurants, and dense embeddings can pick up misleading
lexical overlap (e.g. "quiet" matching a review's typo of "quite"). A hard
recall threshold here would be a flaky, meaningless CI gate. Actual
recall@5/precision@5 numbers are informational, printed by
evaluation/run_eval.py's scorecard - read them there, not as pass/fail here.
"""

import pytest

from evaluation.ground_truth import SEMANTIC_GROUND_TRUTH
from evaluation.semantic_retrieval import RETRIEVAL_K, run_semantic_query


@pytest.mark.parametrize("query, ground_truth_ids", SEMANTIC_GROUND_TRUTH.items())
def test_semantic_query_returns_valid_result(query, ground_truth_ids):
    result = run_semantic_query(query, ground_truth_ids)

    assert result.query == query
    assert len(result.retrieved_ids) <= RETRIEVAL_K
    assert all(isinstance(rid, int) for rid in result.retrieved_ids)
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert result.latency_seconds >= 0.0
