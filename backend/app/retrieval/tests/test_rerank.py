"""
Tests for cross-encoder reranking.

The model itself is not loaded here. Loading it costs ~20s and hundreds of
megabytes, which would make the fast suite neither fast nor runnable on a
constrained machine - and what actually needs pinning is the wiring around it:
that it degrades to fusion order rather than failing, that "unavailable" and
"scored badly" stay distinguishable, and that a disabled reranker changes
nothing. Quality is measured by the evaluation suite (EVALUATION.md), which is
where a real model belongs.
"""

import pytest

from app.retrieval import rerank
from app.retrieval.hybrid import ReviewSnippet, RestaurantCandidate, _rerank_ids


def _candidate(restaurant_id: int, texts: list[str]) -> RestaurantCandidate:
    return RestaurantCandidate(
        id=restaurant_id, name=f"R{restaurant_id}", place="Indiranagar", city="Bangalore",
        cuisines=["Chinese"], price=500.0, rating=4.0, rest_type="Casual Dining", votes=100,
        review_snippets=[ReviewSnippet(id=i, text=t, rating=4.0, similarity=0.3) for i, t in enumerate(texts)],
    )


# --- the switch ---------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    """Off unless asked for - the model costs ~1.5s per query and ~90MB, which
    this project's free-tier deployment cannot spare."""
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert rerank.is_enabled() is False


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True),
    ("0", False), ("false", False), ("no", False), ("", False),
])
def test_enabled_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("RERANK_ENABLED", value)
    assert rerank.is_enabled() is expected


def test_score_pairs_returns_none_when_disabled(monkeypatch):
    """None, not zeros: the caller must be able to tell "did not run" from
    "ran and found nothing good", because only the second should reorder."""
    monkeypatch.setenv("RERANK_ENABLED", "0")
    assert rerank.score_pairs("quiet", ["some review text"]) is None


def test_score_pairs_handles_no_passages(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    assert rerank.score_pairs("quiet", []) == []


def test_model_failure_degrades_instead_of_raising(monkeypatch):
    """A missing download, an OOM, a torch mismatch - all mean the same thing
    here, and none of them should fail a user's search."""
    monkeypatch.setenv("RERANK_ENABLED", "1")
    monkeypatch.setattr(rerank, "_get_model", lambda: (_ for _ in ()).throw(RuntimeError("no weights")))
    assert rerank.score_pairs("quiet", ["text"]) is None


# --- ordering -----------------------------------------------------------------


def test_unavailable_reranker_leaves_fusion_order_untouched(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda *_: None)
    built = {1: _candidate(1, ["a"]), 2: _candidate(2, ["b"])}
    assert _rerank_ids("quiet", [1, 2], built) == [1, 2]


def test_reranking_promotes_the_better_match(monkeypatch):
    """The whole point: a restaurant fusion ranked second can win on evidence
    that actually answers the question."""
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [-9.0, 4.0])
    built = {1: _candidate(1, ["noodle packaging was standard"]), 2: _candidate(2, ["genuinely quiet corner"])}
    assert _rerank_ids("quiet", [1, 2], built) == [2, 1]


def test_restaurants_are_ranked_by_their_best_snippet(monkeypatch):
    """Best, not summed - one review genuinely answering the question is what
    justifies the recommendation, and one is all the reply needs to quote."""
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [-8.0, -7.0, 3.0])
    built = {1: _candidate(1, ["meh", "also meh"]), 2: _candidate(2, ["exactly right"])}
    assert _rerank_ids("quiet", [1, 2], built) == [2, 1]


def test_scores_are_recorded_on_the_snippets(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [2.5])
    built = {1: _candidate(1, ["quiet corner"])}
    _rerank_ids("quiet", [1], built)
    assert built[1].review_snippets[0].rerank_score == 2.5


def test_strongest_evidence_is_ordered_first_within_a_restaurant(monkeypatch):
    """Snippets are quoted in the order given, so the best should lead.

    Only bites when more than one snippet is scored. At the shipped budget of
    SNIPPETS_PER_CANDIDATE=1 this reordering is a no-op - the single scored
    snippet is already the bi-encoder's best - so the test raises the budget
    to exercise it, and it stays useful if that budget is ever raised.
    """
    monkeypatch.setattr(rerank, "SNIPPETS_PER_CANDIDATE", 2)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [-5.0, 6.0])
    built = {1: _candidate(1, ["weak", "strong"])}
    _rerank_ids("quiet", [1], built)
    assert built[1].review_snippets[0].text == "strong"


def test_unscored_snippets_keep_their_bi_encoder_order(monkeypatch):
    """At a budget of 1, the snippets that were never scored must not be
    shuffled - they fall in behind the scored one in the order the bi-encoder
    ranked them."""
    monkeypatch.setattr(rerank, "SNIPPETS_PER_CANDIDATE", 1)
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [1.0])
    built = {1: _candidate(1, ["first", "second", "third"])}
    _rerank_ids("quiet", [1], built)
    assert [s.text for s in built[1].review_snippets] == ["first", "second", "third"]


def test_only_the_budgeted_snippets_are_scored(monkeypatch):
    """SNIPPETS_PER_CANDIDATE is a latency budget; exceeding it silently would
    multiply a ~1.5s step."""
    seen = {}

    def capture(query, passages):
        seen["n"] = len(passages)
        return [0.0] * len(passages)

    monkeypatch.setattr(rerank, "score_pairs", capture)
    monkeypatch.setattr(rerank, "SNIPPETS_PER_CANDIDATE", 1)
    _rerank_ids("quiet", [1], {1: _candidate(1, ["a", "b", "c"])})
    assert seen["n"] == 1


def test_ties_keep_the_fusion_ordering(monkeypatch):
    """Equal evidence means the reranker has no opinion, so the upstream
    ranking - which still knows about rating and votes - should stand."""
    monkeypatch.setattr(rerank, "score_pairs", lambda q, passages: [1.0, 1.0])
    built = {7: _candidate(7, ["same"]), 9: _candidate(9, ["same"])}
    assert _rerank_ids("quiet", [7, 9], built) == [7, 9]
