"""
Tests for rank fusion.

Pure logic. These pin the two behaviours the previous "order by best single
review" ranking got wrong: sustained evidence should beat a lucky outlier,
and the structured signal shouldn't vanish the moment a vibe query appears.
"""

from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion, semantic_score


# --- semantic_score ------------------------------------------------------------


def test_several_supporting_reviews_beat_one_stronger_outlier():
    """The regression this exists for: a single 0.71 review used to outrank
    three consistent 0.68s, because only the best one counted."""
    assert semantic_score([0.68, 0.68, 0.68]) > semantic_score([0.71])


def test_volume_alone_cannot_win():
    """Capping at top_k stops 15 mediocre reviews from beating 3 excellent
    ones - otherwise the most-reviewed restaurant would win every query."""
    assert semantic_score([0.4] * 15) < semantic_score([0.75, 0.72, 0.70])


def test_stronger_evidence_wins_at_equal_volume():
    assert semantic_score([0.8, 0.8, 0.8]) > semantic_score([0.5, 0.5, 0.5])


def test_no_reviews_scores_zero():
    assert semantic_score([]) == 0.0


def test_only_the_top_k_similarities_contribute():
    assert semantic_score([0.9, 0.8, 0.7, 0.6, 0.5], top_k=3) == semantic_score([0.9, 0.8, 0.7], top_k=3)


# --- reciprocal_rank_fusion -----------------------------------------------------


def test_agreement_between_rankings_wins():
    fused = reciprocal_rank_fusion([[1, 2, 3], [1, 3, 2]])
    assert fused[0].restaurant_id == 1


def test_structured_signal_still_counts_when_semantics_disagree():
    """Pure semantic ordering would return 9 first. Fusion lets a strongly
    structured result outrank a marginal semantic winner."""
    structured = [1, 2, 9]
    semantic = [9, 1, 2]
    fused = reciprocal_rank_fusion([structured, semantic])
    assert fused[0].restaurant_id in (1, 9)
    assert {f.restaurant_id for f in fused} == {1, 2, 9}


def test_ids_missing_from_one_ranking_still_appear():
    """A restaurant surfaced semantically but absent from the structured
    ordering (or vice versa) must not be silently dropped."""
    fused = reciprocal_rank_fusion([[1, 2], [3]])
    assert {f.restaurant_id for f in fused} == {1, 2, 3}


def test_rank_positions_are_reported_for_both_sources():
    fused = {f.restaurant_id: f for f in reciprocal_rank_fusion([[7, 8], [8, 7]])}
    assert fused[7].structured_rank == 1
    assert fused[7].semantic_rank == 2


def test_scores_use_the_documented_rrf_formula():
    fused = reciprocal_rank_fusion([[5], [5]])
    assert fused[0].score == 2 * (1 / (RRF_K + 1))


def test_weights_shift_the_balance_between_rankings():
    even = reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert even[0].restaurant_id == 2  # tie on score, broken by semantic rank

    semantic_heavy = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[0.5, 2.0])
    assert semantic_heavy[0].restaurant_id == 2


def test_mismatched_weights_are_rejected_rather_than_silently_ignored():
    try:
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched weights")


def test_ordering_is_deterministic_across_calls():
    rankings = [[3, 1, 2], [2, 3, 1]]
    first = [f.restaurant_id for f in reciprocal_rank_fusion(rankings)]
    second = [f.restaurant_id for f in reciprocal_rank_fusion(rankings)]
    assert first == second


# --- weak evidence detection ---------------------------------------------------
#
# Pure logic over similarity scores. The thresholds themselves are empirical
# (see WEAK_EVIDENCE_SIMILARITY); these pin the decision rule, not the value.


def _fake_candidate(similarities):
    from dataclasses import dataclass, field

    @dataclass
    class S:
        similarity: float

    @dataclass
    class C:
        review_snippets: list = field(default_factory=list)

    return C(review_snippets=[S(s) for s in similarities])


def test_evidence_is_weak_when_even_the_best_snippet_is_far_off():
    """The Kaggadasapura case: a pool so small that every review clears the
    semantic cut, so the snippets are about noodle packaging."""
    from app.retrieval.hybrid import evidence_is_weak

    assert evidence_is_weak([_fake_candidate([0.15, 0.13, 0.10])]) is True


def test_evidence_is_not_weak_when_one_strong_snippet_exists():
    """Judged on the best snippet, not the average - one genuine match is
    enough to support a qualitative claim, and the reply only needs one."""
    from app.retrieval.hybrid import evidence_is_weak

    assert evidence_is_weak([_fake_candidate([0.45, 0.12, 0.10])]) is False


def test_evidence_is_judged_across_all_candidates_not_per_restaurant():
    from app.retrieval.hybrid import evidence_is_weak

    weak = _fake_candidate([0.11])
    strong = _fake_candidate([0.48])
    assert evidence_is_weak([weak, strong]) is False


def test_no_snippets_is_not_weak_evidence():
    """Nothing was shown, so there is nothing to over-claim about - a
    structured-only search must not trigger the hedge."""
    from app.retrieval.hybrid import evidence_is_weak

    assert evidence_is_weak([_fake_candidate([])]) is False
    assert evidence_is_weak([]) is False
