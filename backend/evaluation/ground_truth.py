"""
Manually verified ground truth for the semantic ("vibe") retrieval evaluation.

Each query maps to a set of restaurant IDs a human (project owner) approved
after an independent raw-SQL search of restaurant_reviews.review_text for
several keyword/synonym variants per query (not just the literal phrase) -
deliberately NOT using get_hybrid_candidates(), semantic_search_reviews(),
embeddings, or any other part of the production retrieval pipeline, so the
ground truth isn't circularly defined by the thing it's evaluating. See
EVALUATION.md for the full selection methodology and rationale for widening
this set.

Deliberately a *looser* bar than "the single best match": each ID here is a
restaurant with clear, explicit textual evidence supporting the query, not
necessarily the best-worded example of it. "fresh ingredients and fresh
tasting food" has fewer entries than the others (11 vs ~15) because the
corpus genuinely has less explicit evidence for that specific query -
padding it out with weak matches would defeat the point of hand-verifying
each one.

This is fixed input to the evaluation. Nothing in evaluation/ may re-derive,
re-select, or otherwise modify these IDs.
"""

SEMANTIC_GROUND_TRUTH: dict[str, list[int]] = {
    "quiet place away from crowds": [
        3052, 5352, 3742, 8630, 5572, 2222, 4853, 1711, 2007, 2262, 9125, 4689, 9060, 6425, 2346,
    ],
    "good value for money": [
        3449, 4234, 2497, 6629, 6610, 752, 1492, 3981, 926, 8779, 3924, 5396, 5663, 2677, 1249,
    ],
    "fresh ingredients and fresh tasting food": [
        3935, 7218, 9071, 1618, 1484, 2931, 6288, 7985, 2471, 410, 4962,
    ],
    "friendly and attentive staff": [
        2238, 8053, 1591, 5990, 3230, 6311, 4909, 6344, 3742, 1278, 8369, 3309, 1671, 5275, 632,
    ],
    "authentic traditional Indian food": [
        6749, 3823, 2165, 3627, 2955, 6328, 5427, 2825, 389, 7143, 5216, 3424, 6076, 3251, 1584,
    ],
    "good food for a late-night meal": [
        44, 4962, 3671, 9087, 6219, 4267, 2626, 8515, 3746, 5577, 2314, 5891, 1996, 4351, 7220,
    ],
}
