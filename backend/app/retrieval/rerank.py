"""
Hybrid Retrieval Engine - cross-encoder reranking.

The last stage of retrieval, and the only one that can fix a *wrong* match
rather than merely reorder the matches already found.

Everything upstream runs on bi-encoder similarity: the query and each review
are embedded separately and compared by cosine distance. That is what makes
the search fast enough to run over 58,786 reviews, and it is also its ceiling
- the two texts never meet, so the model judges them by how close their
independent summaries land, not by whether one actually answers the other.
The failure this produces is documented in EVALUATION.md: a review saying
"Ambience is quiet good" (a typo for "quite good", nothing to do with noise)
surfaces for a query about somewhere quiet, because at the level of a single
embedding those really are similar texts.

A cross-encoder reads the query and the review *together* in one forward pass
and scores the pair directly. Far more accurate, and far too slow to run over
a whole corpus - which is exactly why it belongs here, at the end, over a
shortlist that fusion has already narrowed to a few dozen.

The measured difference on real snippets, scoring against "quiet place away
from crowds":

    +0.55   "ambience is very calm and peaceful, perfect for a quiet dinner"
    -8.65   "loved the cosy corner seating, soft music, never crowded"
   -11.32   "ordered schezwan noodles... packing was pretty standard"
   -11.38   "I love moist Biryanis with lots of masala"

Scores are raw logits, so only their order and sign carry meaning - but the
separation is decisive in a way cosine similarity (0.10-0.50 across that same
range) is not.

Failure is never fatal here. If the model cannot be loaded - missing weights,
memory pressure on a small instance, RERANK_ENABLED turned off - retrieval
falls back to the fusion ordering, which is the previous behaviour and still
perfectly serviceable.
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Trained for query/passage relevance ranking, 22.7M parameters. Small enough
# to run on CPU next to the bi-encoder (which is the deployment constraint
# that matters - see the CPU-only torch pin in requirements.txt), and the
# standard baseline for this job rather than an exotic choice.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reviews run long; 384 tokens covers the overwhelming majority whole and
# truncates the rest at a point well past where relevance is decided.
MAX_LENGTH = 384

# Restaurants from the fused ordering to rerank, and how many of each one's
# snippets to score.
#
# Both numbers are a latency budget. Real reviews run to a median of 302
# characters and a mean of 379, which puts measured throughput at ~27 pairs/s
# on CPU - not the ~280/s an early benchmark on short synthetic strings
# suggested, and the reason those first numbers were far too optimistic. 30
# pairs measured 1.14s end to end, which is too much to add to a chat turn, so
# 12 restaurants x 1 snippet - about 0.46s - is where this lands.
#
# The budget is spent on pool depth rather than on snippets per restaurant.
# Depth is what lets a genuinely better match sitting at fused rank 11 reach
# the top 5, which is the entire point of reranking. The snippets within one
# restaurant were already ordered by the bi-encoder and each restaurant is
# ranked by its best one, so scoring the runner-up rarely changes that
# restaurant's standing - it is the cheapest thing to give up.
RERANK_POOL = 12
SNIPPETS_PER_CANDIDATE = 1

BATCH_SIZE = 32

# Deliberately no absolute relevance threshold here.
#
# ms-marco cross-encoders emit logits where positive nominally means "this
# passage answers this query", which makes zero look like a free, principled
# boundary. It isn't, on this data: "loved the cosy corner seating, soft music,
# never crowded" - a genuinely good answer to a query about somewhere quiet -
# scores -8.65, because the scale is calibrated for MS MARCO passages rather
# than restaurant reviews. Only the ORDER of these scores is trustworthy here.
# Whether the evidence is good enough to claim anything stays with
# evidence_is_weak in hybrid.py.


def is_enabled() -> bool:
    """Off unless RERANK_ENABLED is set.

    Default-off is a deployment call, not a verdict on the technique. The
    quality gain is real and visible - for "quiet place away from crowds" in
    Whitefield it promotes reviews reading "slightly away from the crowd and
    noise" and "quiet little place tucked away in the city" over the generic
    ones fusion had ranked first. The cost is that scoring 12 real reviews
    measured ~1.5s on a 12-core development machine, roughly doubling
    retrieval latency, and the model adds ~90MB of resident memory next to the
    bi-encoder.

    This project deploys to a free-tier instance already tight enough on memory
    to need a CPU-only torch pin (see requirements.txt), and slower than the
    machine those numbers came from. Turning this on there would likely cost
    several seconds per chat turn. So it ships available and measured rather
    than on: set RERANK_ENABLED=1 where the hardware allows, and the evaluation
    suite can be run both ways to see what it buys.
    """
    value = os.environ.get("RERANK_ENABLED", "0").strip().lower()
    # An explicitly empty value means off, not on. Reading it as on would make
    # `RERANK_ENABLED=` in an env file silently load a second model.
    return value in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def _get_model():
    """Loaded lazily and once. Import is inside the function so that a process
    which never reranks (ingestion, embedding, most tests) never pays for
    pulling in the model machinery."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder %s", MODEL_NAME)
    return CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)


def score_pairs(query: str, passages: list[str]) -> list[float] | None:
    """Relevance logits for each (query, passage), or None if unavailable.

    None rather than an exception or a list of zeros: the caller needs to tell
    "the reranker says these are all bad" apart from "the reranker did not
    run", because the first should change the ordering and the second must
    not.
    """
    if not passages:
        return []
    if not is_enabled():
        return None

    try:
        model = _get_model()
        scores = model.predict([(query, p) for p in passages], batch_size=BATCH_SIZE)
        return [float(s) for s in scores]
    except Exception as exc:
        # Broad by intention: a missing download, an OOM, a torch/CUDA
        # mismatch, all mean the same thing here - rank without it.
        logger.warning("Cross-encoder reranking unavailable (%s); falling back to fusion order", exc)
        return None


def warm() -> None:
    """Loads the model ahead of first use.

    Worth calling from a long-lived process's startup: the load takes ~20s
    from a warm HuggingFace cache and far longer on a cold one, and paying it
    inside a user's first chat turn is the difference between a slow request
    and an apparently broken one.
    """
    if not is_enabled():
        logger.info("Cross-encoder reranking disabled (RERANK_ENABLED=0)")
        return
    try:
        _get_model()
    except Exception as exc:
        logger.warning("Could not preload the cross-encoder (%s); retrieval will use fusion order", exc)
