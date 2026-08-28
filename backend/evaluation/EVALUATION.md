# Chat Pipeline Evaluation

## What this covers

Every domain under `app/` has its own unit/integration tests, scoped to
that module (including auth's logic and the API layer's protected-route
tests). This directory evaluates the core hybrid-retrieval + chat-generation
pipeline as a whole, across three pieces:

1. **Structured retrieval + generation** (`tests/test_evaluation.py`) — 10
   scenarios run end to end (structured retrieval -> chat prompt
   construction -> Groq -> response formatting), checking **grounding**
   (every restaurant the LLM talks about is actually one it was given) and
   **relevance** (what it recommends genuinely satisfies the requested
   place/cuisine/price/rating).
2. **Semantic ("vibe") retrieval** (`semantic_retrieval.py`,
   `tests/test_semantic_retrieval.py`) — checks retrieval quality in
   isolation, independent of generation, against a small manually-verified
   ground truth.
3. **Hybrid filters + vibe** (`hybrid_retrieval.py`,
   `tests/test_hybrid_retrieval.py`) — 8 scenarios combining structured
   constraints *with* a qualitative query, checking **containment** (nothing
   returned violates the constraints the retriever says it applied),
   **evidence** (every candidate carries review snippets), and judged
   relevance.
4. **A scorecard** (`run_eval.py`) that runs all three suites plus an LLM judge
   and prints/saves a single summary — see "Scorecard" below.

## Scenarios (structured retrieval + generation)

10 scenarios pulled from real data, covering different areas, cuisines,
price tiers, and rating thresholds, plus four deliberate edge cases:

| Place | Cuisines | Max price | Min rating | Expected |
|---|---|---|---|---|
| Church Street | North Indian | 1000 | 4.0 | normal match |
| Ulsoor | South Indian | 1000 | 3.0 | normal match |
| Marathahalli | Desserts | 500 | 4.0 | normal match |
| MG Road | North Indian | 3000 | 4.0 | normal match |
| BTM | Biryani | 900 | 4.0 | normal match |
| Whitefield | Fast Food | 1000 | 4.3 | normal match |
| Kaggadasapura | Chinese | 800 | 4.9 | **relaxation fallback** (max rating there is 4.0) |
| Banashankari | Lucknowi | 1000 | 0 | **zero matches** (both values valid individually, combo has 0 rows) |
| Indiranagar | Chinese, Cafe | 2000 | 0 | **multi-cuisine** (matches restaurants serving either cuisine) |
| Indiranagar | Chinese | — | — | **omitted price/rating** (place+cuisine only, nothing to relax) |

These deliberately skip query understanding's NL-parsing stage (its own
accuracy is covered by `app/query_understanding/tests`) and build
`HybridFilters` directly, so this suite's focus stays on grounding and
relevance rather than intent extraction.

## Semantic ("vibe") retrieval evaluation

`app/retrieval/hybrid.py`'s `get_hybrid_candidates()` does two very different
jobs: exact structured filtering (place/cuisine/price/rating - SQL, checkable
deterministically) and fuzzy semantic search over review text via pgvector
embeddings (qualitative queries like "quiet", "good value for money" - not
checkable by assertion, needs a quality metric). The structured suite above
only exercises the first half (`vibe_query=None` in every scenario). This
piece evaluates the second half in isolation.

**Ground truth.** `ground_truth.py` holds a manually-approved set of (query
-> restaurant IDs) pairs for 6 qualitative queries - ~11-15 IDs per query
(fewer for "fresh ingredients", where the corpus genuinely has less explicit
evidence). These were **not** derived from the production retriever: they
came from an independent raw-SQL search of `restaurant_reviews.review_text`
for several keyword/synonym variants per query (e.g. "value for money",
"worth the price", "pocket friendly" for the value-for-money query),
aggregated by restaurant to find ones with multiple supporting reviews, then
read and hand-picked by a human for restaurants with explicit, unambiguous
textual evidence (not cuisine/rating/name vibes). This keeps the ground
truth independent of the thing being evaluated - it isn't circular.

This set started smaller (3-4 IDs per query, "the single best match") and
was deliberately widened. The original bar was too strict to fairly grade a
retriever with hundreds of plausible answers to choose from - see "Known
finding" below for what that looked like and why it didn't mean retrieval
was broken. The current bar is "clearly supports the query with explicit
evidence", not "the single best-worded example of it".

**Entry point.** `semantic_retrieval.py` calls `get_hybrid_candidates()` with
`HybridFilters()` (no place/cuisine/price/rating) and the query text as
`vibe_query`, `limit=5` - the exact path a vibe-only chat message hits in
production (`app.chat.service.prepare_chat_turn`). It does not call
`semantic_search_reviews` directly, and does not call Groq/Gemini at all -
retrieval quality is measured independent of generation quality.

**Metrics - two independent signals, not one.**

1. *Recall@5 / Precision@5* (`semantic_retrieval.py`) - strict ID match
   against the approved ground truth. Precise, but blind to a genuinely good
   result that just isn't one of the approved IDs - the failure mode
   documented below.
2. *Judged Relevance@5* (`judge.py`'s `judge_retrieval_relevance`, called
   from `run_eval.py`) - for each of a query's top-5 results, an LLM judge
   (Gemini) is shown the query and that result's best-matching review
   snippet and asked, independent of any ID list, whether the evidence
   genuinely supports the query. All 5 results for one query are judged in a
   single batched call (6 calls total for this suite, not 30) to stay well
   within Gemini's free-tier rate limit. This doesn't share metric 1's blind
   spot, but it's only as reliable as the judge itself - see the "LLM-as-
   judge" section below.

Reporting both together is the point: metric 1 says "did it find *my*
approved answers", metric 2 says "are the results it actually returned any
good", and a result can score well on one and poorly on the other.

**Current numbers (rank fusion, 2026-08-26).** After retrieval moved to
summed top-3 review evidence fused with the structured ranking (see
`app/retrieval/fusion.py`), Recall@5 went from 0.000 to 0.044 and Precision@5
from 0.000 to 0.133, with 2 of 6 queries now returning at least one approved
ground-truth id - the retriever had previously never matched a single one.
Judged Relevance@5 moved the other way, 0.767 to 0.700, reproducibly across
two runs. Both directions are consistent with what RRF does: it promotes
higher-rated restaurants that match slightly less strongly on review text.
Read that as a deliberate trade rather than a regression - and note that 30
judgements is a small sample either way.

**Known finding (from before the ground truth was widened).** With the
original, stricter 3-4-ID ground truth, Recall@5 was 0.0 across all 6
queries - not "low", zero. Spot-checking the actual retrieved reviews showed
this wasn't the retriever returning nonsense - e.g. for "quiet place away
from crowds" it surfaced other restaurants with genuinely quiet-sounding
reviews ("looking for a place with less crowd and silent...") that simply
weren't in the tiny hand-picked ground-truth set, out of a 9000+ restaurant
dataset with hundreds of plausibly-quiet restaurants. There's also at least
one real embedding limitation visible in a spot check: a review containing
"Ambience is quiet good" (a colloquial typo for "quite good", unrelated to
noise level) still surfaced for the "quiet" query - a genuine weakness,
separate from the ground-truth-size issue, worth keeping in mind when
reading Judged Relevance@5 too (an LLM judge can make the same "quiet"/
"quite" mistake a human skimming quickly might make).

`tests/test_semantic_retrieval.py` still does **not** assert a recall
threshold (see that file's docstring) - it's a structural/regression test,
not a quality gate, regardless of how the ground truth changes. Read the
actual Recall@5/Precision@5/Judged Relevance@5 numbers from `run_eval.py`'s
scorecard.

## Hybrid (filters + vibe) evaluation

The other two suites each exercise half the retriever: the structured
scenarios run with `vibe_query=None` so semantic search never executes, and
the semantic queries run with an empty `HybridFilters()` so no structured
predicate ever applies. Neither covers *"somewhere quiet in Whitefield under
Rs 1000"* — the most common shape a real user types, and the only path where
the two halves interact: where the filter restricts what semantic search may
rank, where the relaxation ladder can fire, and where rank fusion has both
signals to merge.

That gap had teeth. Two changes confined to this interaction — removing the
500-restaurant pool cap, and switching filtered searches from HNSW to exact
scanning — were invisible to the scorecard, which reported identical numbers
before and after by construction.

**Scenarios.** 8, drawn from real data, covering the selectivity range the
retriever behaves differently across: a narrow place+cuisine pool (134
restaurants), a broad place-only pool (Whitefield, 584 — which exceeded the
old cap), each numeric constraint alone, both together, two relaxation shapes,
and an impossible cuisine.

**Metrics — deliberately not recall.** The lesson from the semantic suite is
that a small hand-picked ground truth grades a retriever unfairly over 9,000
restaurants. This suite checks properties the path is supposed to *guarantee*
instead:

- *Containment* — every returned restaurant satisfies the hard constraints.
  Place and cuisine strictly (never relaxed); price and rating against the
  values the retriever reports it actually applied, so a legitimately relaxed
  search is judged on what it promised rather than what was first asked.
  Fully deterministic, no judge involved.
- *Evidence* — every returned restaurant carries review snippets, since a
  candidate with no supporting text cannot honestly answer a qualitative
  request.
- *Judged Relevance@5* — the same Gemini judge the semantic suite uses,
  scoring whether that evidence genuinely supports the vibe.

Unlike `tests/test_semantic_retrieval.py`, these tests **do** assert. That
isn't inconsistent: recall over a tiny ground truth is fuzzy and a low score
can be a bad draw, whereas containment is binary and a violation is a bug.

## LLM-as-judge

`judge.py` has two judges, both Gemini (`app.llm.gemini_client`) -
deliberately a different model family than Groq, which generates the
structured-scenario answers and (indirectly, via the retriever it feeds
into) shapes what semantic retrieval surfaces - so neither judge is scoring
output related to its own model family and biased toward its own phrasing.

- `judge_helpfulness` - each structured scenario's generated answer, one
  dimension (Helpfulness, 1-5), given the user's request, the candidate
  restaurants available, and the final answer.
- `judge_retrieval_relevance` - each semantic-retrieval query's top-5
  results, batched into one call per query, given the query and each
  result's best-matching review snippet, judged true/false independent of
  the approved ground-truth ID list (see "Semantic retrieval evaluation"
  above for why this exists alongside Recall@5/Precision@5).

An LLM judge is an approximate evaluation signal, not absolute ground truth -
treat its scores as a rough, noisy proxy for spotting regressions between
runs, not a precise or fully reproducible measurement. See the comments in
`judge.py`.

## Scorecard (`run_eval.py`)

Runs all three suites - the 10 structured scenarios (with the helpfulness
judge attached), the 6 semantic retrieval scenarios and the 8 hybrid
filters+vibe scenarios (both with the retrieval-relevance judge attached) -
and prints one scorecard: grounding rate,
relevance rate, avg LLM-judge helpfulness, avg latency (structured:
retrieval+generation; semantic: retrieval only, reported separately so the
two are never confused), semantic Recall@5/Precision@5/Judged Relevance@5,
and scenarios passed/failed for each suite. Also saves a JSON copy to
`evaluation/results/` per run.

```
python -m evaluation.run_eval
```

**Model comparison.** Reads `GROQ_MODEL` from the environment (never
hardcoded - `app.llm.groq_client.get_recommendation` reads it the same way
in production), so the exact same scenarios/ground-truth/scoring can be run
against different Groq models and the resulting JSON files diffed:

```
GROQ_MODEL=llama-3.3-70b-versatile python -m evaluation.run_eval
GROQ_MODEL=openai/gpt-oss-120b python -m evaluation.run_eval
```

## What's intentionally out of scope here

- Statistical/large-scale LLM eval (hundreds of scenarios, scored rubrics),
  a multi-dimensional judge rubric, an evaluation dashboard, Ragas,
  LangSmith, or a CI evaluation gate on these numbers — not warranted at
  this project's scale. 10 structured scenarios + 6 semantic-retrieval
  ground-truth queries + a single-dimension judge + the targeted
  hallucination canary test in `app/llm/tests` give solid, understandable
  confidence without the operational overhead of a larger eval system.
- Frontend component tests — the React UI is manually verified in-browser
  against the live backend.
- Query understanding's NL-parsing accuracy (intent classification,
  vibe-query extraction, reference resolution) — covered by
  `app/query_understanding/tests`.
