# Phase 12 — Testing & Evaluation

## What this phase covers

Phases 1-11 each have their own unit/integration tests, scoped to that
phase (including Phase 9's auth logic and Phase 10's protected-route
tests). Phase 12 evaluates the core recommendation pipeline as a whole,
running real preference scenarios end to end (Phase 4 → Phase 8) and
checking the two things that matter most for an LLM-backed recommendation
service:

1. **Grounding** — every restaurant the LLM recommends is actually one we
   gave it (`id` present in the Phase 5 candidate list), never invented.
2. **Relevance** — what it recommends genuinely satisfies the requested
   place, cuisine, and (when Phase 5 didn't have to relax) price/rating.

## Scenarios

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

## Result

All 10 passed — see `tests/test_phase12_evaluation.py`. No hallucinated
restaurants, no irrelevant recommendations, and all edge cases (relaxed
constraints, zero matches, multi-cuisine overlap, omitted price/rating)
were handled correctly without errors.

## What's intentionally out of scope here

- Statistical/large-scale LLM eval (hundreds of scenarios, scored rubrics) —
  not warranted at this project's scale; 8 well-chosen scenarios plus the
  targeted hallucination canary test in Phase 7 give solid confidence.
- Frontend component tests — the React UI was manually verified in-browser
  by the user against the live backend.
