# Chat Pipeline Evaluation

## What this covers

Every domain under `app/` has its own unit/integration tests, scoped to
that module (including auth's logic and the API layer's protected-route
tests). This evaluates the core hybrid-retrieval + chat-generation pipeline
as a whole, running real scenarios end to end (structured retrieval ->
chat prompt construction -> Groq -> response formatting) and checking the
two things that matter most for an LLM-backed recommendation service:

1. **Grounding** — every restaurant the LLM talks about is actually one it
   was given (`id` present in the retrieved candidate list), never invented.
2. **Relevance** — what it recommends genuinely satisfies the requested
   place, cuisine, and (when retrieval didn't have to relax) price/rating.

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

These deliberately skip query understanding's NL-parsing stage (its own
accuracy is covered by `app/query_understanding/tests`) and build
`HybridFilters` directly, so this suite's focus stays on grounding and
relevance rather than intent extraction.

## What's intentionally out of scope here

- Statistical/large-scale LLM eval (hundreds of scenarios, scored rubrics) —
  not warranted at this project's scale; 10 well-chosen scenarios plus the
  targeted hallucination canary test in `app/llm/tests` give solid
  confidence.
- Frontend component tests — the React UI is manually verified in-browser
  against the live backend.
- Query understanding's NL-parsing accuracy (intent classification,
  vibe-query extraction, reference resolution) — covered by
  `app/query_understanding/tests`.
