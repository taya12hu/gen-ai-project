"""
Hybrid (filters + vibe) retrieval evaluation.

The third retrieval path, and until now the unmeasured one. The existing two
suites each exercise half of the retriever:

- `tests/test_evaluation.py` runs structured scenarios with `vibe_query=None`,
  so semantic search never executes.
- `semantic_retrieval.py` runs vibe queries with an empty `HybridFilters()`,
  so no structured predicate ever applies.

Neither covers a message like "somewhere quiet in Whitefield under Rs 1000",
which is the most common shape a real user types and the only path where the
two halves interact - where the structured filter restricts what semantic
search may rank, where the relaxation ladder can fire, and where rank fusion
has both signals to merge. Changes confined to that interaction (the pool cap
removal, the switch to exact scanning on filtered searches) were invisible to
the scorecard, which is what this suite fixes.

**What it measures.** Deliberately not recall against hand-picked ids - the
lesson from `semantic_retrieval.py` is that a tiny ground truth over 9,000
restaurants grades a retriever unfairly. Instead it checks the properties this
path is actually supposed to guarantee:

1. *Containment* - every returned restaurant satisfies the hard constraints.
   Place and cuisine are checked strictly (they are never relaxed); price and
   rating are checked against the values the retriever reports it actually
   applied, so a legitimately relaxed search is judged on what it promised
   rather than on what was originally asked. This is the property that makes
   "under Rs 800 in Koramangala" a constraint rather than a suggestion, and it
   is fully deterministic - no judge needed.
2. *Evidence* - when a vibe query is given, every returned restaurant carries
   review snippets. A candidate with no supporting text cannot honestly be
   presented as matching a qualitative request.
3. *Judged relevance* - whether that evidence genuinely supports the vibe,
   scored by the same LLM judge the semantic suite uses.
4. *Weak-evidence flagging* - whether the retriever noticed its own evidence
   was too far off to support a qualitative claim. This suite is what found
   that problem: relaxed scenarios scored 0/8 on judged relevance because a
   relaxed filter produces a pool small enough that every review clears the
   semantic cut regardless of what it says. Tracking the flag here means a
   regression in that detection shows up as a number rather than as a reply
   that quietly over-claims.

Retrieval only, no Groq/Gemini call, so this stays fast and isolates retrieval
from generation. The judging happens in `run_eval.py`, keeping this module's
"no LLM call" property intact - the same split `semantic_retrieval.py` uses.
"""

import time
from dataclasses import dataclass, field

from app.retrieval.hybrid import HybridFilters, get_hybrid_candidates

# Matches SEARCH_LIMIT in app.chat.service, so results are comparable with the
# other two suites and with what a real chat turn sees.
RETRIEVAL_K = 5


@dataclass(frozen=True)
class HybridScenario:
    """One filters-plus-vibe request, with what we expect the retriever to do.

    `expect_relaxed=None` means the scenario doesn't assert either way.
    """

    place: str | None
    cuisines: tuple[str, ...]
    max_price: float | None
    min_rating: float | None
    vibe_query: str
    expect_found: bool
    expect_relaxed: bool | None = None
    note: str = ""

    def filters(self) -> HybridFilters:
        return HybridFilters(
            place=self.place,
            cuisines=list(self.cuisines),
            max_price=self.max_price,
            min_rating=self.min_rating,
        )

    def label(self) -> str:
        parts = [self.place or "anywhere"]
        if self.cuisines:
            parts.append("/".join(self.cuisines))
        if self.max_price is not None:
            parts.append(f"<=Rs{self.max_price:.0f}")
        if self.min_rating is not None:
            parts.append(f">={self.min_rating}")
        return f"{' '.join(parts)} + \"{self.vibe_query}\""


# Drawn from real data (see the review-coverage query in EVALUATION.md), chosen
# to cover the selectivity range the retriever behaves differently across:
# a narrow place+cuisine pool, a broad place-only pool, each constraint type on
# its own, both together, two relaxation shapes, and an impossible filter.
HYBRID_SCENARIOS: tuple[HybridScenario, ...] = (
    HybridScenario(
        "Indiranagar", ("North Indian",), None, None,
        "quiet place good for conversation",
        expect_found=True, expect_relaxed=False,
        note="narrow pool (134 restaurants), no numeric constraints",
    ),
    HybridScenario(
        "Whitefield", (), None, None,
        "quiet place away from crowds",
        expect_found=True, expect_relaxed=False,
        note="broad pool (584) - exceeded the old 500 pool cap, so its lowest-rated members were unreachable",
    ),
    HybridScenario(
        "BTM", ("Chinese",), 500, None,
        "good value for money",
        expect_found=True, expect_relaxed=False,
        note="budget constraint only",
    ),
    HybridScenario(
        "HSR", ("North Indian",), None, 4.0,
        "friendly and attentive staff",
        expect_found=True, expect_relaxed=False,
        note="rating constraint only",
    ),
    HybridScenario(
        "Whitefield", ("North Indian",), 1000, 4.0,
        "great ambience for a date",
        expect_found=True, expect_relaxed=False,
        note="both numeric constraints, both satisfiable",
    ),
    HybridScenario(
        "HSR", ("North Indian",), 300, 4.7,
        "great ambience for a date",
        expect_found=True, expect_relaxed=True,
        note="relaxation: widens budget and lowers rating together",
    ),
    HybridScenario(
        "Kaggadasapura", ("Chinese",), 800, 4.9,
        "cosy and quiet",
        expect_found=True, expect_relaxed=True,
        note="relaxation: rating floor dropped (nothing there is rated above 4.0)",
    ),
    HybridScenario(
        "Indiranagar", ("Klingon Cuisine",), None, None,
        "quiet",
        expect_found=False,
        note="impossible cuisine - must return nothing, NOT fall back to an unfiltered vibe search",
    ),
)


@dataclass
class HybridQueryResult:
    scenario: HybridScenario
    candidates: list  # RestaurantCandidate objects - evidence for the judge, filled in by run_eval
    relaxation_note: str | None
    relaxed: bool
    evidence_weak: bool
    violations: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    judged_relevant: list[bool | None] | None = None

    @property
    def contained(self) -> bool:
        return not self.violations

    @property
    def with_evidence(self) -> int:
        return sum(1 for c in self.candidates if c.review_snippets)

    @property
    def evidence_rate(self) -> float:
        if not self.candidates:
            return 0.0
        return self.with_evidence / len(self.candidates)

    @property
    def passed(self) -> bool:
        return not self.violations


def _check(scenario: HybridScenario, result) -> list[str]:
    """Every way this scenario's outcome could be wrong, as plain sentences."""
    violations: list[str] = []
    candidates = result.candidates

    if scenario.expect_found and not candidates:
        violations.append("expected candidates, got none")
    if not scenario.expect_found and candidates:
        # The dangerous failure: silently widening an impossible query into an
        # unfiltered vibe search, which returns plausible-looking results that
        # ignore what the user asked for.
        violations.append(f"expected no candidates, got {len(candidates)}")
    if scenario.expect_relaxed is not None and result.relaxed is not scenario.expect_relaxed:
        violations.append(f"expected relaxed={scenario.expect_relaxed}, got {result.relaxed}")

    # Price and rating are judged against what the retriever says it applied,
    # not against the original request - relaxing is allowed, misreporting it
    # is not.
    applied = result.relaxation
    effective_price = applied.used_max_price if applied is not None else scenario.max_price
    effective_rating = applied.used_min_rating if applied is not None else scenario.min_rating

    for c in candidates:
        if scenario.place and c.place != scenario.place:
            violations.append(f"{c.name}: place {c.place!r} != {scenario.place!r}")
        if scenario.cuisines and not set(scenario.cuisines) & set(c.cuisines):
            violations.append(f"{c.name}: cuisines {c.cuisines} miss {list(scenario.cuisines)}")
        if effective_price is not None and float(c.price) > float(effective_price):
            violations.append(f"{c.name}: price {c.price} over applied limit {effective_price}")
        if effective_rating is not None and float(c.rating) < float(effective_rating):
            violations.append(f"{c.name}: rating {c.rating} under applied floor {effective_rating}")
        if not c.review_snippets:
            violations.append(f"{c.name}: no review evidence for a vibe query")

    return violations


def run_scenario(scenario: HybridScenario, k: int = RETRIEVAL_K) -> HybridQueryResult:
    """Runs one scenario through the production retrieval entry point."""
    start = time.monotonic()
    result = get_hybrid_candidates(scenario.filters(), scenario.vibe_query, limit=k)
    latency = time.monotonic() - start

    return HybridQueryResult(
        scenario=scenario,
        candidates=result.candidates,
        relaxation_note=result.relaxation_note(),
        relaxed=result.relaxed,
        evidence_weak=result.evidence_is_weak,
        violations=_check(scenario, result),
        latency_seconds=latency,
    )


def run_all(k: int = RETRIEVAL_K) -> list[HybridQueryResult]:
    return [run_scenario(s, k=k) for s in HYBRID_SCENARIOS]
