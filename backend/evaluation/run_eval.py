"""
Evaluation Scorecard.

Runs both evaluation suites against the live pipeline and prints one
scorecard:

1. The existing 10-scenario structured retrieval + generation suite
   (evaluation/tests/test_evaluation.py) - reused as-is (SCENARIOS,
   run_scenario), not duplicated.
2. The semantic ("vibe") retrieval suite (evaluation/semantic_retrieval.py)
   against the manually approved ground truth (evaluation/ground_truth.py).

Each structured scenario's generated answer is also scored by an LLM judge
(evaluation/judge.py, Gemini - see that module's docstring for why a
different model than Groq) on a single Helpfulness (1-5) dimension.

Semantic retrieval results get a second, independent quality signal beyond
the strict ground-truth Recall@5/Precision@5: judge_retrieval_relevance
(also evaluation/judge.py) judges each retrieved result's evidence directly,
with no fixed ID list involved, so it can credit a result the small
hand-picked ground truth wouldn't have (see EVALUATION.md's "Known finding").

Latency is measured with time.monotonic():
- structured scenarios: the full pipeline relevant to that turn (retrieval
  + Groq/Gemini generation) - that's what a real chat turn actually waits on.
- semantic retrieval scenarios: retrieval only, since nothing is generated
  on that path - reported as a separate number so the two are never averaged
  together or confused for one another.

Usage (from backend/):
    python -m evaluation.run_eval

To compare models, set GROQ_MODEL before each run - this script never
hardcodes a model; app.llm.groq_client.get_recommendation reads GROQ_MODEL
itself, exactly as it does in production:
    GROQ_MODEL=llama-3.3-70b-versatile python -m evaluation.run_eval
    GROQ_MODEL=openai/gpt-oss-120b python -m evaluation.run_eval
Each run also writes a JSON scorecard to evaluation/results/ so two runs can
be diffed later. Scenarios, ground truth, and scoring logic never change
between runs - only GROQ_MODEL does.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.llm.groq_client import DEFAULT_MODEL as DEFAULT_GROQ_MODEL
from app.reviews.embedding_model import get_embedder
from evaluation.hybrid_retrieval import run_all as run_all_hybrid_scenarios
from evaluation.judge import judge_helpfulness, judge_retrieval_relevance
from evaluation.semantic_retrieval import run_all_semantic_scenarios
from evaluation.tests.test_evaluation import SCENARIOS, run_scenario

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class StructuredScenarioResult:
    place: str
    cuisines: list[str]
    passed: bool
    failures: list[str]
    grounded: bool | None
    relevant: bool | None
    latency_seconds: float
    judge_helpfulness: int | None


def _check_structured_scenario(
    place, cuisines, max_price, min_rating, expect_found, expect_relaxed
) -> StructuredScenarioResult:
    """Same checks as test_evaluation.py's test_scenario_grounding_and_relevance,
    but collecting pass/fail instead of asserting, so one bad scenario doesn't
    stop the scorecard from reporting the rest."""
    start = time.monotonic()
    filters, result, reply = run_scenario(place, cuisines, max_price, min_rating)
    latency = time.monotonic() - start

    failures = []
    found_any = len(result.candidates) > 0
    if found_any is not expect_found:
        failures.append(f"expected found={expect_found}, got {found_any}")
    if expect_relaxed is not None and result.relaxed is not expect_relaxed:
        failures.append(f"expected relaxed={expect_relaxed}, got {result.relaxed}")

    grounded = None
    relevant = None
    judge_score = None

    if expect_found:
        grounded = len(reply.matched_restaurants) > 0
        if not grounded:
            failures.append("no matched restaurants (ungrounded reply)")

        candidate_ids = {c.id for c in result.candidates}
        relevant = True
        for r in reply.matched_restaurants:
            if r.id not in candidate_ids:
                relevant = False
                failures.append(f"restaurant {r.id} not among offered candidates")
            if r.place != filters.place:
                relevant = False
                failures.append(f"restaurant {r.id} place mismatch ({r.place} != {filters.place})")
            if not any(c in r.cuisines for c in filters.cuisines):
                relevant = False
                failures.append(f"restaurant {r.id} cuisine mismatch")
            if not result.relaxed:
                if filters.max_price is not None and r.price > filters.max_price:
                    relevant = False
                    failures.append(f"restaurant {r.id} over budget")
                if filters.min_rating is not None and r.rating < filters.min_rating:
                    relevant = False
                    failures.append(f"restaurant {r.id} under min rating")

        user_message = f"Suggest a {', '.join(cuisines)} restaurant in {place}"
        judge_score = judge_helpfulness(user_message, result.candidates, reply.reply_text)

    return StructuredScenarioResult(
        place=place,
        cuisines=cuisines,
        passed=not failures,
        failures=failures,
        grounded=grounded,
        relevant=relevant,
        latency_seconds=latency,
        judge_helpfulness=judge_score,
    )


def _avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_structured_suite() -> list[StructuredScenarioResult]:
    return [_check_structured_scenario(*scenario) for scenario in SCENARIOS]


def run_hybrid_suite():
    """The filters-plus-vibe path - the only one where the structured and
    semantic halves interact, and the one neither other suite touches."""
    get_embedder()  # warm the model so per-scenario latency reflects steady state
    results = run_all_hybrid_scenarios()
    for r in results:
        if r.candidates:
            r.judged_relevant = judge_retrieval_relevance(r.scenario.vibe_query, r.candidates)
    return results


def _hybrid_result_to_dict(r) -> dict:
    """Manual serialization, same reasoning as _semantic_result_to_dict: the
    full RestaurantCandidate objects are judge evidence, not scorecard data."""
    return {
        "scenario": r.scenario.label(),
        "note": r.scenario.note,
        "retrieved_ids": [c.id for c in r.candidates],
        "relaxed": r.relaxed,
        "relaxation_note": r.relaxation_note,
        "evidence_weak": r.evidence_weak,
        "contained": r.contained,
        "violations": r.violations,
        "evidence_rate": r.evidence_rate,
        "judged_relevant": r.judged_relevant,
        "latency_seconds": r.latency_seconds,
    }


def run_semantic_suite():
    get_embedder()  # warm the model once so per-query latency reflects steady state, not cold model load
    results = run_all_semantic_scenarios()
    for r in results:
        r.judged_relevant = judge_retrieval_relevance(r.query, r.retrieved_candidates)
    return results


def _judged_relevance_rate(semantic_results) -> float | None:
    """Fraction of judged-True among all judged (non-None) verdicts, across
    every candidate of every query - a single number summarizing the
    ground-truth-independent relevance signal."""
    verdicts = [v for r in semantic_results for v in (r.judged_relevant or []) if v is not None]
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v) / len(verdicts)


def _semantic_result_to_dict(r) -> dict:
    """Manual (not asdict()) serialization for the JSON scorecard - excludes
    retrieved_candidates (full RestaurantCandidate objects, only needed
    in-process as judge evidence) so the saved JSON stays lean instead of
    duplicating restaurant/review data already implied by retrieved_ids."""
    return {
        "query": r.query,
        "ground_truth_ids": r.ground_truth_ids,
        "retrieved_ids": r.retrieved_ids,
        "recall_at_k": r.recall_at_k,
        "precision_at_k": r.precision_at_k,
        "judged_relevant": r.judged_relevant,
        "latency_seconds": r.latency_seconds,
    }


def build_scorecard(structured_results, semantic_results, hybrid_results, groq_model: str) -> dict:
    structured_total = len(structured_results)
    structured_passed = sum(r.passed for r in structured_results)
    grounding_rate = _avg([1.0 if r.grounded else 0.0 for r in structured_results if r.grounded is not None])
    relevance_rate = _avg([1.0 if r.relevant else 0.0 for r in structured_results if r.relevant is not None])
    avg_judge = _avg([r.judge_helpfulness for r in structured_results if r.judge_helpfulness is not None])
    avg_structured_latency = _avg([r.latency_seconds for r in structured_results])

    semantic_total = len(semantic_results)
    semantic_hits = sum(1 for r in semantic_results if r.recall_at_k > 0)
    avg_recall = _avg([r.recall_at_k for r in semantic_results])
    avg_precision = _avg([r.precision_at_k for r in semantic_results])
    avg_judged_relevance = _judged_relevance_rate(semantic_results)
    avg_semantic_latency = _avg([r.latency_seconds for r in semantic_results])

    hybrid_total = len(hybrid_results)
    hybrid_passed = sum(r.passed for r in hybrid_results)
    hybrid_contained = sum(r.contained for r in hybrid_results)
    returned = [c for r in hybrid_results for c in r.candidates]
    hybrid_evidence_rate = _avg([1.0 if c.review_snippets else 0.0 for c in returned])
    hybrid_judged = _judged_relevance_rate(hybrid_results)
    hybrid_weak = sum(1 for x in hybrid_results if x.evidence_weak)
    avg_hybrid_latency = _avg([r.latency_seconds for r in hybrid_results])

    return {
        "groq_model": groq_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "structured_scenarios": {
            "total": structured_total,
            "passed": structured_passed,
            "failed": structured_total - structured_passed,
            "grounding_rate": grounding_rate,
            "relevance_rate": relevance_rate,
            "avg_judge_helpfulness": avg_judge,
            "avg_latency_seconds_pipeline": avg_structured_latency,
            "details": [asdict(r) for r in structured_results],
        },
        "semantic_retrieval_scenarios": {
            "total": semantic_total,
            "with_at_least_one_hit_at_5": semantic_hits,
            "avg_recall_at_5": avg_recall,
            "avg_precision_at_5": avg_precision,
            "avg_judged_relevance_at_5": avg_judged_relevance,
            "avg_latency_seconds_retrieval_only": avg_semantic_latency,
            "details": [_semantic_result_to_dict(r) for r in semantic_results],
        },
        "hybrid_scenarios": {
            "total": hybrid_total,
            "passed": hybrid_passed,
            "failed": hybrid_total - hybrid_passed,
            "containment_rate": _avg([1.0 if r.contained else 0.0 for r in hybrid_results]),
            "scenarios_fully_contained": hybrid_contained,
            "restaurants_returned": len(returned),
            "evidence_rate": hybrid_evidence_rate,
            "avg_judged_relevance_at_5": hybrid_judged,
            "scenarios_flagged_weak_evidence": hybrid_weak,
            "avg_latency_seconds_retrieval_only": avg_hybrid_latency,
            "details": [_hybrid_result_to_dict(r) for r in hybrid_results],
        },
    }


def _fmt(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_scorecard(scorecard: dict) -> None:
    s = scorecard["structured_scenarios"]
    v = scorecard["semantic_retrieval_scenarios"]

    print()
    print("=" * 64)
    print(f"EVALUATION SCORECARD  (GROQ_MODEL={scorecard['groq_model']})")
    print("=" * 64)
    print()
    print("Structured retrieval + generation (10 scenarios, full pipeline):")
    print(f"  Passed / total                    : {s['passed']}/{s['total']}")
    print(f"  Grounding rate                     : {_fmt(s['grounding_rate'])}")
    print(f"  Relevance rate                     : {_fmt(s['relevance_rate'])}")
    print(f"  Avg LLM-judge helpfulness (1-5)    : {_fmt(s['avg_judge_helpfulness'], 2)}")
    print(f"  Avg latency (retrieval+generation) : {_fmt(s['avg_latency_seconds_pipeline'])}s")
    print()
    print("Semantic ('vibe') retrieval (6 approved ground-truth queries, retrieval only):")
    print(f"  Queries with >=1 hit@5             : {v['with_at_least_one_hit_at_5']}/{v['total']}")
    print(f"  Avg Recall@5 (strict ID match)     : {_fmt(v['avg_recall_at_5'])}")
    print(f"  Avg Precision@5 (strict ID match)  : {_fmt(v['avg_precision_at_5'])}")
    print(f"  Avg Judged Relevance@5 (LLM judge) : {_fmt(v['avg_judged_relevance_at_5'])}")
    print(f"  Avg latency (retrieval only)       : {_fmt(v['avg_latency_seconds_retrieval_only'])}s")
    print()
    h = scorecard["hybrid_scenarios"]
    print(f"Hybrid filters + vibe ({h['total']} scenarios, retrieval only):")
    print(f"  Passed / total                     : {h['passed']}/{h['total']}")
    print(f"  Containment rate                   : {_fmt(h['containment_rate'])}")
    evidence_label = f"  Evidence rate ({h['restaurants_returned']} returned)"
    print(f"{evidence_label:<37}: {_fmt(h['evidence_rate'])}")
    print(f"  Avg Judged Relevance@5 (LLM judge) : {_fmt(h['avg_judged_relevance_at_5'])}")
    print(f"  Flagged as weak evidence           : {h['scenarios_flagged_weak_evidence']}/{h['total']}")
    print(f"  Avg latency (retrieval only)       : {_fmt(h['avg_latency_seconds_retrieval_only'])}s")
    print()
    print("=" * 64)


def save_scorecard(scorecard: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    model_slug = scorecard["groq_model"].replace("/", "-")
    path = RESULTS_DIR / f"{model_slug}_{int(time.time())}.json"
    path.write_text(json.dumps(scorecard, indent=2))
    return path


def main() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)  # quiet the pipeline's own per-call INFO logging

    groq_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    print("Running structured retrieval + generation scenarios (calls Groq/Gemini)...")
    structured_results = run_structured_suite()

    print("Running semantic retrieval scenarios (retrieval only, no LLM call)...")
    semantic_results = run_semantic_suite()

    print("Running hybrid filters+vibe scenarios (retrieval only, no LLM call)...")
    hybrid_results = run_hybrid_suite()

    scorecard = build_scorecard(structured_results, semantic_results, hybrid_results, groq_model)
    print_scorecard(scorecard)

    path = save_scorecard(scorecard)
    print(f"Saved JSON scorecard to {path}")


if __name__ == "__main__":
    main()
