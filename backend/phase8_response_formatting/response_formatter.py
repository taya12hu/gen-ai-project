"""
Phase 8 - Response Formatting & Output.

Turns the LLM's raw text (Phase 7) into a consistent, trustworthy response:
cross-references restaurant names the LLM actually mentioned against the
candidate list (Phase 5) so the price/rating/cuisine facts shown to the user
always come from our database, not whatever numbers the LLM restated in
prose. Also produces a clean, user-friendly display string, including a
graceful message when nothing matched.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase5_retrieval_engine"))

from retrieval import RetrievalResult  # noqa: E402


@dataclass
class Recommendation:
    matched_restaurants: list[dict]  # candidates the LLM actually mentioned, DB-sourced facts
    explanation: str  # the LLM's natural-language text
    relaxed: bool  # whether price/rating constraints had to be relaxed (Phase 5)
    found_any: bool  # whether any candidates existed at all


def format_response(result: RetrievalResult, llm_text: str) -> Recommendation:
    matched = [c for c in result.candidates if c["name"] in llm_text]

    return Recommendation(
        matched_restaurants=matched,
        explanation=llm_text.strip(),
        relaxed=result.relaxed,
        found_any=len(result.candidates) > 0,
    )


def format_for_display(rec: Recommendation) -> str:
    if not rec.found_any:
        return (
            "No restaurants matched your preferences. Try widening your budget, "
            "lowering the minimum rating, or picking a different place or cuisine."
        )

    lines: list[str] = []

    if rec.relaxed:
        lines.append(
            "Note: showing the closest matches — no restaurant fully met both "
            "your budget and minimum rating."
        )
        lines.append("")

    for r in rec.matched_restaurants:
        cuisines = ", ".join(r["cuisines"])
        lines.append(f"{r['name']} ({r['place']})")
        lines.append(
            f"  {cuisines} | Rs {r['price']:.0f} for two | {r['rating']} rating ({r['votes']} votes)"
        )
        lines.append("")

    lines.append(rec.explanation)
    return "\n".join(lines).strip()
