"""
Phase 6 - Prompt Construction.

Turns a UserPreferences (Phase 4) + RetrievalResult (Phase 5) into a chat
message list ready to send to an LLM (Phase 7). Builds the prompt only -
no API calls here. The system prompt constrains the model to recommend
only from the given candidate list, so it can't hallucinate restaurants
that aren't actually in our data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4_preference_input"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase5_retrieval_engine"))

from preferences import UserPreferences  # noqa: E402
from retrieval import RetrievalResult  # noqa: E402

SYSTEM_PROMPT = """
You are a restaurant recommendation assistant. You will be given a user's
preferences and a shortlist of candidate restaurants pulled from a database.

Rules:
1. Recommend ONLY restaurants that appear in the candidate list below. Never
   invent, assume, or reference any restaurant not in that list.
2. Pick the single best match for the user's preferences (or up to 3 if
   several are similarly strong), and briefly explain why each fits.
3. Ground every claim (price, rating, cuisine, location) in the data
   provided - do not guess or embellish beyond it.
4. If the candidate list is empty, say clearly that no restaurant matched
   and suggest the user relax their preferences (e.g. wider budget, lower
   minimum rating).
5. If told that price/rating constraints were relaxed to produce this list,
   mention that to the user and explain the candidates are the closest
   available match for their place and cuisine, not a guaranteed fit on
   budget/rating.
6. Keep the tone friendly and concise.
""".strip()


def _format_candidate(candidate: dict, index: int) -> str:
    cuisines = ", ".join(candidate["cuisines"])
    rest_type = f"; {candidate['rest_type']}" if candidate.get("rest_type") else ""
    return (
        f"{index}. {candidate['name']} — {cuisines}; "
        f"Rs {candidate['price']:.0f} for two; "
        f"{candidate['rating']}★ ({candidate['votes']} votes){rest_type}"
    )


def build_prompt(prefs: UserPreferences, result: RetrievalResult) -> list[dict]:
    preference_lines = [
        f"- Place: {prefs.place}",
        f"- Cuisines: {', '.join(prefs.cuisines)}",
    ]
    if prefs.max_price is not None:
        preference_lines.append(f"- Budget (max price for two): Rs {prefs.max_price:.0f}")
    if prefs.min_rating is not None:
        preference_lines.append(f"- Minimum rating: {prefs.min_rating}")
    preference_block = "\n".join(preference_lines)

    if result.candidates:
        candidate_block = "\n".join(
            _format_candidate(c, i) for i, c in enumerate(result.candidates, start=1)
        )
    else:
        candidate_block = "(no candidates found matching this place and cuisine)"

    relaxed_note = ""
    if result.relaxed:
        relaxed_note = (
            "\nNote: no restaurant fully satisfied both the budget and minimum rating, "
            "so those constraints were relaxed to still show relevant options for the "
            "requested place and cuisine. Mention this to the user.\n"
        )

    user_content = (
        f"User preferences:\n{preference_block}\n"
        f"{relaxed_note}\n"
        f"Candidate restaurants (ranked by rating):\n{candidate_block}\n\n"
        "Based only on the candidates above, recommend the best restaurant(s) for "
        "this user and explain briefly why each recommendation fits their preferences."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
