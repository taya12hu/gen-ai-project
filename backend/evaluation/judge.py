"""
LLM-as-judge: single-dimension Helpfulness scoring for a generated chat
reply, and binary relevance judging for semantic retrieval results.

Uses Gemini (app.llm.gemini_client) for both - deliberately a different model
family than Groq, which generates the chat replies judged here - so the
judge isn't scoring output from its own model family and biased toward its
own style/phrasing.

NOTE: an LLM judge is an approximate evaluation signal, not absolute ground
truth. Treat scores as a rough, noisy proxy for spotting regressions between
runs, not as a precise or fully reproducible measurement.
"""

import json
import logging
import re
import time

from app.llm.gemini_client import get_recommendation_gemini

logger = logging.getLogger(__name__)

# Two distinct transient failure modes hit in practice, both worth a retry
# rather than an immediate skip: the Gemini free tier caps generateContent at
# 5 requests/minute per model (judge_helpfulness is called once per
# structured scenario, judge_retrieval_relevance once per semantic query,
# batched - see below - comfortably enough total calls to occasionally hit
# that limit in a full run); and Gemini occasionally returns 503 "high
# demand, try again later" independent of our own call volume. A fixed
# backoff (rather than parsing a suggested retry-after out of the error
# message) keeps this simple; a judge score is a nice-to-have signal, not
# worth a more elaborate retry policy.
TRANSIENT_ERROR_RETRY_SECONDS = 15.0
MAX_ATTEMPTS = 3

JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for a restaurant recommendation chat assistant.
You will be shown a user's request, the restaurant candidates the assistant
had available, and the assistant's final answer. Score ONLY how helpful,
relevant, and appropriate that answer is for the user's request.

Score on a single dimension, Helpfulness, from 1 to 5:
1 = not helpful / fails the request
2 = mostly unhelpful
3 = partially useful
4 = useful
5 = very helpful and directly addresses the request

Respond with ONLY the integer score (1-5) and nothing else - no explanation.
""".strip()

RETRIEVAL_JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for a restaurant search engine's semantic
retrieval step. You will be given a search query and a numbered list of
restaurants that were retrieved for it, each with the review excerpt that
caused it to be retrieved. For EACH restaurant, judge whether that specific
review evidence is genuinely relevant to the query - not whether the
restaurant sounds good overall, only whether the given evidence text
actually supports the query.

Respond with ONLY a JSON array of true/false booleans, one per restaurant,
in the same order given, and nothing else - e.g. [true, false, true].
""".strip()


def _call_judge_with_retry(messages: list[dict]) -> str | None:
    """Shared Gemini call + retry used by both judge functions below (see
    module docstring for why Gemini specifically, and the
    TRANSIENT_ERROR_RETRY_SECONDS comment above for why a fixed backoff is enough
    here). Returns None if the call ultimately fails - callers must treat
    that as "skip this judgment", never as a score of 0/false.

    Retries on 429 (rate limit) and 503 (Gemini's own "high demand, try
    again later" response) - both are the API telling us to back off and
    retry, not a permanent failure. Anything else (bad request, auth error,
    etc.) fails immediately since retrying wouldn't help."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return get_recommendation_gemini(messages)
        except Exception as exc:
            is_transient = any(marker in str(exc) for marker in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if is_transient and attempt < MAX_ATTEMPTS:
                logger.info("LLM judge call transiently failed (attempt %d/%d); retrying in %.0fs", attempt, MAX_ATTEMPTS, TRANSIENT_ERROR_RETRY_SECONDS)
                time.sleep(TRANSIENT_ERROR_RETRY_SECONDS)
                continue
            logger.warning("LLM judge call failed (%s); skipping this judgment", exc)
            return None
    return None


def _format_candidates(candidates) -> str:
    if not candidates:
        return "(none offered)"
    lines = [f"- {c.name} ({c.place}), {', '.join(c.cuisines)}, Rs {c.price:.0f} for two, {c.rating}★" for c in candidates]
    return "\n".join(lines)


def judge_helpfulness(user_query: str, candidates: list, answer: str) -> int | None:
    """Scores one generated answer 1-5 for helpfulness. Returns None (skip
    from averages, don't treat as 0) if the judge call fails or its response
    can't be parsed as a score."""
    prompt = (
        f"User request: {user_query}\n\n"
        f"Candidate restaurants available to the assistant:\n{_format_candidates(candidates)}\n\n"
        f"Assistant's answer:\n{answer}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = _call_judge_with_retry(messages)
    if response is None:
        return None

    match = re.search(r"[1-5]", response)
    if not match:
        logger.warning("LLM judge returned an unparseable response: %r", response)
        return None
    return int(match.group())


def judge_retrieval_relevance(query: str, candidates: list) -> list[bool | None]:
    """For a semantic retrieval query, judges whether each retrieved
    candidate's evidence (its best-matching review snippet) is genuinely
    relevant to the query - independent of any fixed ground-truth ID list,
    so it can grade results a strict/small ground truth wouldn't have
    approved even if they're perfectly reasonable answers (see
    EVALUATION.md's "Known finding" on why Recall@5 alone is limited here).

    All `candidates` for one query are judged in a SINGLE Gemini call (not
    one call per candidate) to stay well within the free-tier rate limit -
    6 calls total for this suite instead of 30.

    Returns one bool per candidate, same order as `candidates`. Returns a
    same-length list of None (never guessed True/False) if the judge call
    fails, or if its response can't be parsed into exactly one verdict per
    candidate.
    """
    if not candidates:
        return []

    lines = []
    for i, c in enumerate(candidates, start=1):
        snippet = c.review_snippets[0].text if c.review_snippets else "(no review text attached)"
        lines.append(f'{i}. {c.name} - review: "{snippet}"')
    prompt = f"Query: {query}\n\nRetrieved restaurants:\n" + "\n".join(lines)

    messages = [
        {"role": "system", "content": RETRIEVAL_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = _call_judge_with_retry(messages)
    if response is None:
        return [None] * len(candidates)

    array_match = re.search(r"\[.*\]", response, re.DOTALL)
    if not array_match:
        logger.warning("Retrieval judge returned no JSON array: %r", response)
        return [None] * len(candidates)

    try:
        parsed = json.loads(array_match.group())
    except json.JSONDecodeError:
        logger.warning("Retrieval judge returned unparseable JSON: %r", response)
        return [None] * len(candidates)

    if not isinstance(parsed, list) or len(parsed) != len(candidates) or not all(isinstance(v, bool) for v in parsed):
        logger.warning("Retrieval judge returned malformed array (expected %d bools): %r", len(candidates), parsed)
        return [None] * len(candidates)

    return parsed
