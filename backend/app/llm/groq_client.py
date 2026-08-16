"""
LLM Recommendation Engine (primary provider: Groq).

Sends a chat prompt to Groq's chat completion API and returns the model's
natural-language recommendation. This is the only place in the pipeline
that calls an external LLM directly for the final reply - though if Groq
itself errors out (rate limit, outage, bad key), it falls back to Gemini
(gemini_client.py) rather than failing the request outright.
"""

import logging
import os
from collections.abc import Iterator

from groq import Groq

from app.llm.gemini_client import get_recommendation_gemini, stream_recommendation_gemini

DEFAULT_MODEL = "openai/gpt-oss-120b"

logger = logging.getLogger(__name__)


def get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def get_recommendation(messages: list[dict], model: str | None = None) -> str:
    """
    Sends the given chat messages to Groq and returns the model's text
    response. Falls back to Gemini if the Groq call itself fails for any
    reason.
    """
    client = get_client()
    resolved_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    logger.info("Requesting Groq completion (model=%s, messages=%d)", resolved_model, len(messages))

    try:
        completion = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        logger.warning("Groq call failed (%s); falling back to Gemini", exc)
        return get_recommendation_gemini(messages)


def stream_recommendation(messages: list[dict], model: str | None = None) -> Iterator[str]:
    """
    Same call as get_recommendation, but with stream=True: Groq sends the
    reply incrementally as it's generated instead of all at once, and this
    yields each text fragment as it arrives so a caller can forward it to a
    client in real time rather than waiting for the full reply to finish.

    The fallback only applies to *starting* the stream - Groq's rate-limit/
    auth/connection errors surface on the initial call, before any chunk is
    yielded, so switching providers there never produces a reply that's part
    Groq, part Gemini. A failure after streaming has already begun (rare) is
    not retried, to avoid exactly that kind of mixed, confusing output.
    """
    client = get_client()
    resolved_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    logger.info("Requesting streamed Groq completion (model=%s, messages=%d)", resolved_model, len(messages))

    try:
        stream = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=0.3,
            stream=True,
        )
    except Exception as exc:
        logger.warning("Groq streaming call failed (%s); falling back to Gemini", exc)
        yield from stream_recommendation_gemini(messages)
        return

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
