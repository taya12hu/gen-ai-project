"""
LLM Recommendation Engine (fallback provider: Gemini).

Gemini backs up Groq: if a Groq call errors out (rate limit, outage, bad
key), groq_client.py and query_understanding catch it and retry here
instead of failing the whole request. Kept as its own module - rather than
folded into groq_client.py - because it also serves query understanding's
JSON-mode call, and both callers only need three functions, not the SDK
client itself.

Groq's messages format ({"role": "system"/"user"/"assistant", "content":
...}) doesn't match Gemini's: Gemini takes the system prompt separately
(system_instruction) and uses "model" rather than "assistant" for the
model's own turns. _split_messages converts between the two so callers
never have to know which provider actually served a given request.
"""

import logging
import os
from collections.abc import Iterator

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-flash-latest"

logger = logging.getLogger(__name__)


class GeminiNotConfiguredError(Exception):
    """Raised when GEMINI_API_KEY isn't set - distinguishes "fallback isn't
    configured yet" from an actual Gemini API failure."""


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiNotConfiguredError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=api_key)


def _resolved_model(model: str | None) -> str:
    return model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _split_messages(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Groq-style messages -> (system_instruction, Gemini contents)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in messages
        if m["role"] != "system"
    ]
    return system_instruction, contents


def get_recommendation_gemini(messages: list[dict], model: str | None = None) -> str:
    resolved_model = _resolved_model(model)
    logger.info("Requesting Gemini completion (model=%s, messages=%d)", resolved_model, len(messages))
    client = get_client()
    system_instruction, contents = _split_messages(messages)

    response = client.models.generate_content(
        model=resolved_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3),
    )
    return response.text


def stream_recommendation_gemini(messages: list[dict], model: str | None = None) -> Iterator[str]:
    client = get_client()
    system_instruction, contents = _split_messages(messages)

    for chunk in client.models.generate_content_stream(
        model=_resolved_model(model),
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3),
    ):
        if chunk.text:
            yield chunk.text


def get_json_completion_gemini(messages: list[dict], model: str | None = None) -> str:
    """Same shape as Groq's response_format={"type": "json_object"} mode -
    used by query understanding's fallback."""
    logger.info("Requesting Gemini JSON completion (model=%s)", _resolved_model(model))
    client = get_client()
    system_instruction, contents = _split_messages(messages)

    response = client.models.generate_content(
        model=_resolved_model(model),
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return response.text
