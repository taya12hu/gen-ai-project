"""
Phase 7 - LLM Recommendation Engine.

Sends the prompt built in Phase 6 to Groq's chat completion API and returns
the model's natural-language recommendation. This is the only place in the
pipeline that calls an external LLM.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def get_recommendation(messages: list[dict], model: str | None = None) -> str:
    """
    Sends the chat messages (system + user prompt from Phase 6) to Groq and
    returns the model's text response.
    """
    client = get_client()
    resolved_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    completion = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content
