"""
LLM prompts - untrusted content handling.

Two things reach our prompts that we did not write: what the user types, and
review text quoted out of the Zomato dataset. Both arrive as plain text and
get interpolated next to our own instructions, which is the setup for prompt
injection - direct in the first case, and *indirect* in the second, where a
review in the corpus containing instruction-shaped text would be read by the
model as if the system had said it.

The blast radius here is genuinely small: the chat pipeline calls no tools,
performs no writes on the model's say-so, and holds no secrets in context, so
the worst realistic outcome is a wrong or off-topic reply rather than an
action taken on an attacker's behalf. That's a reason to keep the mitigation
proportionate, not a reason to skip it.

Two cheap, complementary measures, used by both prompt-building sites
(app.query_understanding.understanding and app.chat.prompt_builder):

- `fence()` wraps untrusted text in explicit delimiters and strips those
  delimiters from the content first. Stripping is the part that makes it a
  fence rather than decoration: a marker the author of the content can
  reproduce lets them close the fence early, and everything after their
  forged terminator reads as trusted prompt again.
- `truncate()` bounds how much untrusted text can enter at once, so a wall of
  injected instructions can't crowd the real ones out of the context window.

Both are paired with an explicit instruction in each system prompt telling
the model that fenced text is data. Neither is a guarantee - defence in depth
in front of a model that ultimately decides for itself - which is why the
grounding check after generation (app.chat.response_formatter) stays the
thing that actually constrains what a user can be shown.
"""

import logging

logger = logging.getLogger(__name__)

OPEN = "<<<UNTRUSTED_INPUT"
CLOSE = "UNTRUSTED_INPUT>>>"

# Long enough for any genuine restaurant question; short enough that a wall of
# injected text can't dominate the prompt.
MAX_MESSAGE_CHARS = 2000

# Review text is quoted verbatim into the candidate block, up to 3 snippets
# per restaurant across 5 restaurants. Individual Zomato reviews run long, and
# an over-long one is padding rather than evidence.
MAX_SNIPPET_CHARS = 600


def strip_markers(text: str) -> str:
    """Removes fence delimiters so content can't terminate its own fence."""
    return text.replace(OPEN, "").replace(CLOSE, "")


def truncate(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Caps length, logging when it bites so an unusual input is visible."""
    if len(text) > max_chars:
        logger.info("Truncating %d chars of untrusted text to %d", len(text), max_chars)
        return text[:max_chars].rstrip() + "…"
    return text


def sanitize(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """strip_markers + truncate, for content that is embedded inline (e.g. a
    quoted review snippet) rather than given its own fenced block."""
    return truncate(strip_markers(text), max_chars)


def fence(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Wraps untrusted text in delimiters, after sanitizing it."""
    return f"{OPEN}\n{sanitize(text, max_chars)}\n{CLOSE}"
