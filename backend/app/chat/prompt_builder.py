"""
Chat Prompt Construction.

Turns a QueryUnderstanding + retrieved candidates (already grounded -
structured facts + real review snippets) + prior turns + the preferences
actually applied to this turn into a Groq chat message list. Prior turns are
passed through as real user/assistant messages (native multi-turn), not
flattened into text - only the *current* turn carries the retrieval context,
appended to the latest user message.

`preferences` here is deliberately the *applied* subset for this turn (see
app.chat.service.prepare_chat_turn), not everything ever remembered about
the user - the LLM is told about a preference only when it actually shaped
this search, so it never claims to have used something it didn't.
"""

SYSTEM_PROMPT = """
You are a friendly restaurant recommendation chat assistant. You have access
to real restaurant data and real customer review excerpts pulled from a
database - use only what's given to you.

Rules:
1. Only ever recommend or discuss restaurants that appear in the "Candidate
   restaurants" (or "Restaurant being discussed") section of the latest
   message. Never invent a restaurant, or state a fact (price/rating/
   cuisine/location) not given there.
2. When review snippets are shown under a restaurant, use them as your
   evidence for qualitative claims (ambience, service, food quality, good
   for a date, etc.) - phrase it like "reviewers mention..." grounded in
   that text. Don't invent sentiment the snippets don't support, and don't
   claim a quality (e.g. "quiet") if no snippet actually supports it.
3. If the user is just stating a lasting preference about themselves (e.g.
   "I'm vegetarian", "I usually like quiet places") and isn't asking for a
   recommendation right now, warmly acknowledge that you'll remember it -
   don't force a restaurant suggestion into that reply.
4. If there are no candidates because nothing matched, say so plainly and
   suggest what to relax (place, cuisine, budget, rating).
5. If told that price/rating constraints were relaxed to produce the
   candidate list, mention that plainly.
6. If the message is a greeting or unrelated to restaurants, respond
   briefly and invite a restaurant question.
7. Keep replies conversational and concise - this is a chat, not a report.
   Refer to restaurants by name exactly as given.
8. If a "Preferences applied to this search" section is given below, those
   remembered facts genuinely shaped which restaurants you're seeing - say
   so plainly and briefly (e.g. "Since you mentioned preferring vegetarian
   food, I focused on..."). Never claim a preference was used if that
   section isn't present.
""".strip()


def _format_snippets(snippets) -> str:
    if not snippets:
        return ""
    lines = [f'   - "{s.text}"' + (f" (rated {s.rating})" if s.rating is not None else "") for s in snippets]
    return "\n   Review excerpts:\n" + "\n".join(lines)


def _format_candidate(candidate, index: int) -> str:
    cuisines = ", ".join(candidate.cuisines)
    rest_type = f"; {candidate.rest_type}" if candidate.rest_type else ""
    header = (
        f"{index}. {candidate.name} ({candidate.place}) — {cuisines}; "
        f"Rs {candidate.price:.0f} for two; "
        f"{candidate.rating}★ ({candidate.votes} votes){rest_type}"
    )
    return header + _format_snippets(candidate.review_snippets)


def build_chat_prompt(
    user_message: str,
    understanding,
    candidates: list,
    relaxed: bool,
    recent_messages: list[dict],
    preferences: dict[str, str],
    referenced_restaurant=None,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in recent_messages:
        messages.append({"role": m["role"], "content": m["content"]})

    context_parts: list[str] = []

    if preferences:
        pref_lines = "\n".join(f"- {k}: {v}" for k, v in preferences.items())
        context_parts.append(f"Preferences applied to this search:\n{pref_lines}")

    if referenced_restaurant is not None:
        context_parts.append("Restaurant being discussed:\n" + _format_candidate(referenced_restaurant, 1))
    elif candidates:
        candidate_block = "\n".join(_format_candidate(c, i) for i, c in enumerate(candidates, start=1))
        context_parts.append(f"Candidate restaurants:\n{candidate_block}")
        if relaxed:
            context_parts.append(
                "Note: no restaurant fully satisfied both the budget and minimum rating, so those "
                "constraints were relaxed to still show relevant options."
            )
    elif understanding.intent in ("search", "followup_question"):
        context_parts.append("Candidate restaurants: (none matched)")

    context_block = "\n\n".join(context_parts)
    user_content = f"{context_block}\n\nUser: {user_message}" if context_block else user_message

    messages.append({"role": "user", "content": user_content})
    return messages
