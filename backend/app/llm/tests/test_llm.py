"""
Tests for the Groq LLM client.

Model-resolution logic is tested with a fake Groq client (fast, free, no
network). Two integration tests make real calls to Groq to confirm the API
key/connection actually work and that responses stay grounded in a given
candidate list (not hallucinated).
"""

import pytest

from app.llm import groq_client as llm_client


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.last_call = None

    def create(self, model, messages, temperature):
        self.last_call = {"model": model, "messages": messages, "temperature": temperature}
        return FakeCompletion("fake response")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm_client, "get_client", lambda: client)
    return client


# --- Model resolution (no network) --------------------------------------------


def test_uses_explicit_model_when_given(fake_client, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "env-model")
    llm_client.get_recommendation([{"role": "user", "content": "hi"}], model="explicit-model")
    assert fake_client.chat.completions.last_call["model"] == "explicit-model"


def test_falls_back_to_env_var_when_no_model_given(fake_client, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "env-model")
    llm_client.get_recommendation([{"role": "user", "content": "hi"}])
    assert fake_client.chat.completions.last_call["model"] == "env-model"


def test_falls_back_to_default_model_when_no_model_or_env(fake_client, monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    llm_client.get_recommendation([{"role": "user", "content": "hi"}])
    assert fake_client.chat.completions.last_call["model"] == llm_client.DEFAULT_MODEL


def test_passes_messages_through_unchanged(fake_client):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    llm_client.get_recommendation(messages)
    assert fake_client.chat.completions.last_call["messages"] == messages


def test_returns_message_content(fake_client):
    result = llm_client.get_recommendation([{"role": "user", "content": "hi"}])
    assert result == "fake response"


# --- Real Groq integration tests ----------------------------------------------


def test_get_client_constructs_without_error():
    client = llm_client.get_client()
    assert client is not None


def test_real_call_returns_nonempty_response():
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Say 'hello' and nothing else."},
    ]
    response = llm_client.get_recommendation(messages)
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_real_call_stays_grounded_in_candidate_list():
    # A deliberately distinctive, made-up name a model wouldn't know unless
    # it actually read the candidate list we gave it.
    distinctive_name = "Zzyzx Noodle Palace"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a restaurant recommendation assistant. Recommend ONLY "
                "restaurants that appear in the candidate list below. Never invent, "
                "assume, or reference any restaurant not in that list."
            ),
        },
        {
            "role": "user",
            "content": (
                "User preferences:\n- Place: Indiranagar\n- Cuisines: Chinese\n\n"
                "Candidate restaurants (ranked by rating):\n"
                f"1. {distinctive_name} — Chinese; Rs 500 for two; 4.5★ (100 votes); Casual Dining\n\n"
                "Based only on the candidates above, recommend the best restaurant(s) "
                "for this user and explain briefly why each recommendation fits."
            ),
        },
    ]

    response = llm_client.get_recommendation(messages)
    assert distinctive_name in response
