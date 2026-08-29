# DineMind

An AI chat assistant that helps you find restaurants in Bangalore. Ask for what you want
in plain language ("suggest a quiet vegetarian place under ₹1000") and it
grounds its answer in real restaurant data and real customer reviews,
remembering your stated preferences across conversations.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works under the hood.

## Features

- Natural language chat, not a filter form: cuisine, place, price, and
  rating are parsed out of what you type
- Semantic search over real review text for descriptive requests like
  "good for a date" or "quiet"
- Remembers preferences you mention (e.g. "I'm vegetarian") across sessions
- Multi-turn conversations with history, so you can say "what about
  something cheaper?" and it knows what you're referring to
- Streamed replies (token by token), with an automatic fallback LLM
  provider if the primary one errors out
- Email/password auth with forgot/reset password flows

## Tech stack

- **Backend:** Python, FastAPI, PostgreSQL with pgvector, Groq (primary LLM)
  and Gemini (fallback LLM)
- **Frontend:** React, Vite

## Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL with the `vector` extension available (pgvector)
- A [Groq](https://console.groq.com) API key
- Optional: a [Gemini](https://aistudio.google.com/apikey) API key (LLM
  fallback) and a [Resend](https://resend.com) API key (password reset
  email)

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate       # on Windows
# source .venv/bin/activate  # on macOS/Linux

pip install -e .
pip install -r requirements.txt

copy .env.example .env       # on Windows
# cp .env.example .env       # on macOS/Linux
```

Fill in `.env` with your database credentials, `GROQ_API_KEY`, and
`JWT_SECRET_KEY` at minimum. `GEMINI_API_KEY` and the `RESEND_*` keys are
optional; without them, the Gemini fallback and password reset email simply
won't work.

Set up the database schemas (each is idempotent and safe to re-run):

```bash
python -m app.storage.schema
python -m app.auth.schema
python -m app.conversation.schema
python -m app.reviews.schema
```

Fetch and load the restaurant dataset. Every step is re-runnable: the load
upserts on `(name, place)` rather than truncating, so restaurant ids stay
stable and existing conversations keep pointing at the right rows.


```bash
python -m app.data.acquisition
python -m app.data.cleaning
python -m app.storage.load
python -m app.reviews.ingest
python -m app.reviews.embed
```

Run the API:

```bash
uvicorn app.api.main:app --reload
```

The API is now available at `http://localhost:8000`.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app is now available at `http://localhost:5173`.

## Running tests

```bash
cd backend
pytest
```

## Evaluation

Scenario-based grounding and relevance checks against the live pipeline
live in [backend/evaluation/EVALUATION.md](backend/evaluation/EVALUATION.md).
