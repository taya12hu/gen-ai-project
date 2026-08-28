"""
Application / Interface Layer (backend).

FastAPI app exposing auth (register/login/me/forgot-reset-password) and the
chat endpoints (conversations, messages, streaming), behind a JWT dependency.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import anyio.to_thread
import psycopg2
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from app.auth.email import EmailSendError, send_password_reset_email
from app.auth.password_reset import InvalidResetTokenError, consume_reset_token, create_reset_token
from app.auth.tokens import InvalidTokenError, create_access_token, decode_access_token
from app.auth.users import (
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
    set_password,
)
from app.chat.service import (
    finalize_chat_turn,
    handle_chat_message,
    prepare_chat_turn,
    stream_chat_message_tokens,
)
from app.conversation.filters import FILTER_DIMENSIONS, clear_dimension, get_state
from app.conversation.preferences import delete_preference, list_preferences
from app.conversation.store import (
    ConversationNotFoundError,
    create_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)
from app.logging_config import set_request_id
from app.retrieval.hybrid import get_restaurants_by_ids, get_review_snippets_by_ids
from app.storage.db import MAX_POOL_SIZE, get_connection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Every route here is a sync def, so FastAPI runs it on AnyIO's worker
    # thread pool - which defaults to 40 threads regardless of how many
    # database connections exist. That mismatch meant concurrency past the
    # pool size didn't queue, it raised PoolError and failed requests.
    #
    # Capping the threadpool at the pool size makes the connection pool the
    # single place concurrency is bounded: excess requests wait for a worker
    # instead of racing to fail on a connection that isn't there. It's safe to
    # match exactly rather than halve because no request path holds two
    # pooled connections at once (see app.retrieval.hybrid).
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = MAX_POOL_SIZE
    logger.info("Request threadpool capped at %d workers to match the DB pool", MAX_POOL_SIZE)
    yield


app = FastAPI(title="AI Restaurant Recommendation Service", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tags every log line produced while serving this request.

    Without it, the pipeline's log lines - query understood, retrieval counts,
    which LLM provider served the reply, turn persisted - interleave across
    concurrent requests in one stream with no way to tell which turn each
    belongs to. An inbound X-Request-ID is honoured so a proxy's id carries
    through rather than being replaced.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Local dev origins are always allowed; FRONTEND_URL (already used for
# password-reset links) adds the deployed frontend's origin in production.
_allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_frontend_url = os.environ.get("FRONTEND_URL")
if _frontend_url and _frontend_url not in _allow_origins:
    _allow_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


# --- Auth models -----------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class MessageResponse(BaseModel):
    message: str


# --- Chat models ---------------------------------------------------------------


class ConversationOut(BaseModel):
    id: int
    title: str | None = None
    created_at: object


class ReviewSnippetOut(BaseModel):
    id: int
    text: str
    rating: float | None = None


class MatchedRestaurantOut(BaseModel):
    id: int
    name: str
    place: str
    city: str
    cuisines: list[str]
    price: float
    rating: float
    rest_type: str | None
    votes: int
    review_snippets: list[ReviewSnippetOut]


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    mentioned_restaurant_ids: list[int] | None = None
    mentioned_review_ids: list[int] | None = None
    matched_restaurants: list[MatchedRestaurantOut] = []
    created_at: object


class FilterChipOut(BaseModel):
    """One active constraint, ready to render. The label is written by the
    backend (see SearchState.as_chips) so the chip and the assistant's reply
    describe the same constraint the same way."""

    dimension: str
    label: str


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    conversation_id: int
    reply: str
    matched_restaurants: list[MatchedRestaurantOut]
    new_preferences: dict[str, str] = {}
    active_filters: list[FilterChipOut] = []


class PreferenceOut(BaseModel):
    key: str
    value: str
    updated_at: object


# --- Auth dependency ---------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")

    return user


# --- Health --------------------------------------------------------------------


@app.get("/health")
def health(response: Response):
    """Liveness *and* dependency readiness.

    This used to return {"status": "ok"} unconditionally, which made it
    actively misleading: it's wired up as the platform's healthCheckPath
    (see render.yaml), so an instance whose database was unreachable kept
    reporting healthy and kept being sent traffic it could only fail. A
    health check that can't fail isn't one.

    The query is deliberately trivial - this reports whether a connection can
    be acquired and used at all, not how the database is performing.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("select 1;")
                cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Health check failed: database unreachable (%s)", exc)
        response.status_code = 503
        return {"status": "degraded", "database": "unavailable"}

    return {"status": "ok", "database": "ok"}


# --- Auth routes ---------------------------------------------------------------


@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest):
    try:
        user = create_user(req.email, req.password, req.display_name)
    except EmailAlreadyRegisteredError:
        raise HTTPException(status_code=409, detail="Email already registered")

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(access_token=token, user=UserOut(**user))


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    try:
        user = authenticate_user(req.email, req.password)
    except InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(access_token=token, user=UserOut(**user))


@app.get("/auth/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    return UserOut(**current_user)


@app.post("/auth/forgot-password", response_model=MessageResponse)
def forgot_password(req: ForgotPasswordRequest):
    # Always return the same generic response whether or not the email is
    # registered - confirming/denying an email's existence here would let
    # an attacker enumerate registered accounts.
    generic_response = MessageResponse(
        message="If an account with that email exists, a password reset link has been sent."
    )

    user = get_user_by_email(req.email)
    if user is None:
        return generic_response

    token = create_reset_token(user["id"])
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
    reset_url = f"{frontend_url}/reset-password?token={token}"
    try:
        send_password_reset_email(user["email"], reset_url)
    except EmailSendError:
        # Don't leak "our email provider failed" (or "this email exists")
        # to the caller via a different status code/response shape - log it
        # server-side and still return the generic response. The token is
        # already issued, so a correctly-configured retry would still work.
        logger.exception("Failed to send password reset email to user_id=%s", user["id"])

    return generic_response


@app.post("/auth/reset-password", response_model=MessageResponse)
def reset_password(req: ResetPasswordRequest):
    try:
        user_id = consume_reset_token(req.token)
    except InvalidResetTokenError:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired")

    set_password(user_id, req.new_password)
    return MessageResponse(message="Password updated. You can now log in with your new password.")


# --- Preferences routes (protected) -----------------------------------------------


def _preference_not_found_detail() -> dict:
    return {"error_code": "preference_not_found", "message": "Preference not found"}


@app.get("/preferences", response_model=list[PreferenceOut])
def get_user_preferences(current_user: dict = Depends(get_current_user)):
    return [PreferenceOut(**p) for p in list_preferences(current_user["id"])]


@app.delete("/preferences/{key}", status_code=204, response_class=Response)
def forget_user_preference(key: str, current_user: dict = Depends(get_current_user)):
    if not delete_preference(current_user["id"], key):
        raise HTTPException(status_code=404, detail=_preference_not_found_detail())
    return Response(status_code=204)


# --- Chat routes (protected) ----------------------------------------------------


def _filter_not_found_detail(dimension: str) -> dict:
    return {
        "error_code": "unknown_filter",
        "message": f"Unknown filter {dimension!r}; expected one of {', '.join(FILTER_DIMENSIONS)}",
    }


@app.get("/chat/conversations/{conversation_id}/filters", response_model=list[FilterChipOut])
def get_chat_filters(conversation_id: int, current_user: dict = Depends(get_current_user)):
    try:
        get_conversation(conversation_id, current_user["id"])
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    return [FilterChipOut(**c) for c in get_state(conversation_id).as_chips()]


@app.delete("/chat/conversations/{conversation_id}/filters/{dimension}", response_model=list[FilterChipOut])
def clear_chat_filter(
    conversation_id: int, dimension: str, current_user: dict = Depends(get_current_user)
):
    """Drops one constraint and returns what remains.

    Returns the surviving chips rather than 204, so the UI updates from an
    authoritative list instead of guessing what removal left behind.
    """
    try:
        get_conversation(conversation_id, current_user["id"])
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    if dimension not in FILTER_DIMENSIONS:
        raise HTTPException(status_code=404, detail=_filter_not_found_detail(dimension))

    return [FilterChipOut(**c) for c in clear_dimension(conversation_id, dimension).as_chips()]


@app.post("/chat/conversations", response_model=ConversationOut, status_code=201)
def create_chat_conversation(current_user: dict = Depends(get_current_user)):
    conversation = create_conversation(current_user["id"])
    return ConversationOut(**conversation)


@app.get("/chat/conversations", response_model=list[ConversationOut])
def list_chat_conversations(current_user: dict = Depends(get_current_user)):
    return [ConversationOut(**c) for c in list_conversations(current_user["id"])]


def _hydrate_messages(messages: list[dict]) -> list[MessageOut]:
    """Attaches the restaurant/review data behind each message's persisted ids.

    A persisted message only carries ids (mentioned_restaurant_ids,
    mentioned_review_ids) - the full restaurant/review data shown live isn't
    stored, so a reloaded conversation has to look it up again. This replays
    the *exact* rows the live reply used (via the persisted ids), not a fresh
    top-rated/semantic re-selection: a structured-only query that showed no
    reviews live must show none on reload either.

    Lookups are batched across the whole conversation rather than run
    per-message. Hydrating one message at a time meant two queries (and two
    pool checkouts) per message, so opening a 40-message conversation issued
    80 round trips against a pool that holds 10 connections - the classic N+1,
    and enough to starve concurrent requests on its own.
    """
    all_restaurant_ids = {rid for m in messages for rid in (m["mentioned_restaurant_ids"] or [])}
    all_review_ids = {rid for m in messages for rid in (m["mentioned_review_ids"] or [])}

    restaurants_by_id = get_restaurants_by_ids(sorted(all_restaurant_ids))
    snippets_by_restaurant = get_review_snippets_by_ids(sorted(all_review_ids))

    hydrated: list[MessageOut] = []
    for m in messages:
        restaurant_ids = m["mentioned_restaurant_ids"] or []
        # Restrict each message to the review ids *it* recorded: two messages
        # can cite the same restaurant with different supporting snippets, and
        # the shared lookup above is keyed only by restaurant.
        message_review_ids = set(m["mentioned_review_ids"] or [])

        matched = []
        for restaurant_id in restaurant_ids:
            restaurant = restaurants_by_id.get(restaurant_id)
            if restaurant is None:
                continue
            snippets = [
                s for s in snippets_by_restaurant.get(restaurant_id, []) if s.id in message_review_ids
            ]
            matched.append(_matched_restaurant_out(restaurant, snippets))
        hydrated.append(MessageOut(**m, matched_restaurants=matched))
    return hydrated


@app.get("/chat/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def get_chat_messages(conversation_id: int, current_user: dict = Depends(get_current_user)):
    try:
        get_conversation(conversation_id, current_user["id"])
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    return _hydrate_messages(get_messages(conversation_id))


def _matched_restaurant_out(r, snippets=None) -> MatchedRestaurantOut:
    """`snippets` overrides the candidate's own attached snippets.

    The batched reload path (_hydrate_messages) shares one restaurant object
    across every message that cited it, so it passes each message's snippets
    in explicitly rather than assigning them onto the shared object - which
    would leak one message's evidence into another's card.
    """
    snippets = r.review_snippets if snippets is None else snippets
    return MatchedRestaurantOut(
        id=r.id,
        name=r.name,
        place=r.place,
        city=r.city,
        cuisines=r.cuisines,
        price=r.price,
        rating=r.rating,
        rest_type=r.rest_type,
        votes=r.votes,
        review_snippets=[ReviewSnippetOut(id=s.id, text=s.text, rating=s.rating) for s in snippets],
    )


# Stable, frontend-facing error codes for the chat endpoints. The frontend
# owns the actual user-facing wording (see frontend/src/lib/errorMessages.js)
# - "message" here is a plain-English fallback/log description only, not
# what a user necessarily sees, so wording can drift between the two without
# anything breaking.
def _conversation_not_found_detail() -> dict:
    return {"error_code": "conversation_not_found", "message": "Conversation not found"}


def _classify_chat_error(exc: Exception) -> tuple[int, dict]:
    """Turns whatever exception escaped the chat pipeline into a stable,
    safe-to-display error code instead of a raw provider/database exception
    string. get_recommendation/understand_query already try Gemini if Groq
    fails, so anything other than a DB error reaching here means BOTH
    providers failed - hence "llm_unavailable" by elimination, without this
    needing to enumerate every possible Groq/Gemini SDK exception type."""
    if isinstance(exc, psycopg2.Error):
        return 503, {
            "error_code": "database_unavailable",
            "message": "Database temporarily unavailable",
        }
    return 503, {"error_code": "llm_unavailable", "message": "AI assistant temporarily unavailable"}


@app.post("/chat/conversations/{conversation_id}/messages", response_model=ChatMessageResponse)
def send_chat_message(
    conversation_id: int, req: ChatMessageRequest, current_user: dict = Depends(get_current_user)
):
    try:
        reply, resolved_conversation_id = handle_chat_message(
            current_user["id"], conversation_id, req.message
        )
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    except Exception as exc:
        logger.exception("Chat message failed")
        status_code, detail = _classify_chat_error(exc)
        raise HTTPException(status_code=status_code, detail=detail)

    return ChatMessageResponse(
        conversation_id=resolved_conversation_id,
        reply=reply.reply_text,
        matched_restaurants=[_matched_restaurant_out(r) for r in reply.matched_restaurants],
        new_preferences=reply.new_preferences,
        active_filters=[FilterChipOut(**c) for c in get_state(resolved_conversation_id).as_chips()],
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/conversations/{conversation_id}/messages/stream")
def send_chat_message_stream(
    conversation_id: int, req: ChatMessageRequest, current_user: dict = Depends(get_current_user)
):
    # Runs synchronously here, NOT inside the generator below: a
    # StreamingResponse commits to a 200 status the moment it starts
    # sending, so an ownership check that can fail with 404 has to happen
    # before we return one, not from partway through a stream already in
    # flight.
    try:
        prepared = prepare_chat_turn(current_user["id"], conversation_id, req.message)
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    except Exception as exc:
        logger.exception("Chat prepare failed")
        status_code, detail = _classify_chat_error(exc)
        raise HTTPException(status_code=status_code, detail=detail)

    def event_stream():
        chunks: list[str] = []
        try:
            for chunk in stream_chat_message_tokens(prepared):
                chunks.append(chunk)
                yield _sse("token", {"text": chunk})
        except Exception as exc:  # a Groq/network failure mid-stream - can't raise HTTPException now
            logger.exception("Streaming chat reply failed")
            _, detail = _classify_chat_error(exc)
            yield _sse("error", detail)
            return

        # Finalizing has to be inside the generator's error handling too, not
        # just token streaming. It was previously outside, so a database blip
        # while persisting a *fully generated* reply let the exception escape
        # uncaught: the connection simply ended with neither a "done" nor an
        # "error" event, the client read that as a dropped stream, and its
        # retry logic regenerated the whole turn from scratch - a second LLM
        # call, and the answer the user had already watched arrive thrown
        # away.
        #
        # Persistence failing is also not a reason to withhold the reply. The
        # text is already on the user's screen; "done" carries the restaurant
        # cards that make it useful. The turn is lost from history either way,
        # and that's the smaller loss.
        reply_text = "".join(chunks)
        try:
            reply = finalize_chat_turn(prepared, reply_text)
            payload = {
                "matched_restaurants": [
                    _matched_restaurant_out(r).model_dump() for r in reply.matched_restaurants
                ],
                "new_preferences": reply.new_preferences,
                "active_filters": prepared.search_state.as_chips(),
            }
        except Exception:
            logger.exception(
                "Failed to finalize/persist a fully-streamed chat turn for conversation_id=%s",
                prepared.conversation_id,
            )
            payload = {"matched_restaurants": [], "new_preferences": {}, "persisted": False}

        yield _sse("done", payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
