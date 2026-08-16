"""
Application / Interface Layer (backend).

FastAPI app exposing auth (register/login/me/forgot-reset-password) and the
chat endpoints (conversations, messages, streaming), behind a JWT dependency.
"""

import json
import logging
import os

import psycopg2
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from app.conversation.store import (
    ConversationNotFoundError,
    create_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)
from app.retrieval.hybrid import get_restaurants_by_ids, get_review_snippets_by_ids

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Restaurant Recommendation Service")

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


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    conversation_id: int
    reply: str
    matched_restaurants: list[MatchedRestaurantOut]


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
def health():
    return {"status": "ok"}


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


# --- Chat routes (protected) ----------------------------------------------------


@app.post("/chat/conversations", response_model=ConversationOut, status_code=201)
def create_chat_conversation(current_user: dict = Depends(get_current_user)):
    conversation = create_conversation(current_user["id"])
    return ConversationOut(**conversation)


@app.get("/chat/conversations", response_model=list[ConversationOut])
def list_chat_conversations(current_user: dict = Depends(get_current_user)):
    return [ConversationOut(**c) for c in list_conversations(current_user["id"])]


def _hydrate_message(m: dict) -> MessageOut:
    # A persisted message only carries ids (mentioned_restaurant_ids,
    # mentioned_review_ids) - the full restaurant/review data shown live
    # isn't stored, so a reloaded conversation has to look it up again. This
    # replays the *exact* rows the live reply used (via the persisted ids),
    # not a fresh top-rated/semantic re-selection - a structured-only query
    # that showed no reviews live must show none on reload either.
    restaurant_ids = m["mentioned_restaurant_ids"] or []
    review_ids = m["mentioned_review_ids"] or []
    restaurants_by_id = get_restaurants_by_ids(restaurant_ids)
    snippets_by_restaurant = get_review_snippets_by_ids(review_ids)

    matched = []
    for restaurant_id in restaurant_ids:
        restaurant = restaurants_by_id.get(restaurant_id)
        if restaurant is None:
            continue
        restaurant.review_snippets = snippets_by_restaurant.get(restaurant_id, [])
        matched.append(_matched_restaurant_out(restaurant))
    return MessageOut(**m, matched_restaurants=matched)


@app.get("/chat/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def get_chat_messages(conversation_id: int, current_user: dict = Depends(get_current_user)):
    try:
        get_conversation(conversation_id, current_user["id"])
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail=_conversation_not_found_detail())
    return [_hydrate_message(m) for m in get_messages(conversation_id)]


def _matched_restaurant_out(r) -> MatchedRestaurantOut:
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
        review_snippets=[ReviewSnippetOut(id=s.id, text=s.text, rating=s.rating) for s in r.review_snippets],
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

        reply = finalize_chat_turn(prepared, "".join(chunks))
        yield _sse(
            "done",
            {
                "matched_restaurants": [
                    _matched_restaurant_out(r).model_dump() for r in reply.matched_restaurants
                ]
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
