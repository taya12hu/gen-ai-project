"""
Phase 10 - Application / Interface Layer (backend).

FastAPI app that wires together the full pipeline: validates and normalizes
user preferences (Phase 4), retrieves a candidate shortlist (Phase 5),
builds a prompt (Phase 6), calls the LLM (Phase 7), and formats the response
(Phase 8) - exposed as a POST /recommend endpoint the React frontend calls.

Also exposes the Phase 9 auth endpoints (register/login/me) and protects
/recommend behind a valid JWT, via the get_current_user dependency.
"""

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, ValidationError

BACKEND_DIR = Path(__file__).resolve().parent.parent
for phase in [
    "phase4_preference_input",
    "phase5_retrieval_engine",
    "phase6_prompt_construction",
    "phase7_llm_engine",
    "phase8_response_formatting",
    "phase9_authentication",
]:
    sys.path.insert(0, str(BACKEND_DIR / phase))

from preferences import UnknownValueError, normalize_preferences  # noqa: E402
from known_values import get_known_cuisines, get_known_places  # noqa: E402
from retrieval import get_candidates  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402
from llm_client import get_recommendation  # noqa: E402
from response_formatter import format_response, format_for_display  # noqa: E402
from auth import InvalidTokenError, create_access_token, decode_access_token  # noqa: E402
from user_store import (  # noqa: E402
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    authenticate_user,
    create_user,
    get_user_by_id,
)

app = FastAPI(title="AI Restaurant Recommendation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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


# --- Recommendation models ---------------------------------------------------


class RecommendRequest(BaseModel):
    place: str
    cuisines: list[str] = Field(min_length=1)
    max_price: float | None = None
    min_rating: float | None = None


class RestaurantOut(BaseModel):
    id: int
    name: str
    place: str
    city: str
    cuisines: list[str]
    price: float
    rating: float
    rest_type: str | None
    votes: int


class RecommendResponse(BaseModel):
    found_any: bool
    relaxed: bool
    matched_restaurants: list[RestaurantOut]
    explanation: str
    display_text: str


class OptionsResponse(BaseModel):
    places: list[str]
    cuisines: list[str]


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


# --- Health / options ---------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options", response_model=OptionsResponse)
def options():
    return OptionsResponse(
        places=sorted(get_known_places()),
        cuisines=sorted(get_known_cuisines()),
    )


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


# --- Recommendation route (protected) -----------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, current_user: dict = Depends(get_current_user)):
    known_places = get_known_places()
    known_cuisines = get_known_cuisines()

    try:
        prefs = normalize_preferences(
            req.place, req.cuisines, req.max_price, req.min_rating, known_places, known_cuisines
        )
    except UnknownValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    result = get_candidates(prefs)

    if result.candidates:
        messages = build_prompt(prefs, result)
        llm_text = get_recommendation(messages)
    else:
        llm_text = "No matching restaurants were found for your preferences."

    rec = format_response(result, llm_text)

    return RecommendResponse(
        found_any=rec.found_any,
        relaxed=rec.relaxed,
        matched_restaurants=rec.matched_restaurants,
        explanation=rec.explanation,
        display_text=format_for_display(rec),
    )
