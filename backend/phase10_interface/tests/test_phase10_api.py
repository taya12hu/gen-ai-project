"""
Tests for Phase 10 - Application / Interface Layer (backend API).

Uses FastAPI's TestClient against the real app, which in turn hits the
live database, Groq API, and Phase 9 auth logic - this exercises the full
wired-together pipeline through the actual HTTP interface, including that
/recommend is now behind a login requirement.
"""

import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PHASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE_DIR.parent / "phase3_storage_indexing"))

from api import app  # noqa: E402
from db import get_connection  # noqa: E402

client = TestClient(app)


@pytest.fixture
def auth_headers():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post(
        "/auth/register", json={"email": email, "password": "password123", "display_name": "Test User"}
    )
    token = response.json()["access_token"]

    yield {"Authorization": f"Bearer {token}"}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


# --- Health / options (unauthenticated) --------------------------------------


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_options_returns_places_and_cuisines():
    response = client.get("/options")
    assert response.status_code == 200

    data = response.json()
    assert "Indiranagar" in data["places"]
    assert "Chinese" in data["cuisines"]
    assert len(data["places"]) > 0
    assert len(data["cuisines"]) > 0


# --- Auth: register / login / me ----------------------------------------------


def test_register_creates_user_and_returns_token():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post("/auth/register", json={"email": email, "password": "password123"})

    assert response.status_code == 201
    data = response.json()
    assert data["user"]["email"] == email
    assert len(data["access_token"]) > 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


def test_register_duplicate_email_returns_409(auth_headers):
    # auth_headers fixture already registered a user; grab its email via /auth/me
    me = client.get("/auth/me", headers=auth_headers).json()

    response = client.post("/auth/register", json={"email": me["email"], "password": "whatever123"})
    assert response.status_code == 409


def test_register_short_password_returns_422():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    response = client.post("/auth/register", json={"email": email, "password": "short"})
    assert response.status_code == 422


def test_register_invalid_email_returns_422():
    response = client.post("/auth/register", json={"email": "not-an-email", "password": "password123"})
    assert response.status_code == 422


def test_login_succeeds_with_correct_credentials():
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "password123"})

    response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    assert response.json()["user"]["email"] == email

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("delete from users where email = %s;", (email,))
        conn.commit()
    finally:
        conn.close()


def test_login_fails_with_wrong_password(auth_headers):
    me = client.get("/auth/me", headers=auth_headers).json()

    response = client.post("/auth/login", json={"email": me["email"], "password": "wrong-password"})
    assert response.status_code == 401


def test_login_fails_for_unregistered_email():
    response = client.post(
        "/auth/login", json={"email": "nobody-registered@example.com", "password": "whatever123"}
    )
    assert response.status_code == 401


def test_me_returns_current_user(auth_headers):
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert "email" in response.json()


def test_me_without_token_returns_401():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_with_invalid_token_returns_401():
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


# --- /recommend: auth requirement ---------------------------------------------


def test_recommend_without_token_returns_401():
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0},
    )
    assert response.status_code == 401


def test_recommend_with_invalid_token_returns_401():
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401


# --- /recommend: behavior once authenticated ----------------------------------


def test_recommend_valid_request_returns_matched_restaurants(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 200

    data = response.json()
    assert data["found_any"] is True
    assert data["relaxed"] is False
    assert len(data["matched_restaurants"]) > 0
    assert len(data["explanation"]) > 0
    assert len(data["display_text"]) > 0

    for r in data["matched_restaurants"]:
        assert r["place"] == "Indiranagar"
        assert "Chinese" in r["cuisines"]
        assert r["price"] <= 800
        assert r["rating"] >= 4.0


def test_recommend_is_case_insensitive_on_place_and_cuisine(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "indiranagar", "cuisines": ["chinese"], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["found_any"] is True
    if data["matched_restaurants"]:
        assert data["matched_restaurants"][0]["place"] == "Indiranagar"


def test_recommend_unknown_place_returns_400(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Atlantis", "cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Atlantis" in response.json()["detail"]


def test_recommend_unknown_cuisine_returns_400(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Klingon"], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Klingon" in response.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_price", -100),
        ("max_price", 0),
        ("min_rating", -1),
        ("min_rating", 10),
    ],
)
def test_recommend_out_of_range_values_return_422(auth_headers, field, value):
    payload = {"place": "Indiranagar", "cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0}
    payload[field] = value

    response = client.post("/recommend", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_recommend_missing_place_returns_422(auth_headers):
    response = client.post(
        "/recommend",
        json={"cuisines": ["Chinese"], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_recommend_empty_cuisines_list_returns_422(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": [], "max_price": 800, "min_rating": 4.0},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_recommend_without_price_or_rating_still_returns_results(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese"]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["found_any"] is True


def test_recommend_with_multiple_cuisines_matches_any(auth_headers):
    response = client.post(
        "/recommend",
        json={"place": "Indiranagar", "cuisines": ["Chinese", "Cafe"], "max_price": 2000, "min_rating": 0},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["found_any"] is True
    for r in data["matched_restaurants"]:
        assert "Chinese" in r["cuisines"] or "Cafe" in r["cuisines"]


def test_cors_headers_present_for_allowed_origin():
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
