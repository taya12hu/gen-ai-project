"""
Tests for known place/cuisine values (live database).
"""

import pytest

from app.retrieval.known_values import get_known_cuisines, get_known_places


@pytest.fixture(scope="module")
def known_places():
    return get_known_places()


@pytest.fixture(scope="module")
def known_cuisines():
    return get_known_cuisines()


def test_known_places_is_non_empty_and_contains_indiranagar(known_places):
    assert len(known_places) > 0
    assert "Indiranagar" in known_places


def test_known_cuisines_is_non_empty_and_contains_chinese(known_cuisines):
    assert len(known_cuisines) > 0
    assert "Chinese" in known_cuisines
