"""Tests for review ingestion parsing."""

from app.reviews.ingest import parse_reviews


def test_parse_reviews_dedupes_repeated_text():
    # Mirrors the real dataset artifact: the same review tripled in one cell.
    raw = repr(
        [
            ("Rated 4.0", "RATED\n  Great food and quick service."),
            ("Rated 4.0", "RATED\n  Great food and quick service."),
            ("Rated 4.0", "RATED\n  Great food and quick service."),
            ("Rated 5.0", "RATED\n  Best filter coffee in town."),
        ]
    )

    reviews = parse_reviews(raw)

    assert reviews == [
        (4.0, "Great food and quick service."),
        (5.0, "Best filter coffee in town."),
    ]


def test_parse_reviews_keeps_first_rating_for_duplicate_text():
    raw = repr(
        [
            ("Rated 3.0", "RATED\n  Same text, different rating first time."),
            ("Rated 5.0", "RATED\n  Same text, different rating first time."),
        ]
    )

    reviews = parse_reviews(raw)

    assert reviews == [(3.0, "Same text, different rating first time.")]


def test_parse_reviews_skips_blank_and_handles_missing_rating():
    raw = repr(
        [
            (None, "RATED\n  No star rating attached."),
            ("Rated 4.5", "RATED\n   "),
        ]
    )

    reviews = parse_reviews(raw)

    assert reviews == [(None, "No star rating attached.")]


def test_parse_reviews_empty_or_invalid_input():
    assert parse_reviews("") == []
    assert parse_reviews("not a python literal") == []
