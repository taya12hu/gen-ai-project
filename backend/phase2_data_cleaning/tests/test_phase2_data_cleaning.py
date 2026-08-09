"""
Tests for Phase 2 - Data Cleaning & Normalization.

Covers the parsing functions directly (unit tests for the quirky raw formats
found in the dataset) and the produced artifact (data/processed/restaurants_clean.csv).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PHASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = PHASE_DIR / "data" / "processed" / "restaurants_clean.csv"
RAW_CSV = PHASE_DIR.parent / "phase1_data_acquisition" / "data" / "raw" / "train.csv"

sys.path.insert(0, str(PHASE_DIR))

import clean_data  # noqa: E402


# --- Unit tests: parse_rating -----------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("4.1/5", 4.1),
        ("3.9 /5", 3.9),   # inconsistent spacing found in the raw data
        ("3.0/5", 3.0),
        (" 4.5/5 ", 4.5),
    ],
)
def test_parse_rating_valid_formats(raw, expected):
    assert clean_data.parse_rating(raw) == expected


@pytest.mark.parametrize("raw", ["NEW", "-", None, float("nan"), "garbage"])
def test_parse_rating_missing_or_placeholder_values(raw):
    assert clean_data.parse_rating(raw) is None


# --- Unit tests: parse_price -------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("800", 800.0),
        ("1,200", 1200.0),
        ("1,000", 1000.0),
    ],
)
def test_parse_price_valid_formats(raw, expected):
    assert clean_data.parse_price(raw) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), "N/A"])
def test_parse_price_missing_or_invalid(raw):
    assert clean_data.parse_price(raw) is None


# --- Unit tests: parse_cuisines ----------------------------------------------


def test_parse_cuisines_trims_and_rejoins():
    assert (
        clean_data.parse_cuisines("Chinese, North Indian ,Thai")
        == "Chinese, North Indian, Thai"
    )


def test_parse_cuisines_single_value():
    assert clean_data.parse_cuisines("North Indian") == "North Indian"


@pytest.mark.parametrize("raw", [None, float("nan"), ""])
def test_parse_cuisines_missing(raw):
    assert clean_data.parse_cuisines(raw) is None


# --- Unit tests: clean_text ----------------------------------------------


def test_clean_text_trims_whitespace():
    assert clean_data.clean_text("  Banashankari  ") == "Banashankari"


@pytest.mark.parametrize("raw", [None, float("nan"), "   "])
def test_clean_text_missing_or_blank(raw):
    assert clean_data.clean_text(raw) is None


# --- Integration tests: produced artifact -------------------------------------


def test_processed_csv_exists():
    assert PROCESSED_CSV.exists(), (
        f"{PROCESSED_CSV} not found — run `python clean_data.py` first."
    )


@pytest.fixture(scope="module")
def clean_df():
    return pd.read_csv(PROCESSED_CSV)


def test_processed_csv_has_expected_columns(clean_df):
    expected = {"name", "place", "city", "cuisines", "price", "rating", "rest_type", "votes"}
    assert expected.issubset(set(clean_df.columns))


def test_no_nulls_in_critical_columns(clean_df):
    critical = ["name", "place", "cuisines", "price", "rating"]
    for col in critical:
        assert clean_df[col].isna().sum() == 0, f"{col} has null values"


def test_no_duplicate_name_place_pairs(clean_df):
    dupes = clean_df.duplicated(subset=["name", "place"]).sum()
    assert dupes == 0


def test_rating_within_valid_range(clean_df):
    assert clean_df["rating"].between(0, 5).all()


def test_price_is_positive(clean_df):
    assert (clean_df["price"] > 0).all()


def test_cuisines_have_no_stray_whitespace(clean_df):
    # No leading/trailing space around any comma-separated cuisine token.
    bad = clean_df["cuisines"].str.contains(r",\S|\s,", regex=True, na=False)
    assert not bad.any()


def test_row_count_is_smaller_than_raw_due_to_dedup_and_dropna(clean_df):
    raw_row_count = sum(1 for _ in open(RAW_CSV, encoding="utf-8")) - 1
    assert 0 < len(clean_df) < raw_row_count
