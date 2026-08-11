"""
Tests for Phase 1 - Data Acquisition.

Verifies that fetch_data.py is wired up correctly and that the raw dataset
artifacts it produces (train.csv, schema.json) are present and well-formed.
Does not hit the network: it checks the artifacts already produced by
running `fetch_data.py`, plus the script's own configuration.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PHASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PHASE_DIR / "data" / "raw"
TRAIN_CSV = RAW_DIR / "train.csv"
SCHEMA_JSON = RAW_DIR / "schema.json"

EXPECTED_DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"

# Columns the rest of the pipeline (cleaning, filtering) depends on.
REQUIRED_COLUMNS = {
    "rate",  # -> rating
    "location",  # -> place
    "listed_in(city)",  # -> place (broader)
    "cuisines",  # -> cuisine
    "approx_cost(for two people)",  # -> price
    "name",
}

sys.path.insert(0, str(PHASE_DIR))


def test_fetch_script_targets_correct_dataset():
    import fetch_data

    assert fetch_data.DATASET_ID == EXPECTED_DATASET_ID


def test_fetch_script_raw_dir_resolves_under_phase_data_raw():
    import fetch_data

    assert fetch_data.RAW_DIR == RAW_DIR


def test_raw_csv_exists():
    assert TRAIN_CSV.exists(), (
        f"{TRAIN_CSV} not found — run `python fetch_data.py` first."
    )


def test_raw_csv_is_not_empty():
    df = pd.read_csv(TRAIN_CSV, nrows=5)
    assert len(df) > 0


def test_raw_csv_has_required_columns():
    header = pd.read_csv(TRAIN_CSV, nrows=0)
    missing = REQUIRED_COLUMNS - set(header.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_raw_csv_row_count_matches_schema():
    # Some columns (reviews_list, dish_liked, menu_item) are free-text and can
    # contain embedded newlines within quoted CSV fields, so a naive line count
    # would overcount. Use pandas' CSV parser, which handles quoting correctly.
    schema = json.loads(SCHEMA_JSON.read_text())
    expected_rows = schema["train"]["num_rows"]

    row_count = len(pd.read_csv(TRAIN_CSV))
    assert row_count == expected_rows


def test_schema_json_exists_and_is_valid():
    assert SCHEMA_JSON.exists()
    schema = json.loads(SCHEMA_JSON.read_text())
    assert "train" in schema
    assert schema["train"]["num_rows"] > 0
    assert REQUIRED_COLUMNS.issubset(schema["train"]["columns"].keys())


@pytest.mark.parametrize(
    "value",
    ["4.1/5", "3.8/5"],
)
def test_sample_rate_values_look_parseable(value):
    # Sanity check on the raw 'rate' format assumed by Phase 2 cleaning.
    assert value.endswith("/5")
    float(value.split("/")[0])
