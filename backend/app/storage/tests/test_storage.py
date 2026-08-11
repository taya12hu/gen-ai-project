"""
Tests for Storage & Indexing.

Runs against the live Supabase Postgres database configured in backend/.env.
Verifies the schema (table, columns, indexes, constraints) and that the
cleaned dataset is queryable and correctly indexed.
"""

import pandas as pd
import pytest

from app.settings import DATA_DIR
from app.storage.db import get_connection

CLEAN_CSV = DATA_DIR / "processed" / "restaurants_clean.csv"


@pytest.fixture(scope="module")
def conn():
    connection = get_connection()
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def cur(conn):
    with conn.cursor() as cursor:
        yield cursor


def test_connection_succeeds(conn):
    assert conn.closed == 0


def test_restaurants_table_exists(cur):
    cur.execute(
        "select 1 from information_schema.tables "
        "where table_name = 'restaurants';"
    )
    assert cur.fetchone() is not None


def test_restaurants_columns_and_types(cur):
    cur.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_name = 'restaurants';"
    )
    columns = dict(cur.fetchall())
    assert columns["name"] == "text"
    assert columns["place"] == "text"
    assert columns["city"] == "text"
    assert columns["cuisines"] == "ARRAY"
    assert columns["price"] == "numeric"
    assert columns["rating"] == "numeric"
    assert columns["votes"] == "integer"


def test_expected_indexes_exist(cur):
    cur.execute("select indexname from pg_indexes where tablename = 'restaurants';")
    index_names = {row[0] for row in cur.fetchall()}
    assert "restaurants_place_idx" in index_names
    assert "restaurants_price_idx" in index_names
    assert "restaurants_rating_idx" in index_names
    assert "restaurants_cuisines_idx" in index_names


def test_cuisines_index_is_gin(cur):
    cur.execute(
        "select am.amname from pg_class c "
        "join pg_am am on c.relam = am.oid "
        "where c.relname = 'restaurants_cuisines_idx';"
    )
    assert cur.fetchone()[0] == "gin"


def test_unique_constraint_on_name_place(cur):
    cur.execute(
        "select 1 from pg_constraint where conname = 'restaurants_name_place_key';"
    )
    assert cur.fetchone() is not None


def test_row_count_matches_processed_csv(cur):
    expected = len(pd.read_csv(CLEAN_CSV))
    cur.execute("select count(*) from restaurants;")
    assert cur.fetchone()[0] == expected


def test_no_null_values_in_critical_columns(cur):
    cur.execute(
        "select count(*) from restaurants "
        "where name is null or place is null or cuisines is null "
        "or price is null or rating is null;"
    )
    assert cur.fetchone()[0] == 0


def test_cuisines_containment_filter_returns_matches(cur):
    cur.execute("select count(*) from restaurants where cuisines @> array['Chinese'];")
    count = cur.fetchone()[0]
    assert count > 0


def test_cuisines_containment_uses_gin_index(cur):
    cur.execute("explain select * from restaurants where cuisines @> array['Chinese'];")
    plan = "\n".join(row[0] for row in cur.fetchall())
    assert "restaurants_cuisines_idx" in plan


def test_place_filter_uses_btree_index(cur):
    cur.execute("explain select * from restaurants where place = 'Indiranagar';")
    plan = "\n".join(row[0] for row in cur.fetchall())
    assert "restaurants_place_idx" in plan


def test_combined_price_rating_filter(cur):
    cur.execute(
        "select price, rating from restaurants where price <= 800 and rating >= 4.0;"
    )
    rows = cur.fetchall()
    assert len(rows) > 0
    for price, rating in rows:
        assert price <= 800
        assert rating >= 4.0
