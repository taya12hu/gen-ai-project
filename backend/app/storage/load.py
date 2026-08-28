"""
Storage & Indexing.

Loads the cleaned dataset (backend/data/processed/restaurants_clean.csv,
produced by app.data.cleaning) into the `restaurants` table.

Idempotent, and re-runnable *in place*. This used to
`truncate table restaurants restart identity`, which had two problems that
only appear on the second run:

- It fails outright once restaurant_reviews exists. Postgres refuses to
  truncate a table referenced by a foreign key unless CASCADE is given - and
  CASCADE here would silently delete every ingested review as well.
- `restart identity` reassigns every id. Assistant messages persist the
  restaurant ids they grounded their reply in (messages.mentioned_restaurant_ids,
  see app.conversation.schema), so renumbering makes every saved conversation
  replay with *different* restaurants than the user was actually shown -
  silently, with no error anywhere.

Instead this upserts on the natural key the dataset already dedupes by,
(name, place), so a restaurant keeps its id across reloads and existing
conversations stay correct. Rows that disappear from the dataset are reported
rather than deleted, since deleting them would cascade into review and
conversation history; removing them is a deliberate decision, not something a
reload should do on its own.
"""

import argparse
import logging

import pandas as pd
from psycopg2.extras import execute_values

from app.settings import DATA_DIR
from app.storage.db import get_connection

logger = logging.getLogger(__name__)

CLEAN_CSV = DATA_DIR / "processed" / "restaurants_clean.csv"

BATCH_SIZE = 1000


def _rows_from_csv() -> list[tuple]:
    df = pd.read_csv(CLEAN_CSV)
    return [
        (
            row["name"],
            row["place"],
            row["city"],
            [c.strip() for c in row["cuisines"].split(",")],
            row["price"],
            row["rating"],
            None if pd.isna(row["rest_type"]) else row["rest_type"],
            int(row["votes"]),
        )
        for _, row in df.iterrows()
    ]


def _stale_rows(conn, loaded_keys: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """(name, place) pairs in the table that the incoming dataset no longer has."""
    with conn.cursor() as cur:
        cur.execute("select name, place from restaurants;")
        existing = {(name, place) for name, place in cur.fetchall()}
    return sorted(existing - loaded_keys)


def main(report_stale: bool = True) -> None:
    rows = _rows_from_csv()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                execute_values(
                    cur,
                    """
                    insert into restaurants
                        (name, place, city, cuisines, price, rating, rest_type, votes)
                    values %s
                    on conflict (name, place) do update set
                        city = excluded.city,
                        cuisines = excluded.cuisines,
                        price = excluded.price,
                        rating = excluded.rating,
                        rest_type = excluded.rest_type,
                        votes = excluded.votes
                    """,
                    batch,
                )
        conn.commit()
        print(f"Upserted {len(rows)} rows into restaurants (ids preserved).")

        if report_stale:
            stale = _stale_rows(conn, {(r[0], r[1]) for r in rows})
            if stale:
                # Not deleted here on purpose: `on delete cascade` would take
                # their reviews with them, and any conversation that
                # referenced them would lose its cards.
                print(
                    f"Note: {len(stale)} existing restaurant(s) are not in this dataset "
                    f"and were left untouched (e.g. {stale[0][0]} in {stale[0][1]})."
                )
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the cleaned restaurant dataset into Postgres.")
    parser.add_argument(
        "--skip-stale-check",
        action="store_true",
        help="Skip reporting rows present in the table but absent from the dataset.",
    )
    args = parser.parse_args()
    main(report_stale=not args.skip_stale_check)
