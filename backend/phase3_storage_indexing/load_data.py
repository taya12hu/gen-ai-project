"""
Phase 3 - Storage & Indexing.

Loads the cleaned dataset from Phase 2 (phase2_data_cleaning/data/processed/
restaurants_clean.csv) into the `restaurants` table. Idempotent: truncates
the table first, so re-running produces a consistent state rather than
duplicate/appended rows.
"""

from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values

from db import get_connection

PHASE_DIR = Path(__file__).resolve().parent
CLEAN_CSV = (
    PHASE_DIR.parent / "phase2_data_cleaning" / "data" / "processed" / "restaurants_clean.csv"
)

BATCH_SIZE = 1000


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)

    rows = [
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

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("truncate table restaurants restart identity;")
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                execute_values(
                    cur,
                    """
                    insert into restaurants
                        (name, place, city, cuisines, price, rating, rest_type, votes)
                    values %s
                    """,
                    batch,
                )
        conn.commit()
        print(f"Loaded {len(rows)} rows into restaurants.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
