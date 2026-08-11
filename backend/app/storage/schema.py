"""
Storage & Indexing.

Applies schema.sql (table + indexes) to the configured Postgres database.
Safe to re-run: uses `if not exists` throughout.
"""

from pathlib import Path

from app.storage.db import get_connection

SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"


def main() -> None:
    sql = SCHEMA_SQL.read_text()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema applied: restaurants table + indexes ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
