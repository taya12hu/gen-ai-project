"""
Phase 9 - Authentication & Authorization.

Applies schema.sql (users table) to the configured Postgres database.
Safe to re-run: uses `if not exists`.
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR.parent / "phase3_storage_indexing"))

from db import get_connection  # noqa: E402

SCHEMA_SQL = PHASE_DIR / "schema.sql"


def main() -> None:
    sql = SCHEMA_SQL.read_text()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema applied: users table ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
