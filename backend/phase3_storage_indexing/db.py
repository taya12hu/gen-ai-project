"""
Phase 3 - Storage & Indexing.

Shared DB connection helper. Reads connection info from backend/.env
(PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) so credentials never
live in code.
"""

from pathlib import Path

import psycopg2
from dotenv import load_dotenv
import os

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def get_connection():
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )
