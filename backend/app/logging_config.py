"""
Central logging setup for the whole `app` package.

Configured once, on first import of `app` (see __init__.py), so the FastAPI
process, pytest runs, and standalone scripts (ingest, embed) all end up
writing to the same backend/dump.log without each entry point needing its
own setup. Every module logs via `logging.getLogger(__name__)`, which makes
it a child of the "app" logger configured here and inherits both handlers.
"""

import logging
from logging.handlers import RotatingFileHandler

from app.settings import BACKEND_DIR

LOG_FILE = BACKEND_DIR / "dump.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
