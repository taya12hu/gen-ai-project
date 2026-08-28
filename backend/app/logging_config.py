"""
Central logging setup for the whole `app` package.

Configured once, on first import of `app` (see __init__.py), so the FastAPI
process, pytest runs, and standalone scripts (ingest, embed) all end up using
the same configuration without each entry point needing its own setup. Every
module logs via `logging.getLogger(__name__)`, which makes it a child of the
"app" logger configured here and inherits its handlers.

Two things this gets right that the previous version didn't:

**Where logs go.** Everything was written to backend/dump.log on local disk.
On a container platform that filesystem is ephemeral - wiped on every deploy
and restart - so the careful logging this project does was invisible in
exactly the environment where you can't attach a debugger. stdout is the
handler that always exists; the file is opt-in via LOG_FILE for local runs
where tailing a file is genuinely convenient.

**Which turn a line belongs to.** Retrieval, LLM and persistence lines from
concurrent requests interleave in one stream, so "Structured retrieval: 0
candidates" couldn't be tied to the request that produced it. A per-request id
(see app.api.main's middleware) is carried in a ContextVar and stamped onto
every record by _RequestIdFilter, so one turn can be followed end to end with
a grep.
"""

import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

from app.settings import BACKEND_DIR

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

# "-" rather than an empty string so the bracket in the format string never
# collapses to "[]" for work that isn't request-scoped (ingest, embed, tests).
_NO_REQUEST_ID = "-"

_request_id: ContextVar[str] = ContextVar("request_id", default=_NO_REQUEST_ID)

_configured = False


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """Makes %(request_id)s available on every record.

    A filter rather than a custom Formatter, so it applies uniformly to any
    handler attached to the "app" logger - including ones added later.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _resolve_log_file() -> str | None:
    """Explicit LOG_FILE wins; otherwise no file handler at all.

    Defaulting to a file was what made production logging disappear, so the
    default is now 'stdout only' and a file is something you ask for. Set
    LOG_FILE=dump.log locally for the old behaviour.
    """
    configured = os.environ.get("LOG_FILE", "").strip()
    if not configured:
        return None
    path = os.path.expanduser(configured)
    return path if os.path.isabs(path) else str(BACKEND_DIR / path)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger = logging.getLogger("app")
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    formatter = logging.Formatter(LOG_FORMAT)
    request_id_filter = _RequestIdFilter()

    # stdout, not stderr: these are application events, and every container
    # platform treats stdout as the log stream.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    logger.addHandler(console_handler)

    log_file = _resolve_log_file()
    if log_file:
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(request_id_filter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # A read-only or missing directory must not stop the app booting -
            # stdout logging is already attached and is the one that matters.
            logger.warning("Could not open LOG_FILE %s (%s); logging to stdout only", log_file, exc)
