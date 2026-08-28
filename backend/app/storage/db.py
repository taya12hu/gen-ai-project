"""
Storage & Indexing.

Shared DB connection helper. Reads connection info from backend/.env
(PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) so credentials never
live in code.

get_connection() is backed by a small thread-safe connection pool rather
than opening a fresh TCP+TLS+Postgres-auth handshake on every call - every
caller across the codebase does its own get_connection()/close() per query,
which used to pay that full setup cost each time. ThreadedConnectionPool
(not the plain, single-threaded pool) because FastAPI runs sync route
handlers across a worker thread pool, so concurrent requests really do call
get_connection() from different OS threads.

_PoolConnection is a real psycopg2 connection subclass (via
connection_factory), not a wrapper/proxy object - an earlier version of
this wrapped the connection in a plain Python class, which broke anything
that does isinstance(conn, psycopg2.extensions.connection), e.g. pgvector's
register_vector(). Subclassing keeps every caller's `conn.close()` working
completely unchanged: it's overridden here to return the connection to the
pool (after rolling back anything left uncommitted, so a caller that raised
before committing never hands the next borrower a connection stuck
mid-transaction) instead of tearing down the socket - and the object is
still a genuine psycopg2 connection everywhere else, so nothing else in the
codebase (or any library it calls into) had to change for this.
"""

import logging
import os
import threading

from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as _Psycopg2Connection
from psycopg2.pool import PoolError, ThreadedConnectionPool

logger = logging.getLogger(__name__)

MIN_POOL_SIZE = 1

# Must be read together with the server's request concurrency, not chosen in
# isolation. FastAPI runs sync route handlers (which all of these are) on
# AnyIO's worker thread pool, which defaults to 40 threads - so up to 40
# requests can be inside get_connection() at once. With a pool of 10, the
# 11th concurrent request didn't queue, it raised PoolError and failed the
# request outright.
#
# The two numbers are now set from one place: DB_MAX_CONNECTIONS caps the
# pool, and app.api.main sizes the request threadpool to match it so the
# server never admits more concurrent DB-touching work than it has
# connections for. The default stays conservative because managed Postgres
# tiers (Supabase's pooler in particular) cap total connections per project,
# and a handful of app instances each holding a large pool exhausts that
# ceiling long before the app itself is saturated.
MAX_POOL_SIZE = int(os.environ.get("DB_MAX_CONNECTIONS", "16"))
CONNECT_TIMEOUT_SECONDS = 10


class _PoolConnection(_Psycopg2Connection):
    def close(self):
        if getattr(self, "_pc_returning", False):
            # Re-entrant call: psycopg2.pool's own _putconn() calls conn.close()
            # directly (not through our override) when it decides to discard
            # this connection instead of recycling it - e.g. the idle pool is
            # already at minconn, or the connection's transaction status came
            # back UNKNOWN (server dropped it). That happens *while we're
            # still inside* the outer close() call below, on the same thread,
            # holding ThreadedConnectionPool's non-reentrant lock. Recursing
            # into _pool.putconn() here would try to re-acquire that lock and
            # deadlock forever (this was silent - low CPU, no error - because
            # threading.Lock.acquire() just blocks). Honor the pool's verdict
            # with a real close instead of fighting it.
            _Psycopg2Connection.close(self)
            return
        self._pc_returning = True
        try:
            try:
                self.rollback()
            except Exception:
                pass
            _pool.putconn(self)
        finally:
            self._pc_returning = False


# Built lazily (on first get_connection(), not on import) so a slow or
# unreachable database can't block app startup - psycopg2.connect() has no
# default timeout, and this module is imported transitively by
# app.api.main, which uvicorn loads before it binds to $PORT. Eagerly
# constructing the pool here used to make the whole process hang mid-import
# whenever the database was slow to respond, so uvicorn never got the chance
# to open a port at all.
_pool: ThreadedConnectionPool | None = None
_pool_init_lock = threading.Lock()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_init_lock:
            if _pool is None:
                _pool = ThreadedConnectionPool(
                    MIN_POOL_SIZE,
                    MAX_POOL_SIZE,
                    host=os.environ["PGHOST"],
                    port=os.environ["PGPORT"],
                    dbname=os.environ["PGDATABASE"],
                    user=os.environ["PGUSER"],
                    password=os.environ["PGPASSWORD"],
                    connect_timeout=CONNECT_TIMEOUT_SECONDS,
                    connection_factory=_PoolConnection,
                )
    return _pool


def get_connection():
    try:
        return _get_pool().getconn()
    except PoolError:
        logger.warning("Connection pool exhausted (max=%d) while acquiring a connection", MAX_POOL_SIZE)
        raise


def ensure_vector_registered(conn) -> None:
    """Registers pgvector's type adapters on `conn`, at most once per
    underlying connection.

    register_vector() looks the `vector` type's OID up in pg_type, which is a
    full round trip - ~150ms against a remote database, and it was being paid
    on every single retrieval call even though the adapters are
    connection-scoped and survive both transactions and the rollback the pool
    performs on release.

    The flag lives on the connection object rather than in a module-level set:
    when the pool discards a connection and dials a new one, that new object
    starts without the flag and re-registers, which is exactly right. Keying a
    cache on id() or DSN would have to guess at that lifecycle.
    """
    if getattr(conn, "_pc_vector_registered", False):
        return
    register_vector(conn)
    conn._pc_vector_registered = True
