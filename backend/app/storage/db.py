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

from psycopg2.extensions import connection as _Psycopg2Connection
from psycopg2.pool import PoolError, ThreadedConnectionPool

logger = logging.getLogger(__name__)

MIN_POOL_SIZE = 1
MAX_POOL_SIZE = 10


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


_pool = ThreadedConnectionPool(
    MIN_POOL_SIZE,
    MAX_POOL_SIZE,
    host=os.environ["PGHOST"],
    port=os.environ["PGPORT"],
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
    connection_factory=_PoolConnection,
)


def get_connection():
    try:
        return _pool.getconn()
    except PoolError:
        logger.warning("Connection pool exhausted (max=%d) while acquiring a connection", MAX_POOL_SIZE)
        raise
