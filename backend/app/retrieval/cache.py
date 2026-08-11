"""
Hybrid Retrieval Engine - query result cache.

A small in-process TTL cache for retrieval results. Repeating the same
question - the same user asking twice, or two different users asking the
same popular thing ("best biryani in Koramangala") - currently re-runs a
DB round trip (and, for a vibe query, a local embed + pgvector search)
every single time even though the underlying data hasn't changed in the
meantime. This cache sits in front of the three public retrieval entry
points so an identical (filters, vibe_query) request within the TTL window
is served from memory instead.

Deliberately not caching query understanding or the final LLM reply: those
depend on conversation history, so an apparently-repeated question rarely
has an identical cache key, and caching the LLM reply itself would risk
serving a stale, context-blind answer. Retrieval results depend only on
the query parameters and the (slow-moving) restaurant/review data, which
is what makes them safe and worth caching.
"""

import time
from typing import Any, Hashable

CACHE_MISS = object()


class TTLCache:
    def __init__(self, ttl_seconds: float, max_size: int = 256):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Any:
        entry = self._store.get(key, CACHE_MISS)
        if entry is CACHE_MISS:
            return CACHE_MISS
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return CACHE_MISS
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if len(self._store) >= self._max_size and key not in self._store:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        self._store.clear()
