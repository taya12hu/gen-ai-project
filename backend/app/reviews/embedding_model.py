"""
Review Ingestion & Embedding.

Shared sentence-transformers loader. Used by embed.py (ingestion time) and
the hybrid retrieval engine (query time) - both must embed with the exact
same model/dimension for cosine similarity to mean anything, so that
constant lives in exactly one place.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)
