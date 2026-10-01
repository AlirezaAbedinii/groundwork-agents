"""BM25 sparse index: built in memory from the collection's stored chunks.

The sparse retriever builds it from the chunk store (:meth:`BM25Index.from_chunks`
over ``VectorStore.all_chunks()``), so dense and sparse retrieval rank exactly
the same chunks and Postgres is the only store.

The tokenizer is the design point. Technical docs win on **exact tokens** —
error codes (``FERRY-429``), config keys (``ferry.worker.concurrency``) — so in
addition to plain alphanumeric words it emits compound tokens (words joined by
``.``/``-``/``_``) intact. A query for "FERRY-429" therefore matches the chunk
containing that literal code even when embedding similarity is weak.

Texts and metadata are kept alongside the token corpus so sparse retrieval can
return full scored chunks without another trip to the store. ``rank_bm25`` is a
tiny pure-Python dependency; it is imported lazily all the same.
"""
from __future__ import annotations

import re

from ..ingestion.chunkers import Chunk

# Plain words plus compound technical tokens kept intact.
_WORD_RE = re.compile(r"[a-z0-9]+")
_COMPOUND_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus dotted/hyphenated compounds kept whole."""
    lower = text.lower()
    return _WORD_RE.findall(lower) + _COMPOUND_RE.findall(lower)


class BM25Index:
    """An upsertable, in-memory BM25 index over chunk texts."""

    def __init__(self) -> None:
        # chunk_id -> (text, metadata); insertion order is preserved and
        # deterministic given the same upsert sequence.
        self._entries: dict[str, tuple[str, dict]] = {}
        self._bm25 = None  # rebuilt lazily after mutations
        self._ids: list[str] = []

    # -- building ----------------------------------------------------------
    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> BM25Index:
        """An index over ``chunks``, e.g. every chunk a collection stores."""
        index = cls()
        index.upsert(chunks)
        return index

    def upsert(self, chunks: list[Chunk]) -> int:
        """Insert or replace chunks by ``chunk_id``; return how many."""
        for chunk in chunks:
            self._entries[chunk.chunk_id] = (chunk.text, chunk.metadata())
        self._bm25 = None
        return len(chunks)

    def count(self) -> int:
        return len(self._entries)

    def _ensure_built(self):
        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:  # pragma: no cover - only without the dep
                raise ImportError(
                    "BM25 requires 'rank-bm25'. Install: pip install -e \".[indexing]\""
                ) from exc
            self._ids = list(self._entries)
            corpus = [tokenize(self._entries[cid][0]) for cid in self._ids]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    # -- querying ----------------------------------------------------------
    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float, str, dict]]:
        """Top-k ``(chunk_id, score, text, metadata)`` for a query string.

        Scores are raw BM25 (unbounded, corpus-dependent); ranking is what
        matters downstream — RRF fuses by rank, not score.
        """
        if not self._entries:
            return []
        bm25 = self._ensure_built()
        scores = bm25.get_scores(tokenize(text))
        ranked = sorted(
            zip(self._ids, scores, strict=True), key=lambda pair: (-pair[1], pair[0])
        )[:top_k]
        return [
            (cid, float(score), self._entries[cid][0], self._entries[cid][1])
            for cid, score in ranked
        ]
