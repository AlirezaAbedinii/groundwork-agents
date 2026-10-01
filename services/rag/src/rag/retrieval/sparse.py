"""Sparse retrieval: BM25 top-k over the collection's stored chunks.

The BM25 index is built in memory from the chunk store when the retriever is
built (see :meth:`SparseRetriever.from_settings`), so it ranks exactly the
chunks dense retrieval searches; chunks stored later are picked up when the
retriever is rebuilt. Results are :class:`ScoredChunk` — the same shape dense
retrieval produces, so fusion treats both sources uniformly. BM25 scores are
raw/unbounded; downstream RRF fuses by **rank**, so no normalization is needed
here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from ..observability.metrics import Stopwatch


class SupportsBM25Query(Protocol):
    """The slice of BM25Index the retriever needs (injectable in tests)."""

    def query(self, text: str, top_k: int) -> list[tuple[str, float, str, dict]]: ...


@dataclass
class SparseRetriever:
    """BM25 keyword retrieval over the collection's chunks."""

    index: SupportsBM25Query
    default_top_k: int = 10

    @classmethod
    def from_settings(cls, settings: Settings | None = None, *, store=None) -> SparseRetriever:
        """Build BM25 over the stored chunks; an empty collection names the seed command.

        ``store`` is the dense side's chunk store, which stays open; without one,
        a store is opened just for this read and closed again.
        """
        from ..indexing.bm25_index import BM25Index

        settings = settings or get_settings()
        if store is not None:
            chunks = store.all_chunks()
        else:
            from ..indexing.vector_store import VectorStore

            own = VectorStore.from_settings(settings)
            try:
                chunks = own.all_chunks()
            finally:
                own.close()
        if not chunks:
            collection = getattr(store, "collection", settings.collection)
            raise FileNotFoundError(
                f"Collection {collection!r} holds no chunks; seed first: "
                "python scripts/seed.py (or docker compose run --rm seed)."
            )
        return cls(index=BM25Index.from_chunks(chunks), default_top_k=settings.top_k)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        stopwatch: Stopwatch | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.default_top_k
        sw = stopwatch or Stopwatch()
        with sw.time("sparse"):
            hits = self.index.query(query, top_k=k)
        return [
            ScoredChunk(chunk_id=cid, text=text, score=score, metadata=metadata)
            for cid, score, text, metadata in hits
        ]
