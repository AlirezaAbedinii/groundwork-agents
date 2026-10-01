"""Indexing: embeddings, the chunk store (Postgres + pgvector), BM25 index.

:func:`index_path` is the shared "make this path searchable" operation — load,
normalize, chunk, embed what's new, **dedup**, and upsert into the collection —
used by ``scripts/ingest.py``, ``scripts/seed.py``, and ``POST /v1/ingest``.
The collection is the only store: sparse retrieval builds its BM25 index from
the stored chunks. The embedder/store are injectable so it is testable without
providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..observability.metrics import Stopwatch


@dataclass
class IndexSummary:
    """What an :func:`index_path` run did, plus its cost/latency."""

    files: int
    chunks_indexed: int
    total_chunks_in_store: int
    embedding_cost_usd: float
    chunks_already_stored: int = 0
    chunks_skipped_duplicates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


def index_path(
    path: str | Path,
    *,
    settings: Settings | None = None,
    embedder=None,
    store=None,
    persist_processed: bool = False,
) -> IndexSummary:
    """Load, chunk, embed, dedup, and store every supported file under ``path``.

    Only chunks the collection doesn't hold yet are embedded (chunk ids hash
    their text and provenance), so re-seeding an unchanged corpus costs nothing.
    Near-duplicates (cosine > ``settings.dedup_cosine_threshold`` vs stored or
    earlier-in-batch content) are skipped; as they're never stored, a re-seed
    embeds them again. An existing collection must match this run's embedding
    model and chunking; a new one is registered before anything is compared.
    """
    settings = settings or get_settings()
    if embedder is None:
        from .embeddings import get_embedding_client

        embedder = get_embedding_client(settings)
    if store is not None:
        return _index(Path(path), settings, embedder, store, persist_processed)

    from .vector_store import VectorStore

    store = VectorStore.from_settings(settings)
    try:
        return _index(Path(path), settings, embedder, store, persist_processed)
    finally:
        store.close()  # a store opened here holds a connection pool


def _index(
    path: Path, settings: Settings, embedder, store, persist_processed: bool
) -> IndexSummary:
    from ..ingestion import build_chunks_for_dir, build_chunks_for_file
    from ..ingestion.dedup import filter_duplicates
    from .vector_store import Chunking

    sw = Stopwatch()
    with sw.time("load_chunk"):
        if path.is_dir():
            chunks = build_chunks_for_dir(path, settings=settings, persist=persist_processed)
        else:
            chunks = build_chunks_for_file(path, settings=settings, persist=persist_processed)

    chunking = Chunking.from_settings(settings)
    info = store.collection_info()
    if info is None:
        stored_ids: set[str] = set()
    else:
        # Raises ConfigError if this run's model or chunking differs from the collection's.
        store.ensure_collection(embedder.model, info.dim, chunking)
        stored_ids = store.existing_ids(c.chunk_id for c in chunks)
    new = [c for c in chunks if c.chunk_id not in stored_ids]

    cost_before = getattr(embedder, "total_cost_usd", 0.0)
    with sw.time("embed"):
        vectors = embedder.embed_texts([c.text for c in new]) if new else []

    if vectors and info is None:
        store.ensure_collection(embedder.model, len(vectors[0]), chunking)

    with sw.time("dedup"):
        kept, kept_vectors, skipped = filter_duplicates(
            new, vectors, store, settings.dedup_cosine_threshold
        )

    with sw.time("store"):
        stored = store.add(kept, kept_vectors)

    return IndexSummary(
        files=len({c.source_file for c in chunks}),
        chunks_indexed=stored,
        total_chunks_in_store=store.count(),
        embedding_cost_usd=round(getattr(embedder, "total_cost_usd", 0.0) - cost_before, 6),
        chunks_already_stored=len(chunks) - len(new),
        chunks_skipped_duplicates=len(skipped),
        timings_ms=sw.as_dict(),
    )
