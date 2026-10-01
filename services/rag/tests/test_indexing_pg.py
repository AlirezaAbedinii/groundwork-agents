"""index_path and sparse retrieval against the real chunk store (marker ``pg``).

The embedder is a deterministic hash (no model, no network); everything else —
the registry, the upserts, dedup's nearest-neighbour queries and the BM25 index
built from the stored rows — runs on Postgres with pgvector.
"""
from __future__ import annotations

import hashlib

import pytest

from rag.config import Settings
from rag.indexing import index_path
from rag.indexing.vector_store import Chunking
from rag.retrieval.sparse import SparseRetriever

pytestmark = pytest.mark.pg


class HashEmbedder:
    model = "hash-16"
    total_cost_usd = 0.0

    def __init__(self) -> None:
        self.embedded = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_reseeding_embeds_only_chunks_the_collection_lacks(pg_store, settings) -> None:
    first = HashEmbedder()
    seeded = index_path(settings.corpus_dir, settings=settings, embedder=first, store=pg_store)
    again = HashEmbedder()
    reseeded = index_path(settings.corpus_dir, settings=settings, embedder=again, store=pg_store)

    assert first.embedded == seeded.chunks_indexed == pg_store.count() > 0
    assert again.embedded == 0
    assert (reseeded.chunks_indexed, reseeded.chunks_already_stored) == (0, seeded.chunks_indexed)
    assert reseeded.total_chunks_in_store == seeded.total_chunks_in_store


def test_the_sparse_side_ranks_the_stored_chunks(pg_store, settings) -> None:
    pytest.importorskip("rank_bm25")
    index_path(settings.corpus_dir, settings=settings, embedder=HashEmbedder(), store=pg_store)

    sparse = SparseRetriever.from_settings(settings, store=pg_store)

    assert sparse.index.count() == pg_store.count()
    top = sparse.retrieve("FERRY-429", top_k=1)[0]
    stored = {c.chunk_id: c for c in pg_store.all_chunks()}
    assert "FERRY-429" in top.text
    assert top.metadata == stored[top.chunk_id].metadata()


def test_sparse_retrieval_needs_a_seeded_collection(pg_store, settings) -> None:
    pytest.importorskip("rank_bm25")
    with pytest.raises(FileNotFoundError, match="seed first"):  # never registered
        SparseRetriever.from_settings(settings, store=pg_store)

    pg_store.ensure_collection("hash-16", 16, Chunking.from_settings(settings))

    with pytest.raises(FileNotFoundError, match="holds no chunks"):  # registered, still empty
        SparseRetriever.from_settings(settings, store=pg_store)
