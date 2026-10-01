"""Phase 1 acceptance: chunk the real sample corpus deterministically.

Validates the Phase 1 acceptance criteria without any network call:
ingesting ``data/raw/ferry_docs`` yields N>0 chunks, each carrying complete
metadata (source_file, strategy, chunk_id; section_heading for markdown), and the
result is deterministic across runs.
"""
from __future__ import annotations

import shutil

import pytest

from rag.config import ConfigError, Settings
from rag.indexing import index_path
from rag.indexing.vector_store import Chunking, CollectionInfo
from rag.ingestion import build_chunks_for_dir
from rag.ingestion.chunkers import Chunk


@pytest.fixture
def settings() -> Settings:
    # Default config; corpus_dir resolves to <repo>/data/raw/ferry_docs.
    return Settings(_env_file=None)


def test_corpus_present(settings: Settings) -> None:
    assert settings.corpus_dir.is_dir(), f"missing corpus at {settings.corpus_dir}"
    assert list(settings.corpus_dir.glob("*.md")), "expected markdown docs in the corpus"


def test_corpus_chunks_have_complete_metadata(settings: Settings) -> None:
    chunks = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    assert len(chunks) > 0

    sources = {c.source_file for c in chunks}
    assert len(sources) >= 6  # the ferry corpus has several markdown docs

    for c in chunks:
        assert c.source_file
        assert c.strategy == "fixed"
        assert len(c.chunk_id) == 16
        assert isinstance(c.ordinal, int)
        assert c.text.strip()
        # Markdown chunks must carry their section heading.
        assert c.section_heading, f"missing heading in {c.source_file} ordinal {c.ordinal}"
        # Metadata never holds None ("" for no heading, -1 for no page).
        assert None not in c.metadata().values()


def test_corpus_chunking_is_deterministic(settings: Settings) -> None:
    first = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    second = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


# --- index_path: one chunk store, read by dense and sparse (fakes; no database) --
class HashEmbedder:
    """Deterministic per-text vectors; distinct texts are dissimilar. Counts its work."""

    model = "hash-16"
    total_cost_usd = 0.0

    def __init__(self) -> None:
        self.embedded = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        self.embedded += len(texts)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i] / 255.0 for i in range(16)])
        return out


class MemoryStore:
    """Minimal in-memory stand-in for VectorStore; logs the calls it gets."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], Chunk]] = {}
        self.events: list[str] = []
        self.registered: tuple | None = None
        self.closed = False

    def collection_info(self) -> CollectionInfo | None:
        self.events.append("collection_info")
        if self.registered is None:
            return None
        return CollectionInfo("memory", *self.registered)

    def ensure_collection(self, embedding_model, dim, chunking) -> None:
        self.events.append("ensure_collection")
        if self.registered not in (None, (embedding_model, dim, chunking)):
            raise ConfigError(f"the collection holds {self.registered}")
        self.registered = (embedding_model, dim, chunking)

    def existing_ids(self, ids) -> set[str]:
        return {i for i in ids if i in self._rows}

    def add(self, chunks, embeddings) -> int:
        self.events.append("add")
        for c, v in zip(chunks, embeddings, strict=True):
            self._rows[c.chunk_id] = (v, c)
        return len(chunks)

    def count(self) -> int:
        self.events.append("count")
        return len(self._rows)

    def all_chunks(self) -> list[Chunk]:
        return [chunk for _, chunk in self._rows.values()]

    def close(self) -> None:
        self.closed = True

    def query(self, query_embedding, top_k):
        self.events.append("query")
        import math

        from rag.indexing.vector_store import ScoredChunk

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = [
            ScoredChunk(cid, c.text, cos(query_embedding, vec), c.metadata())
            for cid, (vec, c) in self._rows.items()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def test_index_path_registers_the_collection_before_comparing_chunks(settings: Settings) -> None:
    store = MemoryStore()
    index_path(settings.corpus_dir, settings=settings, embedder=HashEmbedder(), store=store)

    # The embedder's model and the vectors' dimension, before dedup reads the store.
    assert store.registered == ("hash-16", 16, Chunking("fixed", 800, 120))
    assert store.events.index("ensure_collection") < store.events.index("count")
    assert not store.closed  # a store passed in belongs to the caller


def test_index_path_closes_a_store_it_opened(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag.indexing.vector_store import VectorStore

    opened = MemoryStore()
    monkeypatch.setattr(VectorStore, "from_settings", classmethod(lambda cls, s=None: opened))

    index_path(settings.corpus_dir, settings=settings, embedder=HashEmbedder())

    assert opened.closed  # its connection pool doesn't outlive the call


def test_a_second_seed_embeds_nothing(settings: Settings) -> None:
    store = MemoryStore()
    first = index_path(settings.corpus_dir, settings=settings, embedder=HashEmbedder(), store=store)
    embedder = HashEmbedder()

    again = index_path(settings.corpus_dir, settings=settings, embedder=embedder, store=store)

    assert first.chunks_indexed > 0
    assert embedder.embedded == 0  # an unchanged corpus costs nothing to re-seed
    assert (again.chunks_indexed, again.chunks_already_stored) == (0, first.chunks_indexed)
    assert again.total_chunks_in_store == first.total_chunks_in_store


def test_a_changed_file_embeds_only_its_new_chunks(settings: Settings, tmp_path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(settings.corpus_dir, corpus)
    store = MemoryStore()
    first = index_path(corpus, settings=settings, embedder=HashEmbedder(), store=store)
    page = corpus / "05-rate-limits.md"
    page.write_text(page.read_text(encoding="utf-8") + "\n\nOne more paragraph.\n", "utf-8")
    embedder = HashEmbedder()

    again = index_path(corpus, settings=settings, embedder=embedder, store=store)

    assert 0 < embedder.embedded == again.chunks_indexed < first.chunks_indexed


def test_another_embedding_model_fails_before_anything_is_embedded(settings: Settings) -> None:
    store = MemoryStore()
    index_path(settings.corpus_dir, settings=settings, embedder=HashEmbedder(), store=store)
    other = HashEmbedder()
    other.model = "other-model"

    with pytest.raises(ConfigError):
        index_path(settings.corpus_dir, settings=settings, embedder=other, store=store)
    assert other.embedded == 0


def test_dense_and_sparse_read_the_same_stored_chunks(settings: Settings) -> None:
    pytest.importorskip("rank_bm25")
    from rag.retrieval.sparse import SparseRetriever

    store = MemoryStore()
    summary = index_path(
        settings.corpus_dir, settings=settings, embedder=HashEmbedder(), store=store
    )
    sparse = SparseRetriever.from_settings(settings, store=store)

    assert sparse.index.count() == store.count() == summary.total_chunks_in_store
    hits = sparse.retrieve("FERRY-429", top_k=3)
    assert {h.chunk_id for h in hits} <= {c.chunk_id for c in store.all_chunks()}
    assert "FERRY-429" in hits[0].text
    assert "dedup" in summary.timings_ms and "store" in summary.timings_ms


def test_an_empty_collection_cannot_build_sparse_retrieval(settings: Settings) -> None:
    pytest.importorskip("rank_bm25")
    from rag.retrieval.sparse import SparseRetriever

    with pytest.raises(FileNotFoundError, match="seed first"):
        SparseRetriever.from_settings(settings, store=MemoryStore())


def test_index_path_dedups_repeated_content(settings: Settings, tmp_path) -> None:
    # A corpus with a duplicated file: same text, different filename.
    src = (settings.corpus_dir / "01-overview.md").read_text(encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "original.md").write_text(src, encoding="utf-8")
    (corpus / "copy.md").write_text(src, encoding="utf-8")

    summary = index_path(corpus, settings=settings, embedder=HashEmbedder(), store=MemoryStore())

    # The copied file's chunks embed identically -> skipped as duplicates.
    assert summary.chunks_skipped_duplicates > 0
    assert summary.total_chunks_in_store == summary.chunks_indexed
    assert summary.chunks_indexed + summary.chunks_skipped_duplicates == (
        summary.chunks_indexed * 2
    )
