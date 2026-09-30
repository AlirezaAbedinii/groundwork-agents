"""The pgvector chunk store against a real Postgres (marker ``pg``).

Every test works in fresh collections from the ``pg_store`` fixtures, dropped
afterwards. The vectors are tiny and hand-made, so each score can be checked by
hand.
"""
from __future__ import annotations

import math

import pytest

from rag.config import ConfigError
from rag.indexing.vector_store import Chunking, CollectionInfo, VectorStore, table_name
from rag.ingestion.chunkers import Chunk

pytestmark = pytest.mark.pg

MODEL = "fake-embedder"
FIXED = Chunking("fixed", 800, 120)


def _chunk(
    chunk_id: str,
    *,
    source: str = "a.md",
    ordinal: int = 0,
    text: str | None = None,
    heading: str | None = "Intro",
    page: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text or f"text of {chunk_id}",
        source_file=source,
        strategy="fixed",
        ordinal=ordinal,
        section_heading=heading,
        page=page,
    )


def _seeded(store: VectorStore, rows: list[tuple[Chunk, list[float]]], dim: int = 3):
    store.ensure_collection(MODEL, dim, FIXED)
    store.add([chunk for chunk, _ in rows], [vector for _, vector in rows])
    return store


def test_add_upserts_by_chunk_id(pg_store) -> None:
    _seeded(pg_store, [(_chunk("c1"), [1.0, 0.0, 0.0]), (_chunk("c2"), [0.0, 1.0, 0.0])])

    pg_store.add([_chunk("c1", text="rewritten")], [[1.0, 0.0, 0.0]])

    assert pg_store.count() == 2
    assert [c.text for c in pg_store.chunks_for_source("a.md")] == ["rewritten", "text of c2"]


def test_query_ranks_and_scores_by_cosine_similarity(pg_store) -> None:
    # Against q = (1, 0, 0) the cosine similarity of v is v_x / |v|:
    #   a (1, 0, 0) -> 1.0     b (1, 1, 0) -> 1/sqrt(2) = 0.7071
    #   c (0, 1, 0) -> 0.0     d (-1, 0, 0) -> -1.0
    _seeded(
        pg_store,
        [
            (_chunk("c"), [0.0, 1.0, 0.0]),
            (_chunk("a"), [1.0, 0.0, 0.0]),
            (_chunk("d"), [-1.0, 0.0, 0.0]),
            (_chunk("b"), [1.0, 1.0, 0.0]),
        ],
    )

    hits = pg_store.query([1.0, 0.0, 0.0], top_k=4)

    assert [h.chunk_id for h in hits] == ["a", "b", "c", "d"]
    assert [h.score for h in hits] == pytest.approx([1.0, 1 / math.sqrt(2), 0.0, -1.0], abs=1e-6)
    assert [h.chunk_id for h in pg_store.query([1.0, 0.0, 0.0], top_k=2)] == ["a", "b"]


def test_count_and_list_sources(pg_store) -> None:
    _seeded(
        pg_store,
        [
            (_chunk("b0", source="b.md"), [1.0, 0.0, 0.0]),
            (_chunk("a0", source="a.md"), [0.0, 1.0, 0.0]),
            (_chunk("a1", source="a.md", ordinal=1), [0.0, 0.0, 1.0]),
        ],
    )

    assert pg_store.count() == 3
    assert list(pg_store.list_sources().items()) == [("a.md", 2), ("b.md", 1)]


def test_metadata_round_trip_keeps_the_index_conventions(pg_store) -> None:
    bare = _chunk("bare", heading=None, page=None)
    paged = _chunk("paged", source="guide.pdf", ordinal=4, heading="Setup", page=3)
    _seeded(pg_store, [(bare, [1.0, 0.0, 0.0]), (paged, [0.0, 1.0, 0.0])])

    hits = {h.chunk_id: h for h in pg_store.query([1.0, 1.0, 0.0], top_k=2)}

    # Hits carry Chunk.metadata(): "" for no heading and -1 for no page, never None.
    assert hits["bare"].metadata == bare.metadata()
    assert (hits["bare"].metadata["section_heading"], hits["bare"].metadata["page"]) == ("", -1)
    assert hits["paged"].metadata == paged.metadata()
    # The stored rows keep None itself.
    assert set(pg_store.all_chunks()) == {bare, paged}


def test_all_chunks_and_chunks_for_source_are_in_document_order(pg_store) -> None:
    _seeded(
        pg_store,
        [
            (_chunk("b0", source="b.md"), [1.0, 0.0, 0.0]),
            (_chunk("a1", source="a.md", ordinal=1), [0.0, 1.0, 0.0]),
            (_chunk("a0", source="a.md"), [0.0, 0.0, 1.0]),
            (_chunk("B0", source="B.md"), [1.0, 1.0, 0.0]),
        ],
    )

    # Sources sort bytewise, as Python sorts them ("B.md" < "a.md"), then by ordinal.
    assert [c.chunk_id for c in pg_store.all_chunks()] == ["B0", "a0", "a1", "b0"]
    assert [c.chunk_id for c in pg_store.chunks_for_source("a.md")] == ["a0", "a1"]
    assert pg_store.chunks_for_source("missing.md") == []


def test_existing_ids(pg_store) -> None:
    _seeded(pg_store, [(_chunk("c1"), [1.0, 0.0, 0.0]), (_chunk("c2"), [0.0, 1.0, 0.0])])

    assert pg_store.existing_ids(["c1", "nope", "c2"]) == {"c1", "c2"}
    assert pg_store.existing_ids([]) == set()


def test_an_unseeded_collection_lists_empty_but_is_not_served(pg_store) -> None:
    assert pg_store.collection_info() is None
    assert pg_store.count() == 0
    assert pg_store.list_sources() == {}
    with pytest.raises(FileNotFoundError, match="seed first"):
        pg_store.query([1.0, 0.0, 0.0])
    with pytest.raises(FileNotFoundError, match="seed first"):
        pg_store.all_chunks()


def test_a_collection_embedded_by_another_model_is_refused(pg_store_factory) -> None:
    built = _seeded(pg_store_factory(), [(_chunk("c1"), [1.0, 0.0, 0.0])])
    serving = pg_store_factory(name=built.collection, expected_model="other-model")

    with pytest.raises(ConfigError) as exc:
        serving.query([1.0, 0.0, 0.0])
    assert MODEL in str(exc.value) and "other-model" in str(exc.value)

    # The registry won't mix in vectors from another model, or chunks cut another way.
    with pytest.raises(ConfigError, match="other-model"):
        serving.ensure_collection("other-model", 3, FIXED)
    with pytest.raises(ConfigError, match="recursive"):
        built.ensure_collection(MODEL, 3, Chunking("recursive", 800, 120))


@pytest.mark.parametrize(
    "name", ["Ferry", "1docs", "docs-v2", 'x"; DROP TABLE y; --', "a" * 45, ""]
)
def test_an_invalid_collection_name_is_rejected_before_any_sql(name: str) -> None:
    # Nothing listens at this URL: a name that got as far as SQL would fail to connect.
    with pytest.raises(ValueError, match="Invalid collection name"):
        VectorStore("postgresql://nobody@127.0.0.1:1/nowhere", name)


def test_the_longest_name_still_fits_postgres_identifiers() -> None:
    # Postgres truncates identifiers past 63 bytes; the longest derived one is the
    # "_source" index, which must not collapse onto the table's own name.
    longest = table_name("a" * 44)
    assert len(f"{longest}_source") == 63
    assert len(f"{longest}_hnsw") < 63


def test_collections_with_different_dimensions_coexist(pg_store_factory) -> None:
    small = _seeded(pg_store_factory(), [(_chunk("s"), [1.0, 0.0, 0.0])], dim=3)
    large = _seeded(pg_store_factory(), [(_chunk("l"), [0.0, 0.0, 0.0, 0.0, 1.0])], dim=5)

    assert (small.collection_info().dim, large.collection_info().dim) == (3, 5)
    assert small.query([1.0, 0.0, 0.0])[0].chunk_id == "s"
    assert large.query([0.0, 0.0, 0.0, 0.0, 1.0])[0].chunk_id == "l"
    with pytest.raises(ValueError, match="3-dim"):
        small.add([_chunk("x")], [[1.0, 0.0, 0.0, 0.0, 0.0]])


def test_ensure_collection_registers_once_and_builds_the_indexes(pg_store, pg_url) -> None:
    import psycopg

    info = pg_store.ensure_collection(MODEL, 3, FIXED)

    assert info == CollectionInfo(pg_store.collection, MODEL, 3, FIXED)
    assert pg_store.ensure_collection(MODEL, 3, FIXED) == info
    assert pg_store.collection_info() == info
    table = table_name(pg_store.collection)
    with psycopg.connect(pg_url) as conn:
        indexes = dict(
            conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s", (table,)
            ).fetchall()
        )
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes[f"{table}_hnsw"]
    assert "(source_file, ordinal)" in indexes[f"{table}_source"]


def test_the_query_can_use_the_hnsw_index(pg_store, pg_url) -> None:
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from psycopg import sql

    from rag.indexing.vector_store import _query_sql

    _seeded(pg_store, [(_chunk(f"c{i}"), [1.0, float(i), 0.0]) for i in range(4)])
    table = table_name(pg_store.collection)

    with psycopg.connect(pg_url) as conn:
        register_vector(conn)
        conn.execute("SET enable_seqscan = off")
        plan = conn.execute(
            sql.SQL("EXPLAIN ") + _query_sql(table), {"q": Vector([1.0, 0.0, 0.0]), "k": 3}
        ).fetchall()

    assert f"Index Scan using {table}_hnsw" in "\n".join(row[0] for row in plan)
