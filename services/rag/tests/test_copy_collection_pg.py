"""scripts/copy_collection.py against the RAG Postgres (marker ``pg``).

Both ends are throwaway collections in the test database, so a copy runs the real
COPY streams, ensure_collection and the checksums without a second server. The
rows carry NULLs, tabs, newlines, a backslash and non-ASCII text, which the text
format has to escape and the binary one mustn't touch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rag.config import ConfigError
from rag.indexing.vector_store import Chunking, table_name
from rag.ingestion.chunkers import Chunk

pytestmark = pytest.mark.pg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "copy_collection.py"
_spec = importlib.util.spec_from_file_location("copy_collection", SCRIPT)
copy_script = importlib.util.module_from_spec(_spec)
sys.modules["copy_collection"] = copy_script  # dataclasses look their module up there
_spec.loader.exec_module(copy_script)

MODEL = "fake-embedder"
FIXED = Chunking("fixed", 800, 120)


def _chunk(chunk_id: str, text: str, *, source: str = "a.md", ordinal: int = 0, **kw) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, source_file=source, strategy="fixed", ordinal=ordinal, **kw
    )


ROWS = [
    (_chunk("c1", "plain", section_heading="Intro"), [1.0, 0.0, 0.0]),
    (
        _chunk("c2", "tab\there, newline\nhere, a backslash \\ and é ✓", ordinal=1, page=3),
        [0.1, -2.5, 3.25],
    ),
    (_chunk("c3", "third", source="b.pdf", page=1), [1e-7, 123456.78, -0.333333]),
]


def _seed(store, rows=ROWS, model: str = MODEL):
    store.ensure_collection(model, 3, FIXED)
    store.add([chunk for chunk, _ in rows], [vector for _, vector in rows])
    return store


def _rows(url: str, collection: str) -> list[tuple]:
    import psycopg
    from psycopg import sql

    with psycopg.connect(url) as conn:
        return conn.execute(
            sql.SQL(
                "SELECT chunk_id, source_file, section_heading, page, strategy, ordinal, text,"
                " embedding::real[] FROM {} ORDER BY chunk_id"
            ).format(sql.Identifier(table_name(collection)))
        ).fetchall()


def _scalar(url: str, query: str, *params):
    import psycopg

    with psycopg.connect(url) as conn:
        return conn.execute(query, params).fetchone()[0]


@pytest.mark.parametrize("fmt", ["binary", "text"])
def test_copy_yields_identical_rows_registry_and_hnsw_index(pg_url, pg_store_factory, fmt) -> None:
    source = _seed(pg_store_factory())
    target = pg_store_factory()

    result = copy_script.copy_collection(
        pg_url, pg_url, source.collection, target.collection, fmt=fmt, log=lambda _: None
    )

    assert (result.copied, result.present, result.rows) == (3, 0, 3)
    assert _rows(pg_url, target.collection) == _rows(pg_url, source.collection)
    found = target.collection_info()
    assert (found.embedding_model, found.dim, found.chunking) == (MODEL, 3, FIXED)
    indexdef = _scalar(
        pg_url,
        "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
        f"{table_name(target.collection)}_hnsw",
    )
    assert "USING hnsw (embedding vector_cosine_ops)" in indexdef


def test_second_run_copies_nothing(pg_url, pg_store_factory, capsys) -> None:
    source = _seed(pg_store_factory())
    target = pg_store_factory()
    args = ["--from", pg_url, "--to", pg_url, "--collection", source.collection,
            "--target-collection", target.collection]

    assert copy_script.main(args) == 0
    assert copy_script.main(args) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "copied 0, already present 3, checksum ok"
    assert "copied 3, already present 0, checksum ok" in lines


def test_only_the_rows_the_target_lacks_are_copied(pg_url, pg_store_factory) -> None:
    source = _seed(pg_store_factory())
    target = _seed(pg_store_factory(), rows=ROWS[:1])

    result = copy_script.copy_collection(
        pg_url, pg_url, source.collection, target.collection, log=lambda _: None
    )

    assert (result.copied, result.present) == (2, 1)
    assert _rows(pg_url, target.collection) == _rows(pg_url, source.collection)


def test_a_target_built_with_another_model_is_refused(pg_url, pg_store_factory) -> None:
    source = _seed(pg_store_factory())
    target = pg_store_factory()
    target.ensure_collection("other-model", 3, FIXED)

    with pytest.raises(ConfigError, match="other-model"):
        copy_script.copy_collection(
            pg_url, pg_url, source.collection, target.collection, log=lambda _: None
        )
    assert target.count() == 0


def test_a_target_holding_other_chunks_is_refused_before_any_write(
    pg_url, pg_store_factory
) -> None:
    source = _seed(pg_store_factory())
    target = _seed(pg_store_factory(), rows=[(_chunk("zz", "not in the source"), [0.0, 0.0, 1.0])])

    with pytest.raises(copy_script.CopyError, match="1 chunks the source doesn't"):
        copy_script.copy_collection(
            pg_url, pg_url, source.collection, target.collection, log=lambda _: None
        )
    assert target.count() == 1


def test_dry_run_reports_and_writes_nothing(pg_url, pg_store_factory) -> None:
    source = _seed(pg_store_factory())
    target = pg_store_factory()

    result = copy_script.copy_collection(
        pg_url, pg_url, source.collection, target.collection, dry_run=True, log=lambda _: None
    )

    assert result.summary() == "would copy 3, already present 0 (dry run)"
    assert _scalar(pg_url, "SELECT to_regclass(%s)", table_name(target.collection)) is None
    assert _scalar(
        pg_url, "SELECT count(*) FROM rag_collections WHERE name = %s", target.collection
    ) == 0


def test_urls_from_the_environment_never_reach_the_output(
    pg_url, pg_store_factory, monkeypatch, capsys
) -> None:
    source = _seed(pg_store_factory())
    target = pg_store_factory()
    monkeypatch.setenv("SOURCE_DATABASE_URL", pg_url)
    monkeypatch.setenv("TARGET_DATABASE_URL", pg_url)

    code = copy_script.main(
        ["--collection", source.collection, "--target-collection", target.collection]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert pg_url not in captured.out + captured.err
    assert captured.out.splitlines()[-1] == "copied 3, already present 0, checksum ok"


def test_a_missing_source_collection_is_an_error(pg_url, pg_store_factory, capsys) -> None:
    target = pg_store_factory()

    code = copy_script.main(
        ["--from", pg_url, "--to", pg_url, "--collection", "no_such_collection",
         "--target-collection", target.collection]
    )

    assert code == 1
    assert "isn't in the source database" in capsys.readouterr().err
