"""Fixtures for the ``pg`` tests: throwaway collections in the RAG Postgres.

These need Postgres with pgvector at DATABASE_URL (locally ``docker compose up
-d --wait db``; a service container in CI) and skip when it can't be reached,
so CI checks that the database is up before running them.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator

import pytest

from rag.config import Settings


@pytest.fixture(scope="session")
def pg_url() -> str:
    """DATABASE_URL, once it answers; otherwise the requesting test skips."""
    psycopg = pytest.importorskip("psycopg", reason="needs the indexing extra")
    pytest.importorskip("pgvector", reason="needs the indexing extra")
    url = Settings(_env_file=None).database_url
    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"RAG Postgres unreachable ({type(exc).__name__}); "
            "run: docker compose up -d --wait db"
        )
    return url


@pytest.fixture
def pg_store_factory(pg_url: str) -> Iterator[Callable[..., object]]:
    """Make stores on fresh collections; all of them are dropped afterwards."""
    from rag.indexing.vector_store import VectorStore

    stores: list[VectorStore] = []

    def make(*, name: str | None = None, expected_model: str | None = None) -> VectorStore:
        store = VectorStore(
            pg_url, name or f"test_{uuid.uuid4().hex[:12]}", expected_model=expected_model
        )
        stores.append(store)
        return store

    yield make
    for store in stores:
        store.close()
    _drop_collections(pg_url, {store.collection for store in stores})


@pytest.fixture
def pg_store(pg_store_factory):
    """A store on a fresh collection, dropped after the test."""
    return pg_store_factory()


def _drop_collections(url: str, names: set[str]) -> None:
    import psycopg
    from psycopg import sql

    from rag.indexing.vector_store import REGISTRY_TABLE, table_name

    with psycopg.connect(url, autocommit=True) as conn:
        for name in names:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name(name)))
            )
        if conn.execute("SELECT to_regclass(%s)", (REGISTRY_TABLE,)).fetchone()[0]:
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE name = ANY(%s)").format(
                    sql.Identifier(REGISTRY_TABLE)
                ),
                (sorted(names),),
            )
