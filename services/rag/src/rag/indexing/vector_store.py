"""Chunk store: Postgres with pgvector, one table per collection.

Each collection has a row in the ``rag_collections`` registry (the embedding
model and dimension it was built with, and how its chunks were cut) and a table
``rag_chunks__<name>`` holding every chunk's text, provenance and embedding,
with an HNSW index for cosine search. Chunks upsert by their stable
``chunk_id``, so re-ingesting the same corpus is idempotent.

Serving reads check the registry first: a collection that was never seeded
raises ``FileNotFoundError`` and one embedded by another model than the
configured one raises ``ConfigError``; the API turns both into a 503.
``psycopg`` and ``pgvector`` are imported lazily, so this module imports
without the ``indexing`` extra installed.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import (
    COLLECTION_NAME_MAX_LEN,
    COLLECTION_NAME_PATTERN,
    ConfigError,
    Settings,
    get_settings,
)
from ..ingestion.chunkers import Chunk

if TYPE_CHECKING:
    from psycopg import sql
    from psycopg_pool import ConnectionPool

Vector = list[float]

REGISTRY_TABLE = "rag_collections"

_NAME_RE = re.compile(COLLECTION_NAME_PATTERN)
_CONNECT_TIMEOUT_S = 5
_POOL_TIMEOUT_S = 10.0
_CHUNK_COLUMNS = "chunk_id, source_file, section_heading, page, strategy, ordinal, text"

# Run once per store before its pool opens: pooled connections register the
# vector type on connect, so the extension has to exist first.
_BOOTSTRAP = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE TABLE IF NOT EXISTS rag_collections (name text PRIMARY KEY,"
    " embedding_model text NOT NULL, dim int NOT NULL, chunk_strategy text NOT NULL,"
    " chunk_size int NOT NULL, chunk_overlap int NOT NULL,"
    " created_at timestamptz NOT NULL DEFAULT now())",
)
_SELECT_INFO = (
    "SELECT embedding_model, dim, chunk_strategy, chunk_size, chunk_overlap"
    " FROM rag_collections WHERE name = %s"
)
_INSERT_INFO = (
    "INSERT INTO rag_collections (name, embedding_model, dim, chunk_strategy, chunk_size,"
    " chunk_overlap) VALUES (%s, %s, %s, %s, %s, %s)"
)


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with its similarity score (1 - cosine distance)."""

    chunk_id: str
    text: str
    score: float
    metadata: dict


@dataclass(frozen=True)
class Chunking:
    """How a collection's chunks were cut; recorded in the registry."""

    strategy: str
    size: int
    overlap: int

    @classmethod
    def from_settings(cls, settings: Settings) -> Chunking:
        return cls(settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap)


@dataclass(frozen=True)
class CollectionInfo:
    """A collection's registry row."""

    name: str
    embedding_model: str
    dim: int
    chunking: Chunking


def table_name(collection: str) -> str:
    """The chunk table of ``collection``; rejects any name outside the safe pattern."""
    if not _NAME_RE.fullmatch(collection):
        raise ValueError(
            f"Invalid collection name {collection!r}: use lowercase letters, digits and "
            f"underscores, starting with a letter, at most {COLLECTION_NAME_MAX_LEN} characters."
        )
    return f"rag_chunks__{collection}"


def _query_sql(table: str) -> sql.Composed:
    """Nearest chunks first. Ordering by the ``<=>`` operator lets the HNSW index serve it."""
    from psycopg import sql

    return sql.SQL(
        f"SELECT {_CHUNK_COLUMNS}, 1 - (embedding <=> %(q)s) AS score FROM {{}}"
        " ORDER BY embedding <=> %(q)s LIMIT %(k)s"
    ).format(sql.Identifier(table))


def _upsert_sql(table: str) -> sql.Composed:
    from psycopg import sql

    return sql.SQL(
        f"INSERT INTO {{}} ({_CHUNK_COLUMNS}, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (chunk_id) DO UPDATE SET source_file = EXCLUDED.source_file,"
        " section_heading = EXCLUDED.section_heading, page = EXCLUDED.page,"
        " strategy = EXCLUDED.strategy, ordinal = EXCLUDED.ordinal, text = EXCLUDED.text,"
        " embedding = EXCLUDED.embedding"
    ).format(sql.Identifier(table))


def _chunk(row: tuple) -> Chunk:
    """A row in ``_CHUNK_COLUMNS`` order back into a :class:`Chunk` (None stays None)."""
    chunk_id, source_file, section_heading, page, strategy, ordinal, text = row[:7]
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        strategy=strategy,
        ordinal=ordinal,
        section_heading=section_heading,
        page=page,
    )


def _describe(info: CollectionInfo) -> str:
    c = info.chunking
    return f"{info.embedding_model} ({info.dim}-dim), {c.strategy} {c.size}/{c.overlap}"


class VectorStore:
    """One collection of chunk embeddings + metadata in Postgres (pgvector)."""

    def __init__(
        self, database_url: str, collection: str, *, expected_model: str | None = None
    ) -> None:
        self._table = table_name(collection)  # validated before any SQL
        self.collection = collection
        self._url = database_url
        # When set, serving refuses a collection embedded by any other model.
        self._expected_model = expected_model
        self._pool: ConnectionPool | None = None
        self._info: CollectionInfo | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VectorStore:
        settings = settings or get_settings()
        return cls(
            settings.database_url, settings.collection, expected_model=settings.embedding_model
        )

    # -- writing -------------------------------------------------------------
    def ensure_collection(
        self, embedding_model: str, dim: int, chunking: Chunking
    ) -> CollectionInfo:
        """Register the collection and create its table and indexes if missing.

        An existing collection must match: another embedding model, dimension or
        chunking raises ``ConfigError`` rather than mixing incomparable vectors.
        """
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        from psycopg import sql

        wanted = CollectionInfo(self.collection, embedding_model, dim, chunking)
        table = sql.Identifier(self._table)
        with self._connection() as conn:
            # Two seeds of the same collection take turns here.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self._table,))
            found = self._fetch_info(conn)
            if found is None:
                conn.execute(
                    _INSERT_INFO,
                    (
                        self.collection,
                        embedding_model,
                        dim,
                        chunking.strategy,
                        chunking.size,
                        chunking.overlap,
                    ),
                )
            elif found != wanted:
                raise ConfigError(
                    f"Collection {self.collection!r} holds {_describe(found)} chunks, but this "
                    f"run makes {_describe(wanted)} ones. Seed into another COLLECTION instead."
                )
            conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (chunk_id text PRIMARY KEY,"
                    " source_file text NOT NULL, section_heading text, page int,"
                    " strategy text NOT NULL, ordinal int NOT NULL, text text NOT NULL,"
                    " embedding vector({}) NOT NULL)"
                ).format(table, sql.Literal(dim))
            )
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
                ).format(sql.Identifier(f"{self._table}_hnsw"), table)
            )
            conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (source_file, ordinal)").format(
                    sql.Identifier(f"{self._table}_source"), table
                )
            )
        self._info = None  # the next serving read re-checks the registry
        return found or wanted

    def add(self, chunks: list[Chunk], embeddings: list[Vector]) -> int:
        """Upsert chunks + their embeddings by ``chunk_id``; return how many."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0
        info = self._require()
        from pgvector import Vector as PgVector

        rows = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            if len(vector) != info.dim:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has a {len(vector)}-dim embedding; collection "
                    f"{self.collection!r} stores {info.dim}-dim vectors."
                )
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.source_file,
                    chunk.section_heading,
                    chunk.page,
                    chunk.strategy,
                    chunk.ordinal,
                    chunk.text,
                    PgVector(vector),
                )
            )
        with self._connection() as conn, conn.cursor() as cur:
            cur.executemany(_upsert_sql(self._table), rows)
        return len(chunks)

    # -- reading -------------------------------------------------------------
    def count(self) -> int:
        """Number of chunks stored (0 before the collection is seeded)."""
        if not self._registered():
            return 0
        with self._connection() as conn:
            return conn.execute(self._sql("SELECT count(*) FROM {}")).fetchone()[0]

    def list_sources(self) -> dict[str, int]:
        """Map of ``source_file`` -> chunk count ({} before the collection is seeded)."""
        if not self._registered():
            return {}
        with self._connection() as conn:
            rows = conn.execute(
                self._sql("SELECT source_file, count(*) FROM {} GROUP BY source_file")
            ).fetchall()
        return dict(sorted(rows))

    def query(self, query_embedding: Vector, top_k: int = 10) -> list[ScoredChunk]:
        """Return the ``top_k`` nearest chunks by cosine similarity."""
        self._require()
        from pgvector import Vector as PgVector

        with self._connection() as conn:
            rows = conn.execute(
                _query_sql(self._table), {"q": PgVector(query_embedding), "k": top_k}
            ).fetchall()
        return [
            ScoredChunk(
                chunk_id=row[0], text=row[6], score=float(row[7]), metadata=_chunk(row).metadata()
            )
            for row in rows
        ]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, ordered by (source_file, ordinal)."""
        self._require()
        with self._connection() as conn:
            rows = conn.execute(
                self._sql(
                    f'SELECT {_CHUNK_COLUMNS} FROM {{}} ORDER BY source_file COLLATE "C",'
                    " ordinal, chunk_id"
                )
            ).fetchall()
        return [_chunk(row) for row in rows]

    def chunks_for_source(self, source_file: str) -> list[Chunk]:
        """One source's chunks in document order ([] if the source isn't indexed)."""
        self._require()
        with self._connection() as conn:
            rows = conn.execute(
                self._sql(
                    f"SELECT {_CHUNK_COLUMNS} FROM {{}} WHERE source_file = %s"
                    " ORDER BY ordinal, chunk_id"
                ),
                (source_file,),
            ).fetchall()
        return [_chunk(row) for row in rows]

    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        """The subset of ``ids`` already stored."""
        ids = list(ids)
        if not ids:
            return set()
        self._require()
        with self._connection() as conn:
            rows = conn.execute(
                self._sql("SELECT chunk_id FROM {} WHERE chunk_id = ANY(%s)"), (ids,)
            ).fetchall()
        return {row[0] for row in rows}

    def collection_info(self) -> CollectionInfo | None:
        """The collection's registry row, or None if it was never seeded."""
        with self._connection() as conn:
            return self._fetch_info(conn)

    def close(self) -> None:
        """Close the connection pool (the next call opens a new one)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # -- internals -----------------------------------------------------------
    def _require(self) -> CollectionInfo:
        """The registry row, checked once: seeded, and by the expected model."""
        if self._info is None:
            info = self.collection_info()
            if info is None:
                raise FileNotFoundError(
                    f"Collection {self.collection!r} is not indexed; seed first: "
                    "python scripts/seed.py (or docker compose run --rm seed)."
                )
            if self._expected_model is not None and info.embedding_model != self._expected_model:
                raise ConfigError(
                    f"Collection {self.collection!r} was embedded with "
                    f"{info.embedding_model!r}, but EMBEDDING_MODEL is "
                    f"{self._expected_model!r}. Serve it with the model it was built with, "
                    "or seed another COLLECTION."
                )
            self._info = info
        return self._info

    def _registered(self) -> bool:
        return self._info is not None or self.collection_info() is not None

    def _fetch_info(self, conn) -> CollectionInfo | None:
        row = conn.execute(_SELECT_INFO, (self.collection,)).fetchone()
        if row is None:
            return None
        model, dim, strategy, size, overlap = row
        return CollectionInfo(self.collection, model, dim, Chunking(strategy, size, overlap))

    def _sql(self, template: str) -> sql.Composed:
        from psycopg import sql

        return sql.SQL(template).format(sql.Identifier(self._table))

    def _connection(self):
        return self._get_pool().connection()

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            try:
                import psycopg
                from pgvector.psycopg import register_vector
                from psycopg_pool import ConnectionPool
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "The chunk store requires psycopg and pgvector. "
                    'Install: pip install -e ".[indexing]"'
                ) from exc
            with psycopg.connect(
                self._url, autocommit=True, connect_timeout=_CONNECT_TIMEOUT_S
            ) as conn:
                for statement in _BOOTSTRAP:
                    try:
                        conn.execute(statement)
                    except (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateObject):
                        pass  # another process created it at the same moment
            pool = ConnectionPool(
                self._url,
                min_size=1,
                max_size=4,
                open=False,
                configure=register_vector,
                check=ConnectionPool.check_connection,
                timeout=_POOL_TIMEOUT_S,
                kwargs={"connect_timeout": _CONNECT_TIMEOUT_S},
                name=f"rag-{self.collection}",
            )
            pool.open(wait=True, timeout=_POOL_TIMEOUT_S)
            self._pool = pool
        return self._pool
