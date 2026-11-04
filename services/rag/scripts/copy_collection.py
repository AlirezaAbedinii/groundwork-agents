#!/usr/bin/env python3
"""Copy an indexed collection, vectors included, from one database to another.

    python scripts/copy_collection.py --collection toolchain_docs            # URLs: environment
    python scripts/copy_collection.py --collection toolchain_docs --dry-run  # what it would do
    python scripts/copy_collection.py --from URL --to URL --collection toolchain_docs \\
        --target-collection toolchain_docs_copy

The URLs come from --from and --to or, to keep a password out of the process list
and the terminal, from SOURCE_DATABASE_URL and TARGET_DATABASE_URL (make seed-cloud
reads the target's from Key Vault). Nothing is re-embedded: the target gets the
source's exact vectors, so retrieval there scores exactly as it does here.

1. Reads the source's registry row: embedding model, dimension, chunking.
2. Runs ensure_collection on the target with the same values, so the target gets
   the API's own table, HNSW index and registry row. A target collection built
   with another model or chunking is refused (ConfigError).
3. Streams the rows the target doesn't hold, ordered by chunk_id, from
   COPY ... TO STDOUT into COPY ... FROM STDIN, in one transaction: an interrupted
   copy leaves nothing behind. --format text is the fallback for a server that
   reads pgvector's binary format differently.
4. Checks both ends: the row count, and a sha256 over every row in chunk_id order,
   embeddings included.

Idempotent: a second run copies nothing. A target holding chunks the source
doesn't is refused before anything is written.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import ConfigError  # noqa: E402
from rag.indexing.vector_store import (  # noqa: E402
    _CHUNK_COLUMNS,  # the store's own column list and registry query, so the copy
    _SELECT_INFO,  # can't drift from the schema the API reads
    REGISTRY_TABLE,
    Chunking,
    CollectionInfo,
    VectorStore,
    table_name,
)

_CONNECT_TIMEOUT_S = 10


class CopyError(Exception):
    """The copy can't proceed, or its result doesn't match the source."""


@dataclass(frozen=True)
class CopyResult:
    copied: int
    present: int
    rows: int
    checksum: str
    dry_run: bool = False

    def summary(self) -> str:
        if self.dry_run:
            return f"would copy {self.copied}, already present {self.present} (dry run)"
        return f"copied {self.copied}, already present {self.present}, checksum ok"


def copy_collection(
    source_url: str,
    target_url: str,
    collection: str,
    target_collection: str | None = None,
    *,
    fmt: str = "binary",
    dry_run: bool = False,
    log=print,
) -> CopyResult:
    """Copy ``collection`` from ``source_url`` to ``target_url`` (see the module doc)."""
    import psycopg

    target = target_collection or collection
    source_table, target_table = table_name(collection), table_name(target)
    if fmt not in ("binary", "text"):
        raise ValueError(f"fmt must be 'binary' or 'text', got {fmt!r}")
    if source_url == target_url and source_table == target_table:
        raise CopyError("source and target are the same collection")

    with psycopg.connect(source_url, connect_timeout=_CONNECT_TIMEOUT_S) as src:
        src.read_only = True
        info = _info(src, collection)
        if info is None:
            raise CopyError(f"collection {collection!r} isn't in the source database")
        log(
            f"{collection} ({_describe(info)}) on {_endpoint(source_url)}"
            f" -> {target} on {_endpoint(target_url)}"
        )
        source_ids = _ids(src, source_table)

        if dry_run:
            with psycopg.connect(target_url, connect_timeout=_CONNECT_TIMEOUT_S) as dst:
                dst.read_only = True
                found = _info(dst, target)
                if found is not None and not _same_build(found, info):
                    raise ConfigError(
                        f"target collection {target!r} holds {_describe(found)} chunks"
                    )
                target_ids = _ids(dst, target_table) if found is not None else set()
            _refuse_extra(target_ids - source_ids, target)
            return CopyResult(
                copied=len(source_ids - target_ids),
                present=len(source_ids & target_ids),
                rows=len(source_ids),
                checksum="",
                dry_run=True,
            )

        store = VectorStore(target_url, target)
        try:
            store.ensure_collection(info.embedding_model, info.dim, info.chunking)
        finally:
            store.close()

        with psycopg.connect(target_url, connect_timeout=_CONNECT_TIMEOUT_S) as dst:
            target_ids = _ids(dst, target_table)
            _refuse_extra(target_ids - source_ids, target)
            missing = sorted(source_ids - target_ids)
            if missing:
                log(f"copying {len(missing)} rows ({fmt} COPY)")
                _stream(src, dst, source_table, target_table, missing, fmt)
            dst.commit()

            source_check, target_check = _checksum(src, source_table), _checksum(dst, target_table)

    if source_check != target_check:
        raise CopyError(
            f"the ends differ after the copy: source {source_check[0]} rows, sha256"
            f" {source_check[1][:12]}; target {target_check[0]} rows, sha256 {target_check[1][:12]}"
        )
    return CopyResult(
        copied=len(missing),
        present=len(source_ids) - len(missing),
        rows=source_check[0],
        checksum=source_check[1],
    )


def _stream(src, dst, source_table: str, target_table: str, ids: list[str], fmt: str) -> None:
    """COPY the ``ids`` rows out of the source and into the target, block by block."""
    from psycopg import sql

    columns = sql.SQL(f"{_CHUNK_COLUMNS}, embedding")
    out = sql.SQL(
        'COPY (SELECT {} FROM {} WHERE chunk_id = ANY({}) ORDER BY chunk_id COLLATE "C")'
        " TO STDOUT (FORMAT {})"
    ).format(columns, sql.Identifier(source_table), sql.Literal(ids), sql.SQL(fmt))
    into = sql.SQL("COPY {} ({}) FROM STDIN (FORMAT {})").format(
        sql.Identifier(target_table), columns, sql.SQL(fmt)
    )
    with src.cursor() as read, dst.cursor() as write:
        with read.copy(out) as reader, write.copy(into) as writer:
            for block in reader:
                writer.write(block)


def _checksum(conn, table: str) -> tuple[int, str]:
    """Row count and a sha256 over every row, chunk_id in byte order.

    Each row is hashed as its text form with the embedding as ``real[]``, which
    Postgres itself prints (shortest exact digits), so equal vectors hash equal
    whatever pgvector version stores them.
    """
    from psycopg import sql

    conn.execute("SET extra_float_digits = 1")
    row = sql.SQL(f"ROW({_CHUNK_COLUMNS}, embedding::real[])::text")
    count, digest = conn.execute(
        sql.SQL(
            "SELECT count(*), encode(sha256(coalesce(string_agg(sha256(convert_to({}, 'UTF8')),"
            ' \'\'::bytea ORDER BY chunk_id COLLATE "C"), \'\'::bytea)), \'hex\') FROM {}'
        ).format(row, sql.Identifier(table))
    ).fetchone()
    return count, digest


def _info(conn, collection: str) -> CollectionInfo | None:
    if conn.execute("SELECT to_regclass(%s)", (REGISTRY_TABLE,)).fetchone()[0] is None:
        return None
    row = conn.execute(_SELECT_INFO, (collection,)).fetchone()
    if row is None:
        return None
    model, dim, strategy, size, overlap = row
    return CollectionInfo(collection, model, dim, Chunking(strategy, size, overlap))


def _ids(conn, table: str) -> set[str]:
    from psycopg import sql

    rows = conn.execute(sql.SQL("SELECT chunk_id FROM {}").format(sql.Identifier(table)))
    return {row[0] for row in rows}


def _refuse_extra(extra: set[str], target: str) -> None:
    if extra:
        raise CopyError(
            f"target collection {target!r} holds {len(extra)} chunks the source doesn't"
            f" (e.g. {sorted(extra)[0]!r}); copy into an empty --target-collection instead"
        )


def _same_build(a: CollectionInfo, b: CollectionInfo) -> bool:
    return (a.embedding_model, a.dim, a.chunking) == (b.embedding_model, b.dim, b.chunking)


def _describe(info: CollectionInfo) -> str:
    c = info.chunking
    return f"{info.embedding_model}, {info.dim}-dim, {c.strategy} {c.size}/{c.overlap}"


def _endpoint(url: str) -> str:
    """host:port/dbname, never the user or the password."""
    from psycopg.conninfo import conninfo_to_dict

    parts = conninfo_to_dict(url)
    return f"{parts.get('host', 'localhost')}:{parts.get('port', 5432)}/{parts.get('dbname', '')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy an indexed collection, vectors included, between databases."
    )
    parser.add_argument("--from", dest="source_url", help="source URL (else $SOURCE_DATABASE_URL)")
    parser.add_argument("--to", dest="target_url", help="target URL (else $TARGET_DATABASE_URL)")
    parser.add_argument("--collection", required=True, help="the collection to copy")
    parser.add_argument("--target-collection", help="its name on the target (default: the same)")
    parser.add_argument(
        "--format", dest="fmt", choices=("binary", "text"), default="binary", help="COPY format"
    )
    parser.add_argument("--dry-run", action="store_true", help="report what a copy would do")
    args = parser.parse_args(argv)

    source_url = args.source_url or os.environ.get("SOURCE_DATABASE_URL", "")
    target_url = args.target_url or os.environ.get("TARGET_DATABASE_URL", "")
    if not source_url or not target_url:
        parser.error("give --from and --to, or set SOURCE_DATABASE_URL and TARGET_DATABASE_URL")

    import psycopg

    try:
        result = copy_collection(
            source_url,
            target_url,
            args.collection,
            args.target_collection,
            fmt=args.fmt,
            dry_run=args.dry_run,
        )
    except (CopyError, ConfigError, ValueError, psycopg.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
