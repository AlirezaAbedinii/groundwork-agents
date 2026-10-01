"""Long-term semantic memory in Postgres with pgvector.

Three kinds — episodes (what was asked and what approach worked), facts
(domain facts discovered), preferences (user preferences observed) — share
the ``memories`` table, each entry carrying user_id, task_id, created_at,
access_count, and an importance score. Embeddings are computed client-side
(memory/embeddings.py) and stored as pgvector vectors.

A query filters one user's entries of one kind in SQL, then ranks them by
exact cosine distance: per-user sets are small, and filtering first can't
lose results the way searching an approximate index before filtering can.
``InMemoryLongTermMemory`` offers the same interface for unit tests.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session, load_only, sessionmaker

from orchestrator.db.models import MemoryRow
from orchestrator.db.session import get_sessionmaker
from orchestrator.memory.embeddings import embed_texts
from orchestrator.memory.management import compute_importance

KINDS = ("episodes", "facts", "preferences")
# A memory's identity lives in columns; the rest of its metadata is attributes.
_IDENTITY = ("kind", "user_id", "task_id")


@dataclass
class MemoryHit:
    id: str
    text: str
    kind: str
    metadata: dict
    distance: float | None = None


class LongTermMemory:
    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._sessions = session_factory or get_sessionmaker()

    def add(
        self,
        kind: str,
        text: str,
        *,
        user_id: str,
        task_id: str | None = None,
        extra: dict | None = None,
    ) -> str:
        _check_kind(kind)
        memory_id = uuid.uuid4().hex
        embedding = embed_texts([text])[0]  # before the transaction: it may call a provider
        with self._sessions() as session, session.begin():
            session.add(
                MemoryRow(
                    id=memory_id,
                    kind=kind,
                    user_id=user_id,
                    task_id=task_id or None,
                    text=text,
                    embedding=embedding,
                    attributes=_new_attributes(extra),
                )
            )
        return memory_id

    def query(self, kind: str, text: str, *, user_id: str, k: int = 3) -> list[MemoryHit]:
        _check_kind(kind)
        mine = (MemoryRow.kind == kind, MemoryRow.user_id == user_id)
        with self._sessions() as session:
            if session.scalar(sa.select(MemoryRow.id).where(*mine).limit(1)) is None:
                return []  # nothing to rank, so no embedding call
        distance = MemoryRow.embedding.cosine_distance(embed_texts([text])[0]).label("distance")
        with self._sessions() as session:
            rows = session.execute(
                sa.select(MemoryRow.id, MemoryRow.text, MemoryRow.task_id, MemoryRow.attributes, distance)
                .where(*mine)
                .order_by(distance, MemoryRow.id)
                .limit(max(1, k))
            ).all()
        return [
            MemoryHit(
                id=row.id,
                text=row.text,
                kind=kind,
                metadata=_metadata(kind, user_id, row.task_id, row.attributes),
                distance=float(row.distance),
            )
            for row in rows
        ]

    def get_all(self, user_id: str) -> dict[str, list[dict]]:
        with self._sessions() as session:
            rows = session.execute(
                sa.select(MemoryRow.id, MemoryRow.kind, MemoryRow.text, MemoryRow.task_id, MemoryRow.attributes)
                .where(MemoryRow.user_id == user_id)
                .order_by(MemoryRow.created_at, MemoryRow.id)
            ).all()
        return {
            kind: _by_importance(
                (row.id, row.text, _metadata(kind, user_id, row.task_id, row.attributes))
                for row in rows
                if row.kind == kind
            )
            for kind in KINDS
        }

    def all_items(self, kind: str, user_id: str | None = None, with_embeddings: bool = False) -> dict:
        """Parallel lists, oldest first: ids, documents, metadatas, embeddings (None unless asked)."""
        _check_kind(kind)
        columns = [MemoryRow.id, MemoryRow.user_id, MemoryRow.task_id, MemoryRow.text, MemoryRow.attributes]
        if with_embeddings:
            columns.append(MemoryRow.embedding)
        stmt = sa.select(*columns).where(MemoryRow.kind == kind).order_by(MemoryRow.created_at, MemoryRow.id)
        if user_id:
            stmt = stmt.where(MemoryRow.user_id == user_id)
        with self._sessions() as session:
            rows = session.execute(stmt).all()
        return {
            "ids": [row.id for row in rows],
            "documents": [row.text for row in rows],
            "metadatas": [_metadata(kind, row.user_id, row.task_id, row.attributes) for row in rows],
            "embeddings": [row.embedding for row in rows] if with_embeddings else None,
        }

    def bump_access(self, kind: str, ids: list[str]) -> None:
        if not ids:
            return
        _check_kind(kind)
        now = time.time()
        with self._sessions() as session, session.begin():
            for row in session.scalars(self._lock(kind, ids)):
                row.attributes = _accessed(row.attributes, now)

    def set_metadata(self, kind: str, ids: list[str], metadatas: list[dict]) -> None:
        """Merge these values into each memory's metadata; identity keys are ignored."""
        _check_kind(kind)
        updates = dict(zip(ids, metadatas, strict=True))
        if not updates:
            return
        with self._sessions() as session, session.begin():
            for row in session.scalars(self._lock(kind, list(updates))):
                row.attributes = {**row.attributes, **_without_identity(updates[row.id])}

    def delete(self, kind: str, ids: list[str]) -> None:
        if not ids:
            return
        _check_kind(kind)
        with self._sessions() as session, session.begin():
            session.execute(
                sa.delete(MemoryRow)
                .where(MemoryRow.kind == kind, MemoryRow.id.in_(list(ids)))
                .execution_options(synchronize_session=False)
            )

    def delete_user(self, user_id: str) -> dict[str, int]:
        with self._sessions() as session, session.begin():
            kinds = (
                session.execute(
                    sa.delete(MemoryRow)
                    .where(MemoryRow.user_id == user_id)
                    .returning(MemoryRow.kind)
                    .execution_options(synchronize_session=False)
                )
                .scalars()
                .all()
            )
        return {kind: kinds.count(kind) for kind in KINDS}

    @staticmethod
    def _lock(kind: str, ids: list[str]):
        """These memories' attributes, locked until commit so concurrent updates queue."""
        return (
            sa.select(MemoryRow)
            .options(load_only(MemoryRow.attributes))
            .where(MemoryRow.kind == kind, MemoryRow.id.in_(list(ids)))
            .with_for_update()
        )


class InMemoryLongTermMemory:
    """Dict-backed long-term memory for unit tests (no Postgres required).

    Same interface and return shapes as :class:`LongTermMemory`, with the exact
    cosine distance computed in Python.
    """

    def __init__(self) -> None:
        # id -> kind, user_id, task_id, text, embedding, attributes (insertion order = age)
        self._rows: dict[str, dict] = {}

    def add(
        self,
        kind: str,
        text: str,
        *,
        user_id: str,
        task_id: str | None = None,
        extra: dict | None = None,
    ) -> str:
        _check_kind(kind)
        memory_id = uuid.uuid4().hex
        self._rows[memory_id] = {
            "kind": kind,
            "user_id": user_id,
            "task_id": task_id or None,
            "text": text,
            "embedding": embed_texts([text])[0],
            "attributes": _new_attributes(extra),
        }
        return memory_id

    def query(self, kind: str, text: str, *, user_id: str, k: int = 3) -> list[MemoryHit]:
        _check_kind(kind)
        mine = self._select(kind, user_id)
        if not mine:
            return []
        vector = embed_texts([text])[0]
        ranked = sorted((_cosine_distance(row["embedding"], vector), mid, row) for mid, row in mine)
        return [
            MemoryHit(
                id=mid,
                text=row["text"],
                kind=kind,
                metadata=_metadata(kind, user_id, row["task_id"], row["attributes"]),
                distance=distance,
            )
            for distance, mid, row in ranked[: max(1, k)]
        ]

    def get_all(self, user_id: str) -> dict[str, list[dict]]:
        return {
            kind: _by_importance(
                (mid, row["text"], _metadata(kind, user_id, row["task_id"], row["attributes"]))
                for mid, row in self._select(kind, user_id)
            )
            for kind in KINDS
        }

    def all_items(self, kind: str, user_id: str | None = None, with_embeddings: bool = False) -> dict:
        _check_kind(kind)
        rows = self._select(kind, user_id or None)
        return {
            "ids": [mid for mid, _ in rows],
            "documents": [row["text"] for _, row in rows],
            "metadatas": [
                _metadata(kind, row["user_id"], row["task_id"], row["attributes"]) for _, row in rows
            ],
            "embeddings": [list(row["embedding"]) for _, row in rows] if with_embeddings else None,
        }

    def bump_access(self, kind: str, ids: list[str]) -> None:
        if not ids:
            return
        _check_kind(kind)
        now = time.time()
        for row in self._rows_of(kind, ids):
            row["attributes"] = _accessed(row["attributes"], now)

    def set_metadata(self, kind: str, ids: list[str], metadatas: list[dict]) -> None:
        _check_kind(kind)
        for mid, metadata in zip(ids, metadatas, strict=True):
            row = self._rows.get(mid)
            if row is not None and row["kind"] == kind:
                row["attributes"] = {**row["attributes"], **_without_identity(metadata)}

    def delete(self, kind: str, ids: list[str]) -> None:
        if not ids:
            return
        _check_kind(kind)
        for mid in ids:
            if self._rows.get(mid, {}).get("kind") == kind:
                del self._rows[mid]

    def delete_user(self, user_id: str) -> dict[str, int]:
        doomed = self._select(user_id=user_id)
        for mid, _ in doomed:
            del self._rows[mid]
        return {kind: sum(row["kind"] == kind for _, row in doomed) for kind in KINDS}

    def _select(self, kind: str | None = None, user_id: str | None = None) -> list[tuple[str, dict]]:
        return [
            (mid, row)
            for mid, row in self._rows.items()
            if (kind is None or row["kind"] == kind) and (user_id is None or row["user_id"] == user_id)
        ]

    def _rows_of(self, kind: str, ids) -> list[dict]:
        return [self._rows[mid] for mid in ids if self._rows.get(mid, {}).get("kind") == kind]


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown memory kind {kind!r}; expected one of {KINDS}")


def _new_attributes(extra: dict | None) -> dict:
    now = time.time()
    attributes = {
        "created_at": now,
        "last_accessed_at": now,
        "access_count": 0,
        "importance": compute_importance(0, now, now=now),
    }
    attributes.update(_without_identity(extra or {}))
    return attributes


def _without_identity(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if key not in _IDENTITY}


def _accessed(attributes: dict, now: float) -> dict:
    count = int(attributes.get("access_count", 0)) + 1
    return {
        **attributes,
        "access_count": count,
        "last_accessed_at": now,
        "importance": compute_importance(count, now, now=now),
    }


def _metadata(kind: str, user_id: str, task_id: str | None, attributes: dict) -> dict:
    """The metadata callers read: identity ("" for no task) plus the attributes."""
    return {"kind": kind, "user_id": user_id, "task_id": task_id or "", **attributes}


def _by_importance(items) -> list[dict]:
    """Dashboard rows from (id, text, metadata), most important first."""
    rows = [
        {
            "id": memory_id,
            "text": text,
            "task_id": meta.get("task_id"),
            "importance": meta.get("importance"),
            "access_count": meta.get("access_count"),
            "created_at": meta.get("created_at"),
            "last_accessed_at": meta.get("last_accessed_at"),
        }
        for memory_id, text, meta in items
    ]
    return sorted(rows, key=lambda row: row["importance"] or 0, reverse=True)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0
