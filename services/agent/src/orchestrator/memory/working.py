"""Task-scoped working memory shared by all agents during one execution.

Backed by Redis (one hash per task plus an error list) or, where there is no
Redis, by a Postgres table with the same behaviour; WORKING_MEMORY_BACKEND picks
one and `get_working_memory()` returns it. Every agent reads and writes through
this module. Memory is scoped to a single task: it is cleared when the task
completes, and a TTL guards against leaks from crashed runs.
`InMemoryWorkingMemory` is a drop-in used by unit tests.
"""

from __future__ import annotations

import json
from datetime import timedelta
from functools import lru_cache
from typing import Any

import redis
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from orchestrator.config import get_settings
from orchestrator.db.models import WorkingMemoryRow
from orchestrator.db.session import get_sessionmaker


class WorkingMemory:
    def __init__(self, client: redis.Redis | None = None):
        self._redis = client or redis.Redis.from_url(
            get_settings().redis_url, decode_responses=True
        )

    @staticmethod
    def _key(task_id: str) -> str:
        return f"task:{task_id}:working"

    @staticmethod
    def _errors_key(task_id: str) -> str:
        return f"task:{task_id}:errors"

    def _touch(self, task_id: str) -> None:
        ttl = get_settings().working_memory_ttl_s
        self._redis.expire(self._key(task_id), ttl)
        self._redis.expire(self._errors_key(task_id), ttl)

    def start(self, task_id: str, user_id: str) -> None:
        self._redis.hset(self._key(task_id), mapping={"user_id": user_id})
        self._touch(task_id)

    def set_plan(self, task_id: str, plan: dict) -> None:
        self._redis.hset(self._key(task_id), "plan", json.dumps(plan))
        self._touch(task_id)

    def get_plan(self, task_id: str) -> dict | None:
        raw = self._redis.hget(self._key(task_id), "plan")
        return json.loads(raw) if raw else None

    def record_subtask_output(self, task_id: str, sid: str, output: str) -> None:
        self._redis.hset(self._key(task_id), f"subtask:{sid}", output)
        self._touch(task_id)

    def get_subtask_outputs(self, task_id: str) -> dict[str, str]:
        entries = self._redis.hgetall(self._key(task_id))
        return {
            field.removeprefix("subtask:"): value
            for field, value in entries.items()
            if field.startswith("subtask:")
        }

    def set_intermediate(self, task_id: str, name: str, value: Any) -> None:
        self._redis.hset(self._key(task_id), f"intermediate:{name}", json.dumps(value))
        self._touch(task_id)

    def get_intermediates(self, task_id: str) -> dict[str, Any]:
        entries = self._redis.hgetall(self._key(task_id))
        return {
            field.removeprefix("intermediate:"): json.loads(value)
            for field, value in entries.items()
            if field.startswith("intermediate:")
        }

    def record_error(self, task_id: str, sid: str, error: str) -> None:
        self._redis.rpush(self._errors_key(task_id), json.dumps({"sid": sid, "error": error}))
        self._touch(task_id)

    def get_errors(self, task_id: str) -> list[dict]:
        return [json.loads(raw) for raw in self._redis.lrange(self._errors_key(task_id), 0, -1)]

    def get_user(self, task_id: str) -> str | None:
        return self._redis.hget(self._key(task_id), "user_id")

    def snapshot(self, task_id: str) -> dict:
        return {
            "fields": self._redis.hgetall(self._key(task_id)),
            "errors": self.get_errors(task_id),
        }

    def exists(self, task_id: str) -> bool:
        return bool(self._redis.exists(self._key(task_id)))

    def clear(self, task_id: str) -> None:
        self._redis.delete(self._key(task_id), self._errors_key(task_id))

    def clear_user(self, user_id: str) -> int:
        """Delete working memory for every task belonging to a user."""
        cleared = 0
        for key in self._redis.scan_iter(match="task:*:working"):
            if self._redis.hget(key, "user_id") == user_id:
                task_id = key.split(":", 2)[1]
                self.clear(task_id)
                cleared += 1
        return cleared


class PostgresWorkingMemory:
    """The Redis store's behaviour on the ``working_memory`` table (models.WorkingMemoryRow).

    A row per (task, field) stands in for the hash, and a row without a field is an
    error. The TTL is ``expires_at``: every write pushes it out for all of the
    task's rows (Redis's ``_touch``), reads skip expired rows, and ``start`` deletes
    them, which is the cleanup Redis does by itself.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._sessions = session_factory or get_sessionmaker()

    @staticmethod
    def _live(task_id: str):
        return (WorkingMemoryRow.task_id == task_id) & (WorkingMemoryRow.expires_at > func.now())

    def _write(self, task_id: str, field: str | None, value: str) -> None:
        expires_at = func.now() + timedelta(seconds=get_settings().working_memory_ttl_s)
        row = {"task_id": task_id, "field": field, "value": value, "expires_at": expires_at}
        with self._sessions() as session, session.begin():
            # an expired task starts over, as a write to an expired Redis key does
            session.execute(
                delete(WorkingMemoryRow).where(
                    WorkingMemoryRow.task_id == task_id, WorkingMemoryRow.expires_at <= func.now()
                )
            )
            if field is None:
                session.execute(insert(WorkingMemoryRow).values(**row))
            else:
                upsert = pg_insert(WorkingMemoryRow).values(**row)
                session.execute(
                    upsert.on_conflict_do_update(
                        constraint="uq_working_memory_task_field", set_={"value": upsert.excluded.value}
                    )
                )
            session.execute(
                update(WorkingMemoryRow)
                .where(WorkingMemoryRow.task_id == task_id)
                .values(expires_at=expires_at)
            )

    def _hget(self, task_id: str, field: str) -> str | None:
        with self._sessions() as session:
            return session.scalar(
                select(WorkingMemoryRow.value).where(self._live(task_id), WorkingMemoryRow.field == field)
            )

    def _hgetall(self, task_id: str) -> dict[str, str]:
        with self._sessions() as session:
            rows = session.execute(
                select(WorkingMemoryRow.field, WorkingMemoryRow.value).where(
                    self._live(task_id), WorkingMemoryRow.field.is_not(None)
                )
            )
            return dict(rows.tuples().all())

    def start(self, task_id: str, user_id: str) -> None:
        with self._sessions() as session, session.begin():
            session.execute(delete(WorkingMemoryRow).where(WorkingMemoryRow.expires_at <= func.now()))
        self._write(task_id, "user_id", user_id)

    def set_plan(self, task_id: str, plan: dict) -> None:
        self._write(task_id, "plan", json.dumps(plan))

    def get_plan(self, task_id: str) -> dict | None:
        raw = self._hget(task_id, "plan")
        return json.loads(raw) if raw else None

    def record_subtask_output(self, task_id: str, sid: str, output: str) -> None:
        self._write(task_id, f"subtask:{sid}", output)

    def get_subtask_outputs(self, task_id: str) -> dict[str, str]:
        return {
            field.removeprefix("subtask:"): value
            for field, value in self._hgetall(task_id).items()
            if field.startswith("subtask:")
        }

    def set_intermediate(self, task_id: str, name: str, value: Any) -> None:
        self._write(task_id, f"intermediate:{name}", json.dumps(value))

    def get_intermediates(self, task_id: str) -> dict[str, Any]:
        return {
            field.removeprefix("intermediate:"): json.loads(value)
            for field, value in self._hgetall(task_id).items()
            if field.startswith("intermediate:")
        }

    def record_error(self, task_id: str, sid: str, error: str) -> None:
        self._write(task_id, None, json.dumps({"sid": sid, "error": error}))

    def get_errors(self, task_id: str) -> list[dict]:
        with self._sessions() as session:
            values = session.scalars(
                select(WorkingMemoryRow.value)
                .where(self._live(task_id), WorkingMemoryRow.field.is_(None))
                .order_by(WorkingMemoryRow.id)
            )
            return [json.loads(value) for value in values]

    def get_user(self, task_id: str) -> str | None:
        return self._hget(task_id, "user_id")

    def snapshot(self, task_id: str) -> dict:
        return {"fields": self._hgetall(task_id), "errors": self.get_errors(task_id)}

    def exists(self, task_id: str) -> bool:
        # like Redis's EXISTS on the hash key: a task with only errors doesn't count
        with self._sessions() as session:
            found = session.scalar(
                select(WorkingMemoryRow.id)
                .where(self._live(task_id), WorkingMemoryRow.field.is_not(None))
                .limit(1)
            )
            return found is not None

    def clear(self, task_id: str) -> None:
        with self._sessions() as session, session.begin():
            session.execute(delete(WorkingMemoryRow).where(WorkingMemoryRow.task_id == task_id))

    def clear_user(self, user_id: str) -> int:
        """Delete working memory for every task belonging to a user, expired rows included."""
        with self._sessions() as session, session.begin():
            owned = session.execute(
                select(WorkingMemoryRow.task_id, WorkingMemoryRow.expires_at > func.now()).where(
                    WorkingMemoryRow.field == "user_id", WorkingMemoryRow.value == user_id
                )
            ).tuples().all()
            session.execute(
                delete(WorkingMemoryRow).where(WorkingMemoryRow.task_id.in_([task_id for task_id, _ in owned]))
            )
            # the count Redis would report: its expired keys are already gone
            return sum(1 for _, live in owned if live)


def get_working_memory() -> WorkingMemory | PostgresWorkingMemory:
    """The process's working memory, on the backend WORKING_MEMORY_BACKEND names."""
    return _working_memory(get_settings().working_memory_backend)


@lru_cache
def _working_memory(backend: str) -> WorkingMemory | PostgresWorkingMemory:
    return PostgresWorkingMemory() if backend == "postgres" else WorkingMemory()


class InMemoryWorkingMemory:
    """Dict-backed drop-in for unit tests (same behaviour, no Redis)."""

    def __init__(self):
        self._fields: dict[str, dict[str, str]] = {}
        self._errors: dict[str, list[dict]] = {}

    def start(self, task_id: str, user_id: str) -> None:
        self._fields.setdefault(task_id, {})["user_id"] = user_id

    def set_plan(self, task_id: str, plan: dict) -> None:
        self._fields.setdefault(task_id, {})["plan"] = json.dumps(plan)

    def get_plan(self, task_id: str) -> dict | None:
        raw = self._fields.get(task_id, {}).get("plan")
        return json.loads(raw) if raw else None

    def record_subtask_output(self, task_id: str, sid: str, output: str) -> None:
        self._fields.setdefault(task_id, {})[f"subtask:{sid}"] = output

    def get_subtask_outputs(self, task_id: str) -> dict[str, str]:
        return {
            field.removeprefix("subtask:"): value
            for field, value in self._fields.get(task_id, {}).items()
            if field.startswith("subtask:")
        }

    def set_intermediate(self, task_id: str, name: str, value: Any) -> None:
        self._fields.setdefault(task_id, {})[f"intermediate:{name}"] = json.dumps(value)

    def get_intermediates(self, task_id: str) -> dict[str, Any]:
        return {
            field.removeprefix("intermediate:"): json.loads(value)
            for field, value in self._fields.get(task_id, {}).items()
            if field.startswith("intermediate:")
        }

    def record_error(self, task_id: str, sid: str, error: str) -> None:
        self._errors.setdefault(task_id, []).append({"sid": sid, "error": error})

    def get_errors(self, task_id: str) -> list[dict]:
        return list(self._errors.get(task_id, []))

    def get_user(self, task_id: str) -> str | None:
        return self._fields.get(task_id, {}).get("user_id")

    def snapshot(self, task_id: str) -> dict:
        return {"fields": dict(self._fields.get(task_id, {})), "errors": self.get_errors(task_id)}

    def exists(self, task_id: str) -> bool:
        return task_id in self._fields

    def clear(self, task_id: str) -> None:
        self._fields.pop(task_id, None)
        self._errors.pop(task_id, None)

    def clear_user(self, user_id: str) -> int:
        task_ids = [t for t, f in self._fields.items() if f.get("user_id") == user_id]
        for task_id in task_ids:
            self.clear(task_id)
        return len(task_ids)
