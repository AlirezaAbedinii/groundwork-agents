"""The Postgres working memory (WORKING_MEMORY_BACKEND=postgres) against a live server.

The in-memory class's behaviour suite, plus what only a store with a TTL has:
expiry, and writes that push it out for the whole task.
"""

import sqlalchemy as sa

from orchestrator.config import get_settings
from orchestrator.db.session import get_engine
from orchestrator.memory.working import PostgresWorkingMemory


def _sql(statement: str, **params):
    with get_engine().begin() as connection:
        return connection.execute(sa.text(statement), params).all()


def _expire(task_id: str) -> None:
    with get_engine().begin() as connection:
        connection.execute(
            sa.text("UPDATE working_memory SET expires_at = now() - interval '1 second' WHERE task_id = :t"),
            {"t": task_id},
        )


def _row_count(task_id: str) -> int:
    return _sql("SELECT count(*) FROM working_memory WHERE task_id = :t", t=task_id)[0][0]


def test_plan_roundtrip_and_user():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.set_plan("t1", {"subtasks": [{"id": "s1"}]})
    assert memory.get_plan("t1") == {"subtasks": [{"id": "s1"}]}
    assert memory.get_user("t1") == "alice"
    assert memory.get_plan("t2") is None
    assert memory.get_user("t2") is None


def test_outputs_grow_and_errors_keep_their_order():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.record_subtask_output("t1", "s1", "OUT-1")
    assert memory.get_subtask_outputs("t1") == {"s1": "OUT-1"}
    memory.record_subtask_output("t1", "s2", "OUT-2")
    assert memory.get_subtask_outputs("t1") == {"s1": "OUT-1", "s2": "OUT-2"}
    for error in ("boom", "bang", "crash"):
        memory.record_error("t1", "s1", error)
    assert memory.get_errors("t1") == [
        {"sid": "s1", "error": "boom"},
        {"sid": "s1", "error": "bang"},
        {"sid": "s1", "error": "crash"},
    ]


def test_intermediates_round_trip_as_json():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    calls = [{"name": "web_search", "arguments": {"query": "naïve café"}, "ok": True}]
    memory.set_intermediate("t1", "tools:s1", calls)
    memory.set_intermediate("t1", "count", 3)
    assert memory.get_intermediates("t1") == {"tools:s1": calls, "count": 3}


def test_a_rewritten_field_keeps_one_row():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.set_plan("t1", {"version": 1})
    memory.set_plan("t1", {"version": 2})
    assert memory.get_plan("t1") == {"version": 2}
    assert _row_count("t1") == 2  # user_id and plan


def test_snapshot_has_the_redis_shape():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.set_plan("t1", {"subtasks": []})
    memory.record_subtask_output("t1", "s1", "OUT-1")
    memory.record_error("t1", "s1", "boom")
    assert memory.snapshot("t1") == {
        "fields": {"user_id": "alice", "plan": '{"subtasks": []}', "subtask:s1": "OUT-1"},
        "errors": [{"sid": "s1", "error": "boom"}],
    }


def test_every_write_pushes_the_expiry_out_for_the_whole_task():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    (before,) = _sql("SELECT expires_at FROM working_memory WHERE task_id = 't1'")[0]
    memory.record_error("t1", "s1", "boom")
    expiries = _sql("SELECT DISTINCT expires_at FROM working_memory WHERE task_id = 't1'")
    assert len(expiries) == 1 and expiries[0][0] > before  # both rows moved together
    (left,) = _sql("SELECT extract(epoch FROM max(expires_at) - now()) FROM working_memory")[0]
    assert get_settings().working_memory_ttl_s - 60 < left <= get_settings().working_memory_ttl_s


def test_expired_memory_reads_as_absent():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.set_plan("t1", {"subtasks": []})
    memory.record_error("t1", "s1", "boom")
    _expire("t1")
    assert not memory.exists("t1")
    assert memory.get_plan("t1") is None
    assert memory.get_user("t1") is None
    assert memory.get_errors("t1") == []
    assert memory.snapshot("t1") == {"fields": {}, "errors": []}


def test_a_write_after_expiry_starts_the_task_over():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.set_plan("t1", {"subtasks": []})
    _expire("t1")
    memory.record_subtask_output("t1", "s1", "OUT-1")
    assert memory.snapshot("t1") == {"fields": {"subtask:s1": "OUT-1"}, "errors": []}
    assert _row_count("t1") == 1


def test_start_sweeps_expired_rows_of_other_tasks():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    _expire("t1")
    memory.start("t2", "bob")
    assert _row_count("t1") == 0
    assert memory.exists("t2")


def test_errors_alone_do_not_make_a_task_exist():
    memory = PostgresWorkingMemory()
    memory.record_error("t1", "s1", "boom")
    assert not memory.exists("t1")  # as Redis: EXISTS looks at the hash, not the error list
    assert memory.get_errors("t1") == [{"sid": "s1", "error": "boom"}]


def test_clear_scopes_to_one_task():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.record_error("t1", "s1", "boom")
    memory.start("t2", "alice")
    memory.clear("t1")
    assert not memory.exists("t1") and _row_count("t1") == 0
    assert memory.exists("t2")


def test_clear_user_only_touches_that_user():
    memory = PostgresWorkingMemory()
    memory.start("t1", "alice")
    memory.start("t2", "bob")
    memory.start("t3", "alice")
    memory.record_error("t3", "s1", "boom")
    memory.start("t4", "alice")
    _expire("t4")  # gone as far as reads go, so not counted, but still deleted
    assert memory.clear_user("alice") == 2
    assert [_row_count(t) for t in ("t1", "t3", "t4")] == [0, 0, 0]
    assert memory.exists("t2")
    assert memory.clear_user("nobody") == 0
