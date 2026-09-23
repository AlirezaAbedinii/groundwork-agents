"""Task runner: executes tasks on the production graph.

The graph runs under ``ainvoke``. Its checkpointer connection is opened per
run (see graph/checkpointing.py), so the graph is compiled per run too; the two
cost a few tens of milliseconds against runs that take seconds. The
loop-independent dependencies (clients, stores, memory) are built once per
process.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot

from orchestrator.db.repo import DBInvocationStore, DBLLMCallStore, DBTaskRepo, MemoryEventStore
from orchestrator.graph.builder import build_graph
from orchestrator.graph.checkpointing import open_checkpointer
from orchestrator.hitl.queue import ApprovalQueue
from orchestrator.llm.clients import get_llm_client
from orchestrator.memory.longterm import LongTermMemory
from orchestrator.memory.working import WorkingMemory
from orchestrator.observability.tracing import TracedLLMClient, setup_tracing, task_run_span
from orchestrator.tools.defaults import build_default_registry

logger = logging.getLogger(__name__)


@lru_cache
def _dependencies() -> dict:
    setup_tracing()
    return dict(
        llm=TracedLLMClient(get_llm_client(), calls=DBLLMCallStore()),
        registry=build_default_registry(DBInvocationStore()),
        repo=DBTaskRepo(),
        working=WorkingMemory(),
        longterm=LongTermMemory(),
        memory_events=MemoryEventStore(),
        approvals=ApprovalQueue(),
    )


@asynccontextmanager
async def production_graph() -> AsyncIterator[CompiledStateGraph]:
    """The production graph, checkpointing to Postgres over a connection that lives for one run."""
    async with open_checkpointer() as checkpointer:
        yield build_graph(checkpointer=checkpointer, **_dependencies())


async def task_state(task_id: str) -> StateSnapshot:
    """The task's latest checkpointed graph state."""
    async with production_graph() as graph:
        return await graph.aget_state({"configurable": {"thread_id": task_id}})


async def run_task(task_id: str) -> None:
    repo = DBTaskRepo()
    bundle = repo.get_task(task_id)
    if bundle is None:
        logger.error("run_task: task %s not found", task_id)
        return
    initial = {
        "task_id": task_id,
        "request": bundle["request"],
        "user_id": bundle["user_id"],
        "require_human_review": bundle["require_human_review"],
        "subtask_results": {},
        "dispatch_log": [],
    }
    try:
        # build (and set up tracing) before opening the root span — a root started
        # under the default no-op provider would never be recorded
        async with production_graph() as graph:
            with task_run_span(task_id, "task"):
                await graph.ainvoke(initial, config={"configurable": {"thread_id": task_id}})
    except Exception as error:
        logger.exception("Task %s crashed", task_id)
        repo.set_status(task_id, "failed", error=str(error))


async def resume_task(task_id: str, decision: dict) -> None:
    """Resume a paused task from its checkpoint with the human decision."""
    from langgraph.types import Command

    try:
        async with production_graph() as graph:
            with task_run_span(task_id, "task:resume"):
                await graph.ainvoke(
                    Command(resume=decision), config={"configurable": {"thread_id": task_id}}
                )
    except Exception as error:
        logger.exception("Task %s crashed while resuming", task_id)
        DBTaskRepo().set_status(task_id, "failed", error=str(error))
