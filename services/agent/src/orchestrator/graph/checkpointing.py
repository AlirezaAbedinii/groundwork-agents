"""Postgres checkpointer for graph state (pause/resume, replay)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from orchestrator.config import get_settings

_tables_ready = False


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """An async Postgres checkpointer on its own connection, closed on exit.

    AsyncPostgresSaver is bound to the event loop it is created on, and graphs
    here run on short-lived loops (one per Celery task via asyncio.run, one per
    TestClient request), so a cached saver would outlive its loop and keep its
    connection open. Each graph run opens one and closes it instead.
    """
    global _tables_ready
    url = get_settings().database_url.replace("+psycopg", "")
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        if not _tables_ready:  # creates/migrates the checkpoint tables; idempotent
            await saver.setup()
            _tables_ready = True
        yield saver
