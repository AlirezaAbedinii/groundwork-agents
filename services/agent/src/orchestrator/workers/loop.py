"""One long-lived event loop per worker thread, for the Celery wrappers.

Async clients belong to the event loop that created them. The providers' HTTP
connection pools, which LangChain shares process-wide, break when a later task
uses them from a new loop ("Event loop is closed"), so a worker runs every task
on the same loop instead of a fresh ``asyncio.run()`` loop per task.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_local = threading.local()


def run_in_worker_loop(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run *coroutine* to completion on this thread's long-lived event loop."""
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = _local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)
