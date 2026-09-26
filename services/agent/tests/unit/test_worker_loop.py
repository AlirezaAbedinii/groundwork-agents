"""Celery wrappers run every task on one event loop per worker thread."""

import asyncio
import threading

from orchestrator.workers.loop import run_in_worker_loop


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_tasks_on_a_thread_share_one_loop_and_what_lives_on_it():
    async def make_future() -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def resolve(future: asyncio.Future) -> str:
        asyncio.get_running_loop().call_soon(future.set_result, "resolved by the second task")
        return await future  # pending here: under asyncio.run per task this fails, it belongs to a dead loop

    def worker(results: dict) -> None:
        # something bound to the loop by one task is still usable by the next,
        # as a cached HTTP connection pool has to be
        future = run_in_worker_loop(make_future())
        results["resolved"] = run_in_worker_loop(resolve(future))
        results["loops"] = {run_in_worker_loop(_running_loop()) for _ in range(3)}

    results: dict = {}
    thread = threading.Thread(target=worker, args=(results,))
    thread.start()
    thread.join()
    assert results["resolved"] == "resolved by the second task"
    assert len(results["loops"]) == 1


def test_each_thread_gets_its_own_loop_and_a_closed_loop_is_replaced():
    def worker(results: list) -> None:
        loop = run_in_worker_loop(_running_loop())
        loop.close()
        results.append((loop, run_in_worker_loop(_running_loop())))

    results: list = []
    threads = [threading.Thread(target=worker, args=(results,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (closed_a, fresh_a), (closed_b, fresh_b) = results
    assert fresh_a is not closed_a and fresh_b is not closed_b
    assert fresh_a is not fresh_b
