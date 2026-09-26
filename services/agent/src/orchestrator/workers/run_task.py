"""Thin Celery wrappers around the graph runner.

The runner is async; each Celery task drives it to completion on the worker
thread's long-lived event loop (see workers/loop.py).
"""

from __future__ import annotations

from orchestrator.workers.celery_app import celery_app
from orchestrator.workers.loop import run_in_worker_loop


@celery_app.task(name="orchestrator.run_task")
def run_task_celery(task_id: str) -> None:
    from orchestrator.graph.runner import run_task

    run_in_worker_loop(run_task(task_id))


@celery_app.task(name="orchestrator.resume_task")
def resume_task_celery(task_id: str, decision: dict) -> None:
    from orchestrator.graph.runner import resume_task

    run_in_worker_loop(resume_task(task_id, decision))


@celery_app.task(name="orchestrator.replay_task")
def replay_task_celery(
    new_task_id: str,
    original_task_id: str,
    llm_call_id: str | None = None,
    response_text: str | None = None,
) -> None:
    from orchestrator.observability.replay import run_replay

    run_in_worker_loop(run_replay(new_task_id, original_task_id, llm_call_id, response_text))
