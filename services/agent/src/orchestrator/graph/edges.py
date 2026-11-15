"""Conditional-edge functions and wave computation for the task graph."""

from __future__ import annotations

from langgraph.types import Send

from orchestrator.config import get_settings
from orchestrator.graph.state import TaskState
from orchestrator.hitl.triggers import attempts_spent, plan_escalation
from orchestrator.planning.schemas import ExecutionPlan


def plan_gate(state: TaskState) -> str:
    """After planning: fail fast, escalate on low confidence or user request."""
    if state.get("plan") is None:
        return "failed"
    escalation = plan_escalation(
        state.get("confidence", 1.0),
        state.get("require_human_review", False),
        get_settings().plan_confidence_threshold,
    )
    return "escalate_plan" if escalation else "schedule"


def _hitl_action(state: TaskState) -> str:
    return (state.get("hitl_decision") or {}).get("action", "approve")


def route_after_plan_escalation(state: TaskState) -> str:
    action = _hitl_action(state)
    if action == "reject":
        return "end"
    if action == "take_over":
        return "deliver"
    return "schedule"


def route_after_subtask_escalation(state: TaskState) -> str:
    """A rejection, or an approval refused because the attempt budget is spent, ends the run."""
    return "end" if _hitl_action(state) in ("reject", "fail") else "schedule"


def route_after_final_gate(state: TaskState) -> str:
    return "end" if _hitl_action(state) == "reject" else "deliver"


def compute_wave(state: TaskState) -> list[dict]:
    """Payloads for every subtask that is ready to run now.

    Ready = not completed, all dependencies completed. Subtasks carrying a
    rework/failed_attempt status are re-dispatched with feedback; escalation
    for exhausted retries is decided in `gather` before this runs again. A
    subtask that has made every attempt it was allowed is held back until an
    escalation decides about it: when two subtasks escalate after the same wave,
    `gather` takes one at a time, and the other must not run meanwhile.
    """
    plan = ExecutionPlan.model_validate(state["plan"])
    results = state.get("subtask_results", {})
    completed = {sid for sid, r in results.items() if r.get("status") == "completed"}
    default_allowance = get_settings().initial_attempt_allowance

    wave = []
    for subtask in plan.subtasks:
        if subtask.id in completed:
            continue
        if not all(dep in completed for dep in subtask.depends_on):
            continue
        prior = results.get(subtask.id)
        if prior and attempts_spent(prior, default_allowance):
            continue
        wave.append(
            {
                "task_id": state["task_id"],
                "spec": subtask.model_dump(),
                "inputs": {dep: results[dep].get("output", "") for dep in subtask.depends_on},
                "feedback": prior.get("feedback") if prior else None,
                "prior": {
                    "error_count": prior.get("error_count", 0) if prior else 0,
                    "rework_count": prior.get("rework_count", 0) if prior else 0,
                    "attempts": prior.get("attempts", 0) if prior else 0,
                    "attempt_allowance": prior.get("attempt_allowance") if prior else None,
                },
            }
        )
    return wave


def dispatch(state: TaskState) -> list[Send] | str:
    """Fan out the current wave to specialist executions; with nothing to run, go
    back to `gather` while a held subtask waits for its escalation, else synthesize."""
    wave = state.get("current_wave") or []
    if wave:
        return [Send("execute", payload) for payload in wave]
    default_allowance = get_settings().initial_attempt_allowance
    if any(attempts_spent(r, default_allowance) for r in (state.get("subtask_results") or {}).values()):
        return "gather"
    return "synthesize"


def after_gather(state: TaskState) -> str:
    return "escalate" if state.get("needs_escalation") else "schedule"
