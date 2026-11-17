"""Conditional-edge functions, wave computation and the deliverables check for the task graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.types import Send

from orchestrator.config import get_settings
from orchestrator.graph.state import TaskState
from orchestrator.hitl.triggers import attempts_spent, plan_escalation
from orchestrator.planning.capabilities import FILE_WRITE
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
    """A rejection, or a modified plan that fails the plan lint, ends the run."""
    action = _hitl_action(state)
    if action in ("reject", "fail"):
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
                "request": state.get("request", ""),
                "rules_version": state.get("rules_version"),
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


def route_after_verify(state: TaskState) -> str:
    """A missing deliverable sends its producer back once, then ends the run."""
    outcome = state.get("deliverables_outcome")
    if outcome == "failed":
        return "end"
    return "schedule" if outcome == "redispatch" else "final_gate"


def _listed(paths: list[str]) -> str:
    return ", ".join(f"`{path}`" for path in paths)


def deliverable_feedback(paths: list[str]) -> str:
    """What a producer sent back for a missing deliverable is told: the exact paths to write."""
    one = len(paths) == 1
    return (
        f"Required {'file' if one else 'files'} not written: {_listed(paths)}. Call {FILE_WRITE} with exactly "
        f"{'that path' if one else 'each of those paths'} and the complete content, then reply with your deliverable."
    )


@dataclass(frozen=True)
class DeliverablesCheck:
    missing: list[str] = field(default_factory=list)  # deliverable paths with no successful write
    redispatch: dict[str, list[str]] = field(default_factory=dict)  # producer -> the paths it is to write
    failure: str | None = None  # the task's error when a missing file cannot be asked for again


def check_deliverables(state: TaskState, written: set[str], max_attempts: int) -> DeliverablesCheck:
    """Which of the plan's deliverables are missing from ``written``, and what to do about them.

    A path counts only if a successful file_write gave exactly that path; files written
    under other names do not matter. A missing file's producer is sent back once, with
    feedback naming the path, if it still has an attempt under ``max_attempts`` (the
    hard cap of ADR 0015) and its output is not a human's take-over. Otherwise, or if
    it was already sent back once, the task fails, naming every missing file; nothing
    is re-run then, since the task cannot complete.
    """
    plan = ExecutionPlan.model_validate(state["plan"])
    missing = [d for d in plan.deliverables if d.path not in written]
    if not missing:
        return DeliverablesCheck()
    by_producer: dict[str, list[str]] = {}
    for deliverable in missing:
        by_producer.setdefault(deliverable.producer, []).append(deliverable.path)
    results = state.get("subtask_results") or {}
    sent_back = set(state.get("deliverables_redispatched") or [])
    blocked: list[str] = []
    for sid in by_producer:
        result = results.get(sid)
        if result is None:
            blocked.append(f"{sid} never ran")
        elif sid in sent_back:
            blocked.append(f"{sid} was already sent back once to write it")
        elif result.get("human_provided"):
            blocked.append(f"{sid}'s output was a human's take-over")
        elif result.get("attempts", 0) >= max_attempts:
            blocked.append(f"{sid} has made all {max_attempts} attempts it may")
    paths = [d.path for d in missing]
    if blocked:
        noun = "file" if len(paths) == 1 else "files"
        return DeliverablesCheck(paths, failure=f"Required {noun} not written: {_listed(paths)}; {'; '.join(blocked)}")
    return DeliverablesCheck(paths, redispatch=by_producer)
