"""The five escalation triggers, as pure predicates over plain values.

Each returns an `Escalation` when the condition holds, else None — no I/O, no
settings access, no graph state; callers supply thresholds. That keeps every
trigger unit-testable in isolation and the graph wiring thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Trigger(str, Enum):
    LOW_PLAN_CONFIDENCE = "low_plan_confidence"
    SPECIALIST_DOUBLE_FAILURE = "specialist_double_failure"
    SENSITIVE_OPERATION = "sensitive_operation"
    LOW_REVIEW_SCORE = "low_review_score"
    USER_REQUESTED = "user_requested"


@dataclass(frozen=True)
class Escalation:
    trigger: Trigger
    reason: str


def low_plan_confidence(confidence: float, threshold: float) -> Escalation | None:
    if confidence < threshold:
        return Escalation(
            Trigger.LOW_PLAN_CONFIDENCE,
            f"Plan confidence {confidence:.2f} is below the threshold {threshold:.2f}",
        )
    return None


def user_requested(require_human_review: bool) -> Escalation | None:
    if require_human_review:
        return Escalation(Trigger.USER_REQUESTED, "The user explicitly requested human review")
    return None


def specialist_double_failure(
    sid: str, error_count: int, last_error: str | None = None, max_failures: int = 2
) -> Escalation | None:
    if error_count >= max_failures:
        detail = f": {last_error}" if last_error else ""
        return Escalation(
            Trigger.SPECIALIST_DOUBLE_FAILURE,
            f"Subtask {sid} failed {error_count} times{detail}",
        )
    return None


def sensitive_operation(tool_name: str, sensitive: bool) -> Escalation | None:
    if sensitive:
        return Escalation(
            Trigger.SENSITIVE_OPERATION,
            f"Tool {tool_name} is about to perform a sensitive operation",
        )
    return None


def low_review_score(
    sid: str, score: int | None, rework_count: int, threshold: int, max_retries: int
) -> Escalation | None:
    if rework_count >= max_retries and score is not None and score < threshold:
        return Escalation(
            Trigger.LOW_REVIEW_SCORE,
            f"Subtask {sid} review score {score} stayed below {threshold} "
            f"after {rework_count} rework cycles",
        )
    return None


def no_tool_result(sid: str, tools: list[str], rework_count: int, max_retries: int) -> Escalation | None:
    """A subtask reworked because none of the tools it lists returned a result (graph/builder.py):
    a review outcome reached without a reviewer, so it escalates as a low review score does."""
    if rework_count >= max_retries:
        return Escalation(
            Trigger.LOW_REVIEW_SCORE,
            f"Subtask {sid}: none of its tools ({', '.join(tools)}) returned a result "
            f"after {rework_count} rework cycles",
        )
    return None


def plan_escalation(
    confidence: float, require_human_review: bool, threshold: float
) -> Escalation | None:
    """Pre-execution gate: low confidence takes precedence over user request."""
    return low_plan_confidence(confidence, threshold) or user_requested(require_human_review)


def attempts_spent(result: dict, default_allowance: int) -> bool:
    """True for a subtask that has not completed and has made every attempt it was
    allowed. Entries from before the allowance existed take `default_allowance`."""
    return result.get("status") in ("failed_attempt", "rework") and (
        result.get("attempts", 0) >= result.get("attempt_allowance", default_allowance)
    )


def subtask_escalation(
    sid: str,
    result: dict,
    *,
    review_threshold: int,
    max_retries: int,
    default_allowance: int,
    max_failures: int = 2,
) -> Escalation | None:
    """Post-wave gate over one subtask's result entry.

    Fires when a counter reaches its limit, or when the subtask has made every attempt
    it was allowed (`attempt_allowance`; `default_allowance` for an entry without one,
    which callers take from `Settings.initial_attempt_allowance` so that the wave's
    hold and this gate agree). An approval
    raises the allowance by one and leaves the counters as they are, so the next
    attempt that does not complete escalates again.
    """
    status = result.get("status")
    if status == "failed_attempt":
        escalation = specialist_double_failure(
            sid, result.get("error_count", 0), result.get("error"), max_failures
        )
    elif status == "rework" and result.get("no_tool_result"):
        escalation = no_tool_result(sid, result["no_tool_result"], result.get("rework_count", 0), max_retries)
    elif status == "rework":
        escalation = low_review_score(
            sid, result.get("score"), result.get("rework_count", 0), review_threshold, max_retries
        )
    else:
        return None
    if escalation is None and attempts_spent(result, default_allowance):
        attempts = result.get("attempts", 0)
        if status == "failed_attempt":
            detail = f": {result['error']}" if result.get("error") else ""
            escalation = Escalation(
                Trigger.SPECIALIST_DOUBLE_FAILURE,
                f"Subtask {sid} failed on attempt {attempts}, the last one it was allowed{detail}",
            )
        elif result.get("no_tool_result"):
            escalation = Escalation(
                Trigger.LOW_REVIEW_SCORE,
                f"Subtask {sid}: none of its tools ({', '.join(result['no_tool_result'])}) returned a result "
                f"on attempt {attempts}, the last one it was allowed",
            )
        else:
            escalation = Escalation(
                Trigger.LOW_REVIEW_SCORE,
                f"Subtask {sid} review score {result.get('score')} was below {review_threshold} "
                f"on attempt {attempts}, the last one it was allowed",
            )
    return escalation
