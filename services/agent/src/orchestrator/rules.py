"""The control rules a task runs under, by version.

Every task records the version of the rules it was created under (``tasks.rules_version``;
NULL, for a task from before versions were recorded, is version 1). A replay or fork runs
under its original's version, and a resumed run under the version in its graph state, so
a past run is re-run under the rules it ran with. Code reads the switches that
``rules_for`` returns and never compares version numbers.

A version changes only what decides which model calls happen or what is delivered. A
change to prompt text needs no new version: a replay serves a recorded call whose prompt
has drifted by its order among the agent's calls (observability/replay.py).
"""

from __future__ import annotations

from dataclasses import dataclass

CURRENT_RULES_VERSION = 4


class UnknownRulesVersion(ValueError):
    """A version this build has no switches for, e.g. one written by a newer deployment."""


@dataclass(frozen=True)
class Rules:
    # The plan lint (planning/capabilities.py) checks the planner's plan and a plan a
    # human modified; PLAN_LINT decides whether its findings reject the plan.
    lint_plans: bool
    # A one-subtask plan's reviewed output is the deliverable, with no synthesis call.
    deliver_single_output: bool
    # Files the plan must deliver: the lint requires the files the request asks to save
    # to be listed, each with a producer that owns file_write, and the verify step after
    # synthesis checks each was written (VERIFY_DELIVERABLES can switch the step off).
    check_deliverables: bool
    # A subtask that lists tools, none of which returned a result in an attempt, is sent
    # back for rework with no reviewer call (REVIEW_EVIDENCE can switch the check off).
    require_tool_results: bool


_SWITCHES = {
    # before ADR 0017
    1: Rules(lint_plans=False, deliver_single_output=False, check_deliverables=False, require_tool_results=False),
    # ADR 0017
    2: Rules(lint_plans=True, deliver_single_output=True, check_deliverables=False, require_tool_results=False),
    # ADR 0019
    3: Rules(lint_plans=True, deliver_single_output=True, check_deliverables=True, require_tool_results=False),
    # ADR 0020
    4: Rules(lint_plans=True, deliver_single_output=True, check_deliverables=True, require_tool_results=True),
}


def effective_version(stored: int | None) -> int:
    """The version a task runs under: ``stored``, or 1 when none was recorded."""
    return 1 if stored is None else stored


def rules_for(version: int | None) -> Rules:
    """The switches for ``version`` (None is version 1). A run, a resume and a replay all
    call this before the graph starts, so an unknown version never runs."""
    try:
        return _SWITCHES[effective_version(version)]
    except KeyError:
        raise UnknownRulesVersion(
            f"Unknown rules version {version}; this build knows {sorted(_SWITCHES)}"
        ) from None
