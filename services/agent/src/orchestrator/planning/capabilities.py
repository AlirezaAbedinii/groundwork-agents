"""What the planner can assign: the capability manifest, and the lint a plan must pass.

Both read the live tool registry, so they describe this deployment: a tool left out
of the registry (``code_exec`` without a sandbox, ``web_search`` without a search
provider, the ``rag_*`` tools without MCP) is absent from the manifest, and a plan
that names it fails the lint. A specialist is only ever offered the tools it owns,
so a subtask given to the wrong specialist would not fail on its own: it would be
answered from the model's memory. The lint catches that before anything runs.

A plan also lists the files the task must deliver. The lint cannot know every file a
request asks for, so it cross-checks the plan against a conservative scan of the
request (``requested_files``): a back-quoted file name that the request asks to save.
The other way round it can be exact: a deliverable must be named in the request, and a
plan that delivers no file must not write one, so the results stay in the subtasks'
reports, where synthesis can state them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orchestrator.config import get_settings
from orchestrator.llm.router import SPECIALISTS
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools.base import ToolSpec
from orchestrator.tools.registry import ToolRegistry

# Specialists whose work is tool calls. A writing subtask may work from its inputs alone.
TOOL_BOUND = ("research", "analysis", "code")

# The tool a deliverable is written with.
FILE_WRITE = "file_write"

# A back-quoted file name the request asks to save: "save", "saves", "saved" or "saving",
# then the name, either at once or after a short object ending in as, to, in or into, all
# within one sentence ("Save the note as `notes.md`", "saved as `memo.md`"). Kept
# conservative on purpose: a bare "as" is not enough ("files such as `pyproject.toml`"),
# and neither is an unquoted name or one only mentioned as an input.
_REQUESTED_FILE = re.compile(
    r"\bsav(?:e|es|ed|ing)\b"
    r"(?:[^`.;:!?\n]{0,60}?\b(?:as|to|in|into)\b)?"
    r"\s+`(?P<path>[\w./-]+\.\w{1,5})`",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolEntry:
    name: str
    summary: str  # the description's first sentence
    limit: int  # calls per task


@dataclass(frozen=True)
class Capability:
    specialist: str
    tools: tuple[ToolEntry, ...]


def requested_files(request: str) -> list[str]:
    """The file names ``request`` asks to save, each once, in the order they appear."""
    return list(dict.fromkeys(match["path"] for match in _REQUESTED_FILE.finditer(request)))


def named_in(request: str, path: str) -> bool:
    """Whether ``path`` occurs in ``request`` literally, as a whole name: not inside a longer
    one (``notes.md`` is not named by ``old_notes.md``, ``docs/notes.md`` or ``notes.md.bak``).
    Any phrasing counts, quoted or not, so it accepts names that ``requested_files`` misses."""
    return re.search(rf"(?<![\w./-]){re.escape(path)}(?![\w/-]|\.\w)", request) is not None


def summary(description: str) -> str:
    """The first sentence of a tool description, without its full stop."""
    first = " ".join(description.split()).split(". ", 1)[0]
    return first.removesuffix(".")


def _available(registry: ToolRegistry) -> dict[str, ToolSpec]:
    """Every registered tool some specialist owns, by name."""
    return {spec.name: spec for specialist in SPECIALISTS for spec in registry.tools_for(specialist)}


def manifest(registry: ToolRegistry) -> list[Capability]:
    """Each specialist with the tools it owns in this deployment, in a stable order."""
    return [
        Capability(
            specialist=specialist,
            tools=tuple(
                ToolEntry(name=spec.name, summary=summary(spec.description), limit=spec.rate_limit)
                for spec in sorted(registry.tools_for(specialist), key=lambda spec: spec.name)
            ),
        )
        for specialist in SPECIALISTS
    ]


def render_manifest(capabilities: list[Capability]) -> str:
    """The manifest as a table, one row per specialist and tool."""
    rows = ["| Specialist | Tool | What it does | Calls per task |", "|---|---|---|---|"]
    for capability in capabilities:
        if not capability.tools:
            rows.append(f"| {capability.specialist} | (none) | Works only from its inputs | |")
        for tool in capability.tools:
            rows.append(f"| {capability.specialist} | {tool.name} | {tool.summary} | {tool.limit} |")
    return "\n".join(rows)


def lint(
    plan: ExecutionPlan,
    registry: ToolRegistry,
    max_subtasks: int,
    request: str = "",
    *,
    deliverables: bool = True,
    only_requested: bool = True,
) -> list[str]:
    """What is wrong with ``plan`` for this registry and ``request``, one message per problem;
    empty if nothing.

    Pure: it reads the plan and the registry and changes neither. Each message says
    what to change, because the messages go back to the planner on its retry. Without
    ``deliverables`` (the rules of a task from before deliverables were checked) the
    plan's deliverables are not linted; without ``only_requested`` (before deliverables
    had to be named in the request) a deliverable the request does not name, and
    file_write in a plan without one, are allowed.
    """
    available = _available(registry)
    errors: list[str] = []
    if len(plan.subtasks) > max_subtasks:
        errors.append(
            f"The plan has {len(plan.subtasks)} subtasks; at most {max_subtasks} are allowed. "
            "Give work that one specialist's tools can do to a single subtask."
        )
    for subtask in plan.subtasks:
        owned = sorted(spec.name for spec in registry.tools_for(subtask.specialist))
        if not subtask.tools and subtask.specialist in TOOL_BOUND:
            errors.append(
                f"{subtask.id}: list the tools this {subtask.specialist} subtask will call "
                f"({subtask.specialist} owns: {', '.join(owned) or 'no tools'})."
            )
        for tool in dict.fromkeys(subtask.tools):  # each name once, in the plan's order
            if tool not in available:
                errors.append(
                    f"{subtask.id}: there is no tool {tool!r} in this deployment "
                    f"(available: {', '.join(sorted(available))})."
                )
            elif subtask.specialist not in available[tool].owners:
                owners = sorted(owner for owner in available[tool].owners if owner in SPECIALISTS)
                errors.append(
                    f"{subtask.id}: {tool!r} belongs to {' or '.join(owners)}, not {subtask.specialist}; "
                    "give this subtask to a specialist that owns all of its tools."
                )
    if deliverables:
        errors.extend(_lint_deliverables(plan, registry, request, only_requested))
    return errors


def _lint_deliverables(plan: ExecutionPlan, registry: ToolRegistry, request: str, only_requested: bool) -> list[str]:
    """Each deliverable's producer owns and lists file_write; every file the request asks to save is listed.

    With ``only_requested``, a deliverable the request does not name is an error (and its
    producer is not checked, since the file is to go), and so is file_write in any
    subtask of a plan that keeps no deliverable. That last check stands aside when the
    request asks to save a file: the plan is then told to list it, which needs file_write.
    """
    errors: list[str] = []
    writers = sorted(
        specialist for specialist in SPECIALISTS
        if any(spec.name == FILE_WRITE for spec in registry.tools_for(specialist))
    )
    named = [d for d in plan.deliverables if not only_requested or named_in(request, d.path)]
    for deliverable in plan.deliverables:
        producer = plan.subtask(deliverable.producer)
        if deliverable not in named:
            errors.append(
                f"The task does not name `{deliverable.path}`; remove it from deliverables. List only files "
                "the task names, with each path exactly as the task gives it."
            )
        elif producer.specialist not in writers:
            errors.append(
                f"{producer.id}: it is to write `{deliverable.path}`, but {producer.specialist} cannot call "
                f"{FILE_WRITE} (owned here by: {', '.join(writers) or 'nobody'}); give the file to a subtask "
                "whose specialist owns it."
            )
        elif FILE_WRITE not in producer.tools:
            errors.append(f"{producer.id}: it is to write `{deliverable.path}`; list {FILE_WRITE} in its tools.")
    listed = {deliverable.path for deliverable in plan.deliverables}
    asked = requested_files(request)
    for path in asked:
        if path not in listed:
            errors.append(
                f"The task asks to save `{path}`; list it in deliverables with exactly that path and the id "
                f"of the subtask that writes it with {FILE_WRITE}."
            )
    if only_requested and not named and not asked:
        errors.extend(
            f"{subtask.id}: the plan delivers no file the task names, so no subtask writes one; remove "
            f"{FILE_WRITE} from its tools and have it report its results instead."
            for subtask in plan.subtasks
            if FILE_WRITE in subtask.tools
        )
    return errors


def check(
    plan: ExecutionPlan,
    registry: ToolRegistry,
    request: str = "",
    *,
    deliverables: bool = True,
    only_requested: bool = True,
) -> tuple[list[str], bool]:
    """The lint's findings under ``MAX_PLAN_SUBTASKS``, and whether ``PLAN_LINT`` makes them reject the plan."""
    settings = get_settings()
    errors = lint(
        plan, registry, settings.max_plan_subtasks, request, deliverables=deliverables, only_requested=only_requested
    )
    return errors, bool(errors) and settings.plan_lint == "enforce"
