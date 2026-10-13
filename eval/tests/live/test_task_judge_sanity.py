"""A live check of the task-rubric judge prompt: six agent transcripts with known verdicts.

Paid (about $0.02 on claude-haiku-4-5, capped at $0.10) and deselected by default:

    .venv/bin/python -m pytest -m live tests/live/test_task_judge_sanity.py -s

Two transcripts are what the agent's mock fixtures produce for the smoke tasks, and two
are those runs doctored to fail one criterion each: FERRY-429 given the wrong meaning,
and a memo that was promised but never saved. Two more use real task rubrics to check
the rules those rubrics rely on: a criterion about a file is judged on the file, not on
a summary of it (the subtask report that states the missing fact isn't shown, as the
harness renders it); an alternative the criterion accepts passes; and "no value, not
even as general knowledge" fails an answer that adds one, without failing the other
criteria. Every verdict must be the expected one. The criteria come verbatim from the
task files, and verdicts are cached like any other judge call, so re-running unchanged
prompts is free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from budget import Budget
from judge import DEFAULT_PROVIDER, Judge, SubtaskOutput, render_task_outputs
from llm import KEY_ENV, load_env
from schemas import AgentTask

pytestmark = pytest.mark.live

MAX_COST_USD = 0.10
TASKS = Path(__file__).parents[2] / "tasks"


def load(name: str) -> dict[str, AgentTask]:
    lines = (TASKS / name).read_text(encoding="utf-8").splitlines()
    return {t.id: t for t in (AgentTask.model_validate_json(line) for line in lines if line)}


@dataclass(frozen=True)
class Transcript:
    name: str
    task_id: str
    final_output: str
    subtasks: tuple[SubtaskOutput, ...]
    expected: dict[str, bool]  # criterion id -> passed
    files: dict[str, str] = field(default_factory=dict)


# The smoke runs, as the fixtures script them (services/agent/tests/fixtures/llm).
FERRY_FINAL = "FINAL: FERRY-429 explained from the Ferry documentation."  # it_84


def ferry_subtasks(meaning: str, short_meaning: str) -> tuple[SubtaskOutput, ...]:
    return (
        SubtaskOutput(  # it_82
            "s1",
            "research",
            "Look up FERRY-429 in the internal documentation",
            f"FERRY-429-FACTS: FERRY-429 means {meaning}; the response carries Retry-After, "
            "and clients retry per ferry.retry.* (source: 04-error-codes.md).",
        ),
        SubtaskOutput(  # it_83
            "s2",
            "writing",
            "Write a short explanation of FERRY-429 and how clients should retry",
            f"FERRY-429 EXPLAINER: FERRY-429 means {short_meaning}. Wait for the Retry-After "
            "interval, then retry with backoff as ferry.retry.* sets it.",
        ),
    )


MEMO = "# Vector DB comparison\n\nQdrant leads on stars; Chroma is simplest to embed."  # it_40
VECTOR_SUBTASKS = (
    SubtaskOutput(  # it_11
        "s1",
        "research",
        "Gather facts about Chroma from the web",
        "CHROMA-FACTS: Chroma is an open-source embedding database (source: example.com/mock)",
    ),
    SubtaskOutput(  # it_21
        "s2",
        "analysis",
        "Compute the GitHub star ranking of vector databases from the demo database",
        "ANALYSIS-RANKING: qdrant (24000) > chroma (21000) > weaviate (13000)",
    ),
    SubtaskOutput("s3", "code", "Generate a comparison table using Python", "CODE-TABLE: TABLE-OK"),
)
VECTOR_WRITE = "Write a comparison memo and save it as memo.md"

OVERRIDES_GUIDE = (
    "# Overriding dependencies in tests\n\n"
    "Add an entry to `app.dependency_overrides` before calling the endpoint under test, and "
    "FastAPI uses your fake instead of the real dependency.\n\n"
    "When the test is done, call `app.dependency_overrides.clear()` so the next test gets "
    "the real dependencies again."
)

TRANSCRIPTS = [
    Transcript(
        "smoke-ferry as the fixtures run it",
        "smoke-ferry",
        FERRY_FINAL,
        ferry_subtasks(
            "the client exceeded its submission quota", "the submission quota was exceeded"
        ),
        {"c1": True, "c2": True},
    ),
    Transcript(
        "smoke-vector as the fixtures run it",
        "smoke-vector",
        "FINAL: comparison memo delivered (memo.md) — qdrant > chroma > weaviate.",  # it_50
        (*VECTOR_SUBTASKS, SubtaskOutput("s4", "writing", VECTOR_WRITE, "MEMO-DELIVERED: memo.md")),
        {"c1": True, "c2": True},
        files={"memo.md": MEMO},
    ),
    Transcript(
        "doctored: FERRY-429 given the wrong meaning",
        "smoke-ferry",
        FERRY_FINAL,
        ferry_subtasks("the Ferry server is temporarily overloaded", "the server is overloaded"),
        {"c1": False, "c2": True},
    ),
    Transcript(
        "doctored: the memo is promised, never saved",
        "smoke-vector",
        "FINAL: comparison memo drafted (still to be saved as memo.md) — qdrant > chroma > "
        "weaviate.",
        (
            *VECTOR_SUBTASKS,
            SubtaskOutput(
                "s4",
                "writing",
                VECTOR_WRITE,
                f"MEMO-DRAFT: the memo below is ready; I will save it as memo.md next.\n{MEMO}",
            ),
        ),
        {"c1": True, "c2": False},
    ),
    Transcript(
        "t06: the file lacks what the summary claims (and only a report states)",
        "t06",
        "The guide is saved as dependency-overrides.md. It explains which attribute of the app "
        "to use, what goes in it as key and value, and how to remove the overrides afterwards.",
        (
            SubtaskOutput(
                "s1",
                "research",
                "Find how to replace a FastAPI dependency in tests in the team's documentation",
                "From fastapi/advanced/testing-dependencies.md: the app has an attribute "
                "`app.dependency_overrides`, a dict. Its key is the original dependency "
                "function and its value is the override function. Reset it with "
                "`app.dependency_overrides = {}`.",
            ),
            SubtaskOutput(
                "s2",
                "writing",
                "Write the testing guide and save it as dependency-overrides.md",
                "Saved dependency-overrides.md.",
            ),
        ),
        # c1 is in the file; c2 only in the hidden report and the summary; c3 by .clear()
        {"c1": True, "c2": False, "c3": True},
        files={"dependency-overrides.md": OVERRIDES_GUIDE},
    ),
    Transcript(
        "t16: declines, then adds outside knowledge",
        "t16",
        "The team's documentation doesn't state Uvicorn's default keep-alive timeout or how to "
        "change it. It covers starting the app with `fastapi run`, but no keep-alive setting. "
        "From general knowledge rather than the documentation: Uvicorn's default keep-alive "
        "timeout is 5 seconds, and `uvicorn --timeout-keep-alive` changes it.",
        (
            SubtaskOutput(
                "s1",
                "research",
                "Find Uvicorn's default keep-alive timeout and how to change it with "
                "`fastapi run`, using only the team's documentation",
                "rag_ask refused: the documentation doesn't cover this. rag_search hits: "
                "fastapi/deployment/manually.md (running a server manually) and "
                "fastapi/fastapi-cli.md (`fastapi run`); none mentions keep-alive.",
            ),
        ),
        {"c1": True, "c2": False, "c3": True},
    ),
]


def test_the_task_judge_gives_the_expected_verdicts():
    pytest.importorskip("judge_prompts", reason="the judge prompts aren't written yet")
    load_env(Path(__file__).parents[2] / ".env")
    provider = os.environ.get("JUDGE_PROVIDER") or DEFAULT_PROVIDER
    if not os.environ.get(KEY_ENV.get(provider, "")):
        pytest.skip(f"no {KEY_ENV.get(provider, provider)} for the judge")
    judge = Judge.from_env(budget=Budget(MAX_COST_USD))
    tasks = load("smoke_tasks.jsonl") | load("agent_tasks.jsonl")

    lines, right, total = [], 0, 0
    for case in TRANSCRIPTS:
        task = tasks[case.task_id]
        outputs = render_task_outputs(task, case.files, case.subtasks)
        verdict = judge.task(task.request, case.final_output, outputs, task.rubric)
        got = {c.id: c for c in verdict.criteria}
        lines.append(f"{case.name}:")
        for cid, expected in case.expected.items():
            total += 1
            found = got.get(cid)
            ok = found is not None and found.passed == expected
            right += ok
            shown = "missing" if found is None else ("pass" if found.passed else "fail")
            why = found.reasoning if found else ""
            want = "pass" if expected else "fail"
            lines.append(f"  {'ok  ' if ok else 'MISS'} {cid} {shown} (expected {want}): {why}")

    report = "\n".join(
        [*lines, f"{right}/{total} right · {judge.cache_hits} cached · {judge.budget.summary()}"]
    )
    print(f"\n{judge.provider}:{judge.model} prompts {judge.prompt_version}\n{report}")
    assert right == total, report
