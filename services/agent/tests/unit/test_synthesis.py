"""The synthesis prompt: a manifest of the run, and rules that keep the answer to it.

Synthesis sees each subtask's tool-call outcomes (from the invocation log), what it
reported, and the deliverables with whether each was written; it is told to add no
facts, to repeat that the documentation does not cover something when a subtask said
so, and to name each deliverable by its path. With no deliverable, it is told instead to
state the requested results in the answer and point to no file.
"""

import json

import pytest

from orchestrator.agents.supervisor import (
    FILES_RULE,
    NO_FILE_RULE,
    NOT_COVERED,
    SYNTH_MARKER,
    Supervisor,
    synthesis_manifest,
)
from orchestrator.config import get_settings
from orchestrator.db.repo import InMemoryTaskRepo
from orchestrator.graph.builder import build_graph
from orchestrator.hitl.queue import InMemoryApprovalQueue
from orchestrator.llm.messages import render_messages
from orchestrator.llm.mock import LLMResponse, MockLLMClient
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools.base import InMemoryInvocationStore, InvocationRecord
from orchestrator.tools.defaults import build_default_registry

pytestmark = pytest.mark.anyio


# The last rules version under which a plan may split one specialist's work into
# several subtasks, as a plan here does; version 6 refuses such a plan (test_plan_lint).
# What these tests check is the same under both versions.
SPLIT_RULES = 5


def step(sid: str, specialist: str, description: str, depends_on: list[str] | None = None) -> dict:
    return {"id": sid, "description": description, "specialist": specialist, "tools": [],
            "depends_on": depends_on or []}


PLAN = ExecutionPlan.model_validate({
    "subtasks": [step("s1", "research", "find the documented timeout"),
                 step("s2", "writing", "write the memo", ["s1"])],
    "deliverables": [{"path": "memo.md", "producer": "s2"}, {"path": "appendix.md", "producer": "s2"}],
    "confidence": 0.9,
})


def call(sid: str, tool: str, status: str, **arguments) -> InvocationRecord:
    return InvocationRecord(task_id="t", subtask_id=sid, specialist="x", tool_name=tool, status=status,
                            latency_ms=0.0, arguments=arguments)


INVOCATIONS = [
    call("s1", "rag_ask", "success", question="timeout?"),
    call("s1", "rag_search", "success", query="timeout"),
    call("s1", "rag_search", "success", query="keep-alive"),
    call("s1", "web_search", "rate_limited", query="timeout"),
    call("s2", "file_write", "failure", path="../memo.md", content="x"),
    call("s2", "file_write", "success", path="memo.md", content="x"),
]
RESULTS = {
    "s1": {"status": "completed", "output": "The keep-alive timeout is not documented."},
    "s2": {"status": "completed", "output": "Memo saved as memo.md."},
}


def test_the_manifest_lists_each_subtasks_tool_outcomes_and_report_in_plan_order():
    subtasks, _ = synthesis_manifest(PLAN, RESULTS, INVOCATIONS)

    assert subtasks == (
        "[s1] research: find the documented timeout\n"
        "Tool calls: rag_ask succeeded; rag_search succeeded x2; web_search refused (limit)\n"
        "Reported by research:\nThe keep-alive timeout is not documented.\n\n"
        "[s2] writing: write the memo\n"
        "Tool calls: file_write `../memo.md` failed; file_write `memo.md` succeeded\n"
        "Reported by writing:\nMemo saved as memo.md."
    )


def test_the_manifest_marks_each_deliverable_written_or_not():
    _, deliverables = synthesis_manifest(PLAN, RESULTS, INVOCATIONS)

    assert deliverables == "- `memo.md` (by s2): written\n- `appendix.md` (by s2): NOT written"


def test_a_plan_without_deliverables_and_a_subtask_without_tools():
    plan = ExecutionPlan.model_validate({"subtasks": [step("s1", "writing", "write it")], "confidence": 0.9})

    subtasks, deliverables = synthesis_manifest(plan, {"s1": {"status": "completed", "output": "TEXT"}}, [])

    assert subtasks == "[s1] writing: write it\nTool calls: none\nReported by writing:\nTEXT"
    assert deliverables == "(none)"


def test_a_human_take_over_is_reported_as_the_humans():
    results = {**RESULTS, "s1": {"status": "completed", "output": "HUMAN TEXT", "human_provided": True}}

    subtasks, _ = synthesis_manifest(PLAN, results, [])

    assert "Reported by a human reviewer, who took over:\nHUMAN TEXT" in subtasks


class Capture:
    def __init__(self):
        self.prompts = []

    async def chat(self, agent, messages, **kwargs):
        self.prompts.append(render_messages(messages))
        return LLMResponse(text="FINAL")


async def test_the_prompt_carries_the_manifest_and_the_grounding_rules():
    llm = Capture()

    assert await Supervisor(llm).synthesize("Write a memo on the timeout.", PLAN, RESULTS, INVOCATIONS) == "FINAL"

    (prompt,) = llm.prompts
    subtasks, deliverables = synthesis_manifest(PLAN, RESULTS, INVOCATIONS)
    assert prompt.startswith(f"[user] {SYNTH_MARKER} for the task below from what its subtasks reported.")
    assert "Task: Write a memo on the timeout." in prompt
    assert subtasks in prompt and deliverables in prompt
    assert "Use only what the subtasks reported. Add no fact, number, command, setting, example" in prompt
    assert f'say "{NOT_COVERED}" for that point' in prompt
    assert "Name each file listed above by its exact path" in prompt


# The prompt as it was before plans without files got a rule of their own.
WITH_FILES = """{marker} for the task below from what its subtasks reported.

Task: {request}

What each subtask did and reported:

{subtasks}

Files the task must deliver:
{deliverables}

Write the final answer for the user from the reports above, under these rules:
1. Use only what the subtasks reported. Add no fact, number, command, setting, example
   or recommendation of your own, even one you believe is correct.
2. Where a subtask reports that the documentation does not cover something, or that it
   is not documented, say "{not_covered}" for that point and give no answer to it
   from anywhere else.
3. Name each file listed above by its exact path, and say a file was written only if it
   is marked written.
4. If the reports leave part of the task undone, say which part.

Respond with the final answer only.
"""


async def test_a_plan_with_files_gets_the_prompt_it_had_before():
    llm = Capture()

    await Supervisor(llm).synthesize("Write a memo on the timeout.", PLAN, RESULTS, INVOCATIONS)

    subtasks, deliverables = synthesis_manifest(PLAN, RESULTS, INVOCATIONS)
    assert llm.prompts == ["[user] " + WITH_FILES.format(
        marker=SYNTH_MARKER, request="Write a memo on the timeout.", subtasks=subtasks, deliverables=deliverables,
        not_covered=NOT_COVERED,
    )]


async def test_a_plan_without_files_is_told_to_state_the_results_in_the_answer():
    plan = ExecutionPlan.model_validate({"subtasks": [step("s1", "analysis", "compute the totals")], "confidence": 0.9})
    results = {"s1": {"status": "completed", "output": "Totals: 3, 5 and 8."}}
    llm = Capture()

    await Supervisor(llm).synthesize("Compute the totals per group.", plan, results, [])

    (prompt,) = llm.prompts
    assert "Files the task must deliver:\n(none)\n" in prompt
    assert FILES_RULE not in prompt and NO_FILE_RULE in prompt
    # the grounding rules come first and still bind it
    order = [prompt.index(text) for text in ("\n1. Use only", "\n2. Where a subtask", NO_FILE_RULE, "\n4. If the")]
    assert order == sorted(order)
    assert "never refer the user to a file instead. Rules 1 and 2 still apply." in NO_FILE_RULE
    assert "(values, lists, code) in full, as the reports give it" in NO_FILE_RULE


async def test_the_graphs_synthesis_sees_the_tasks_invocations(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path / "workspaces")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    def fx(name, agent, text="", match=None, tool_calls=None):
        response = {"text": text, **({"tool_calls": tool_calls} if tool_calls else {})}
        payload = {"agent": agent, "response": response, **({"match": match} if match else {})}
        (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    plan = {"subtasks": [{**step("s1", "research", "look it up"), "tools": ["web_search"]},
                         {**step("s2", "writing", "write the memo", ["s1"]), "tools": ["file_write"]}],
            "deliverables": [{"path": "memo.md", "producer": "s2"}], "confidence": 0.9}
    fx("plan", "supervisor", json.dumps(plan), match=["Create an execution plan"])
    fx("synth", "supervisor", "DONE", match=[SYNTH_MARKER])
    fx("research", "research", tool_calls=[{"name": "web_search", "arguments": {"query": "it"}}])
    fx("r_answer", "research", "NOT DOCUMENTED", match=["-> web_search returned"])
    fx("w_a", "writing", match=["write the memo"],
       tool_calls=[{"name": "file_write", "arguments": {"path": "memo.md", "content": "memo"}}])
    fx("w_b", "writing", "MEMO WRITTEN", match=["write the memo", "-> file_write returned"])
    fx("reviewer", "reviewer", json.dumps({"score": 5, "feedback": "fine"}))

    class Recording:
        def __init__(self, inner):
            self.inner, self.prompts = inner, []

        async def chat(self, agent, messages, **kwargs):
            self.prompts.append((agent, render_messages(messages)))
            return await self.inner.chat(agent, messages, **kwargs)

    repo, llm = InMemoryTaskRepo(), Recording(MockLLMClient(fixtures))
    graph = build_graph(llm=llm, registry=build_default_registry(InMemoryInvocationStore()), repo=repo,
                        checkpointer=None, approvals=InMemoryApprovalQueue())
    task_id = repo.create_task("Write a memo to `memo.md`.")
    await graph.ainvoke({"task_id": task_id, "request": "Write a memo to `memo.md`.", "subtask_results": {},
                         "dispatch_log": [],
                         "rules_version": SPLIT_RULES})

    (prompt,) = [text for agent, text in llm.prompts if agent == "supervisor" and SYNTH_MARKER in text]
    assert "[s1] research: look it up\nTool calls: web_search succeeded\nReported by research:\nNOT DOCUMENTED" in prompt
    assert "[s2] writing: write the memo\nTool calls: file_write `memo.md` succeeded\n" in prompt
    assert "- `memo.md` (by s2): written" in prompt
    assert repo.tasks[task_id]["status"] == "completed"
