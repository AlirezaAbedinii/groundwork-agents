"""The capability manifest and the plan lint (planning/capabilities.py), and how planning uses them.

The failing plans are the shapes the planner produced when it was asked only to
decompose and saw no tools: work given to a specialist that does not own the tool
it needs, tool-less research and analysis steps, code written in one subtask and
run in another, and four to eight subtasks that open with research. Each must
fail. A plan with one subtask, or one subtask per owner of the tools it needs,
must pass, and work one specialist owns every tool of must be one subtask. Files the
request asks to save must be the plan's deliverables, each
written by a subtask that owns and lists file_write; a deliverable must be named in the
request, and a plan without one writes no file.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from orchestrator.config import get_settings
from orchestrator.llm.messages import render_messages
from orchestrator.llm.mock import MockLLMClient
from orchestrator.planning.capabilities import lint, manifest, named_in, render_manifest, requested_files, summary
from orchestrator.planning.decomposer import PLAN_MARKER, PlanValidationError, decompose
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools import api_call, code_exec, db_query, file_io, web_search
from orchestrator.tools.base import InMemoryInvocationStore
from orchestrator.tools.defaults import RAG_MCP_POLICY, RAG_TOOL_PREFIX, build_default_registry
from orchestrator.tools.registry import ToolRegistry

LOCAL = (web_search.SPEC, file_io.READ_SPEC, file_io.WRITE_SPEC, code_exec.SPEC, db_query.SPEC, api_call.SPEC)


def registry(*, rag: bool = True, without: tuple[str, ...] = ()) -> ToolRegistry:
    """The local tools, plus stand-ins for the RAG tools with the agent's MCP policy."""
    tools = ToolRegistry(InMemoryInvocationStore())
    for spec in LOCAL:
        if spec.name not in without:
            tools.register(spec)
    if rag:
        for name, policy in RAG_MCP_POLICY.items():
            tools.register(replace(
                web_search.SPEC, name=RAG_TOOL_PREFIX + name, description=f"The RAG service's {name}. More text.",
                owners=policy.owners, rate_limit=policy.rate_limit,
            ))
    return tools


def step(sid: str, specialist: str, tools: list[str], description: str = "", depends_on=()) -> dict:
    return {"id": sid, "specialist": specialist, "tools": tools,
            "description": description or f"step {sid}", "depends_on": list(depends_on)}


def plan(*subtasks: dict, deliverables: dict[str, str] | None = None) -> ExecutionPlan:
    listed = [{"path": path, "producer": producer} for path, producer in (deliverables or {}).items()]
    return ExecutionPlan.model_validate({"subtasks": list(subtasks), "deliverables": listed, "confidence": 0.9})


# --- the manifest ---------------------------------------------------------------


def test_the_manifest_lists_each_specialists_tools_with_a_summary_and_limit():
    by_specialist = {c.specialist: c.tools for c in manifest(registry())}

    assert list(by_specialist) == ["research", "analysis", "writing", "code"]
    assert [t.name for t in by_specialist["research"]] == [
        "api_call", "file_read", "file_write", "rag_ask", "rag_list_sources", "rag_search", "web_search",
    ]
    assert [t.name for t in by_specialist["analysis"]] == ["code_exec", "db_query", "file_read", "file_write"]
    assert [t.name for t in by_specialist["writing"]] == ["file_read", "file_write"]
    assert [t.name for t in by_specialist["code"]] == ["code_exec", "file_read", "file_write"]
    rag_ask = next(t for t in by_specialist["research"] if t.name == "rag_ask")
    assert (rag_ask.summary, rag_ask.limit) == ("The RAG service's ask", RAG_MCP_POLICY["ask"].rate_limit)


def test_a_summary_is_the_descriptions_first_sentence():
    assert summary("Run a SELECT. Unqualified names resolve to demo. Tables: demo.t(c int)") == "Run a SELECT"
    assert summary("Search the web and return result snippets") == "Search the web and return result snippets"
    assert summary("  Ends with a stop.\n") == "Ends with a stop"


def test_the_rendered_manifest_is_one_row_per_specialist_and_tool():
    table = render_manifest(manifest(registry(rag=False)))

    rows = table.splitlines()
    assert rows[0] == "| Specialist | Tool | What it does | Calls per task |"
    assert "| analysis | db_query | Run a read-only SQL SELECT against the demo database schema | 10 |" in rows
    assert len(rows) == 2 + 4 + 4 + 2 + 3  # header, then research, analysis, writing, code


def test_the_manifest_has_only_the_tools_this_deployment_registered(monkeypatch):
    monkeypatch.setattr(get_settings(), "code_exec_backend", "disabled")
    monkeypatch.setattr(get_settings(), "mcp_rag_url", "")

    table = render_manifest(manifest(build_default_registry(InMemoryInvocationStore())))

    assert "code_exec" not in table
    assert "rag_" not in table
    assert "| code | file_write |" in table  # code keeps the file tools


# --- the lint: recorded shapes fail ----------------------------------------------


def test_a_database_lookup_given_to_research_fails():
    errors = lint(plan(step("s1", "research", ["db_query"], "List the vector databases in the demo database")),
                  registry(), 4)

    assert errors == ["s1: 'db_query' belongs to analysis, not research; "
                      "give this subtask to a specialist that owns all of its tools."]


@pytest.mark.parametrize("specialist", ["research", "analysis", "code"])
def test_a_tool_bound_subtask_without_tools_fails(specialist):
    (error,) = lint(plan(step("s1", specialist, [], "Analyze the gathered documentation")), registry(), 4)

    assert error.startswith(f"s1: list the tools this {specialist} subtask will call ({specialist} owns: ")


def test_a_presupposing_plan_with_a_tool_less_analysis_step_fails():
    shape = plan(
        step("s1", "research", ["rag_search"], "Find the documented limit"),
        step("s2", "analysis", [], "Analyze the limit's implications", ["s1"]),
        step("s3", "writing", [], "Write the answer", ["s2"]),
    )

    assert [e.split(":")[0] for e in lint(shape, registry(), 4)] == ["s2"]


def test_code_written_in_one_subtask_and_run_in_another_after_research_fails_on_size():
    shape = plan(
        step("s1", "research", ["web_search"], "Research the input format"),
        step("s2", "analysis", ["code_exec"], "Analyze the requirements", ["s1"]),
        step("s3", "code", ["code_exec"], "Write the function", ["s2"]),
        step("s4", "code", ["code_exec"], "Run the function on the provided inputs", ["s3"]),
        step("s5", "writing", [], "Report the results", ["s4"]),
    )

    (error,) = lint(shape, registry(), 4)
    assert error.startswith("The plan has 5 subtasks; at most 4 are allowed.")


def test_a_research_first_plan_of_six_steps_fails_on_size():
    shape = plan(step("s1", "research", ["web_search"]), *(step(f"s{n}", "analysis", ["db_query"]) for n in range(2, 7)))

    assert lint(shape, registry(), 4)[0].startswith("The plan has 6 subtasks")
    assert lint(shape, registry(), 6) == []  # the cap is a setting


def test_a_tool_this_deployment_does_not_have_fails():
    errors = lint(plan(step("s1", "code", ["code_exec"]), step("s2", "research", ["teleport"])),
                  registry(without=("code_exec",)), 4)

    assert errors[0].startswith("s1: there is no tool 'code_exec' in this deployment (available: api_call, ")
    assert errors[1].startswith("s2: there is no tool 'teleport' in this deployment")


def test_every_problem_is_reported_once():
    shape = plan(step("s1", "writing", ["db_query", "db_query", "web_search"]), step("s2", "analysis", []))

    errors = lint(shape, registry(), 4)
    assert len(errors) == 3
    assert errors[0].startswith("s1: 'db_query' belongs to analysis, not writing")
    assert errors[1].startswith("s1: 'web_search' belongs to research, not writing")
    assert errors[2].startswith("s2: list the tools this analysis subtask will call")


# --- the lint: minimal plans pass -----------------------------------------------


@pytest.mark.parametrize(("shape", "request_text"), [
    pytest.param(plan(step("s1", "analysis", ["db_query", "code_exec", "file_write"], "Rank and save as top.md"),
                      deliverables={"top.md": "s1"}),
                 "Rank the rows and save the ranking as `top.md`.", id="one analysis subtask"),
    pytest.param(plan(step("s1", "research", ["rag_search", "file_write"], "Summarise the docs into notes.md"),
                      deliverables={"notes.md": "s1"}),
                 "Summarise the docs into notes.md.", id="docs to a file in one research subtask"),
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "research", ["web_search"])),
                 "", id="one subtask per owner"),
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "research", ["web_search"]),
                      step("s3", "writing", [], depends_on=["s1", "s2"])),
                 "", id="writing from two specialists' outputs"),
])
def test_a_minimal_plan_passes(shape, request_text):
    assert lint(shape, registry(), 4, request_text) == []


# --- the lint: one specialist's work is one subtask ------------------------------------


def split(named: str, tools: str, deliverable: bool = False) -> str:
    """The message for a plan that splits work ``named`` owns every listed tool of."""
    files = " and writes every deliverable" if deliverable else ""
    return (
        f"The plan splits work that one specialist can do: {named} owns every tool its subtasks list ({tools}). "
        f"Plan exactly one {named} subtask that lists them all and does the whole task{files}; add a subtask only "
        "for work that needs another specialist's tools."
    )


@pytest.mark.parametrize(("shape", "message"), [
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "analysis", ["code_exec"], depends_on=["s1"])),
                 split("analysis", "db_query, code_exec"), id="one specialist in two subtasks"),
    pytest.param(plan(step("s1", "code", ["code_exec"]), step("s2", "analysis", ["code_exec"], depends_on=["s1"])),
                 split("code", "code_exec"), id="two specialists that both own the tool"),
    pytest.param(plan(step("s1", "research", ["rag_search"]), step("s2", "writing", ["file_write"], depends_on=["s1"]),
                      deliverables={"notes.md": "s2"}),
                 split("research", "rag_search, file_write", deliverable=True), id="a writer for a file research can write"),
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "writing", [], depends_on=["s1"])),
                 split("analysis", "db_query"), id="a writing subtask with no tools adds none"),
    pytest.param(plan(step("s1", "writing", ["file_read"]), step("s2", "code", ["code_exec"], depends_on=["s1"])),
                 split("code", "file_read, code_exec"), id="named: the first in the plan that owns every tool"),
])
def test_a_plan_that_splits_work_one_specialist_owns_fails(shape, message):
    assert lint(shape, registry(), 4, "Put the notes in `notes.md`." if shape.deliverables else "") == [message]


def test_a_plan_whose_subtasks_list_no_tools_is_one_subtask():
    shape = plan(step("s1", "writing", []), step("s2", "writing", [], depends_on=["s1"]))

    assert lint(shape, registry(), 4) == [
        "The plan splits work that one specialist can do: its subtasks list no tools. Plan exactly one writing "
        "subtask that does the whole task; add a subtask only for work that needs another specialist's tools."
    ]


def test_a_misassigned_split_names_the_specialist_that_owns_every_tool():
    shape = plan(step("s1", "research", ["db_query"]), step("s2", "analysis", ["code_exec"], depends_on=["s1"]))

    assert lint(shape, registry(), 4) == [
        "s1: 'db_query' belongs to analysis, not research; give this subtask to a specialist that owns all of its "
        "tools.",
        split("analysis", "db_query, code_exec"),
    ]


@pytest.mark.parametrize("specialist", ["research", "analysis", "code"])
def test_the_check_stands_aside_while_a_tool_bound_subtask_lists_no_tools(specialist):
    shape = plan(step("s1", "analysis", ["db_query"]), step("s2", specialist, [], depends_on=["s1"]))

    (error,) = lint(shape, registry(), 4)
    assert error.startswith(f"s2: list the tools this {specialist} subtask will call")


@pytest.mark.parametrize("shape", [
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "research", ["web_search"])),
                 id="two specialists' tools"),
    pytest.param(plan(step("s1", "analysis", ["db_query", "code_exec"])), id="one subtask"),
    pytest.param(plan(step("s1", "code", ["code_exec"]), step("s2", "research", ["teleport"])),
                 id="a tool nobody owns"),
])
def test_a_plan_no_single_specialist_can_carry_out_is_not_merged(shape):
    assert not any(e.startswith("The plan splits work") for e in lint(shape, registry(), 4))


def test_without_the_switch_one_specialists_work_may_be_split():
    shape = plan(step("s1", "analysis", ["db_query"]), step("s2", "analysis", ["code_exec"], depends_on=["s1"]))

    assert lint(shape, registry(), 4, one_specialist=False) == []


def test_the_lint_changes_neither_the_plan_nor_the_registry():
    tools, shape = registry(), plan(step("s1", "research", ["db_query"]))
    before = (shape.model_dump(), render_manifest(manifest(tools)))

    lint(shape, tools, 1)

    assert (shape.model_dump(), render_manifest(manifest(tools))) == before


# --- planning with the lint ----------------------------------------------------

WRONG_OWNER = plan(step("s1", "research", ["db_query"], "Rank the databases by stars"))
RIGHT_OWNER = plan(step("s1", "analysis", ["db_query"], "Rank the databases by stars"))


class Recording:
    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def chat(self, agent, messages, **kwargs):
        self.prompts.append(render_messages(messages))
        return await self.inner.chat(agent, messages, **kwargs)


def fixtures(tmp_path, first: ExecutionPlan, retry: ExecutionPlan) -> Recording:
    for name, chosen, match in (("plan", first, [PLAN_MARKER]),
                                ("plan_retry", retry, [PLAN_MARKER, "Your previous plan was invalid"])):
        payload = {"agent": "supervisor", "match": match, "response": {"text": chosen.model_dump_json()}}
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return Recording(MockLLMClient(tmp_path))


@pytest.mark.anyio
async def test_the_prompt_carries_the_manifest_and_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_plan_subtasks", 3)
    llm = fixtures(tmp_path, RIGHT_OWNER, RIGHT_OWNER)
    tools = registry(without=("code_exec",))

    await decompose(llm, "rank the databases", tools)

    (prompt,) = llm.prompts
    assert render_manifest(manifest(tools)) in prompt
    assert "code_exec" not in prompt
    assert "never more than 3" in prompt


@pytest.mark.anyio
async def test_a_plan_that_fails_the_lint_is_retried_once_with_the_errors(tmp_path):
    llm = fixtures(tmp_path, WRONG_OWNER, RIGHT_OWNER)

    planned = await decompose(llm, "rank the databases", registry())

    assert planned.plan == RIGHT_OWNER
    expected = lint(WRONG_OWNER, registry(), 4)
    assert planned.lint_errors == expected  # what the first plan was rejected for
    first, retry = llm.prompts
    assert expected[0] not in first
    assert f"Validation error:\n{expected[0]}" in retry


@pytest.mark.anyio
async def test_a_plan_that_fails_the_lint_twice_fails_planning(tmp_path):
    llm = fixtures(tmp_path, WRONG_OWNER, WRONG_OWNER)

    with pytest.raises(PlanValidationError, match="^Plan invalid after retry: s1: 'db_query' belongs") as raised:
        await decompose(llm, "rank the databases", registry())

    assert raised.value.lint_errors == lint(WRONG_OWNER, registry(), 4) * 2
    assert len(llm.prompts) == 2


@pytest.mark.anyio
async def test_with_the_lint_off_its_findings_are_reported_but_reject_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "plan_lint", "off")
    llm = fixtures(tmp_path, WRONG_OWNER, RIGHT_OWNER)

    planned = await decompose(llm, "rank the databases", registry())

    assert planned.plan == WRONG_OWNER
    assert planned.lint_errors == lint(WRONG_OWNER, registry(), 4)
    assert len(llm.prompts) == 1


SPLIT = plan(step("s1", "analysis", ["db_query"], "Fetch the rows"),
             step("s2", "analysis", ["code_exec"], "Compute the ranking", ["s1"]))
WHOLE = plan(step("s1", "analysis", ["db_query", "code_exec"], "Fetch the rows and compute the ranking"))


@pytest.mark.anyio
async def test_a_split_plan_is_retried_once_with_the_specialist_named(tmp_path):
    llm = fixtures(tmp_path, SPLIT, WHOLE)

    planned = await decompose(llm, "rank the rows", registry())

    assert planned.plan == WHOLE
    first, retry = llm.prompts
    assert "2. If one specialist owns every tool the task needs, plan exactly one subtask for it." in first
    assert f"Validation error:\n{split('analysis', 'db_query, code_exec')}" in retry


@pytest.mark.anyio
async def test_a_split_plan_under_rules_before_the_check_is_kept(tmp_path):
    llm = fixtures(tmp_path, SPLIT, WHOLE)

    planned = await decompose(llm, "rank the rows", registry(), one_specialist_one_subtask=False)

    assert planned.plan == SPLIT and planned.lint_errors == []
    assert len(llm.prompts) == 1


# --- deliverables ------------------------------------------------------------------

AGENT_TASKS = Path(__file__).resolve().parents[4] / "eval" / "tasks" / "agent_tasks.jsonl"


def required_files(task: dict) -> list[str]:
    """The paths a task's must_call requires file_write to write, as the harness reads them."""
    return [
        entry["args"]["path"]
        for entry in task["must_call"]
        if "file_write" in ([entry["tool"]] if isinstance(entry["tool"], str) else entry["tool"])
        and "path" in entry.get("args", {})
    ]


def test_the_request_scan_finds_each_evaluation_tasks_required_file_or_nothing():
    tasks = [json.loads(line) for line in AGENT_TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert tasks
    for task in tasks:
        found, required = requested_files(task["request"]), required_files(task)
        assert found in ([], required), task["id"]  # never a file the task does not require
        if required and "save" in task["request"].lower():
            assert found == required, task["id"]  # a request that says save is not missed


@pytest.mark.parametrize(("request_text", "expected"), [
    ("Save the note as `notes.md`.", ["notes.md"]),
    ("Write a memo, saved as `out/memo.md`, then stop.", ["out/memo.md"]),
    ("save `report.txt` when done", ["report.txt"]),
    ("Saving the results to `results.csv` is enough.", ["results.csv"]),
    ("Save it as `a.md`; also save the table as `b.csv`. Save it as `a.md` again.", ["a.md", "b.csv"]),
    ("Read `requirements.txt` and save the plan as `plan.md`.", ["plan.md"]),
])
def test_the_request_scan_finds_a_back_quoted_name_the_request_asks_to_save(request_text, expected):
    assert requested_files(request_text) == expected


@pytest.mark.parametrize("request_text", [
    "Write a comparison memo saved as memo.md.",  # not back-quoted
    "Migrate files such as `pyproject.toml` and `poetry.lock`.",  # mentioned, not saved
    "Put the explanation in `notes.md`.",  # no save: missed, never invented
    "Save time: explain how `BaseModel.model_dump_json` differs.",  # not a file name
    "Save the summary. Then read `config.yaml`.",  # another sentence
])
def test_the_request_scan_ignores_a_name_it_is_not_asked_to_save(request_text):
    assert requested_files(request_text) == []


def test_a_file_the_request_asks_to_save_must_be_a_deliverable():
    request = "Summarise the docs and save the summary as `summary.md`."
    unlisted = plan(step("s1", "research", ["rag_search", "file_write"]))
    listed = plan(step("s1", "research", ["rag_search", "file_write"]), deliverables={"summary.md": "s1"})

    assert lint(unlisted, registry(), 4, request) == [
        "The task asks to save `summary.md`; list it in deliverables with exactly that path and the id "
        "of the subtask that writes it with file_write."
    ]
    assert lint(listed, registry(), 4, request) == []


@pytest.mark.parametrize("request_text", [
    "Write a comparison memo saved as memo.md.",  # not back-quoted
    "Put the comparison in `memo.md`.",  # no "save"
    "Write `memo.md`: two paragraphs on each option.",
])
def test_a_deliverable_the_request_names_is_allowed_where_the_scan_finds_nothing(request_text):
    shape = plan(step("s1", "writing", ["file_write"]), deliverables={"memo.md": "s1"})

    assert requested_files(request_text) == []
    assert lint(shape, registry(), 4, request_text) == []


UNNAMED = (
    "The task does not name `{}`; remove it from deliverables. List only files the task names, with "
    "each path exactly as the task gives it."
)
NO_FILE = (
    "{}: the plan delivers no file the task names, so no subtask writes one; remove file_write from its "
    "tools and have it report its results instead."
)


def test_a_deliverable_the_request_does_not_name_is_refused_and_so_is_its_file_write():
    request = "Count the rows per region and list the three largest."
    shape = plan(step("s1", "analysis", ["db_query", "file_write"]), deliverables={"counts.txt": "s1"})

    # both at once, so a single retry can drop the file and the tool together
    assert lint(shape, registry(), 4, request) == [UNNAMED.format("counts.txt"), NO_FILE.format("s1")]


def test_only_the_unnamed_deliverable_is_refused_beside_a_named_one():
    request = "Put the summary in `summary.md`."
    shape = plan(step("s1", "research", ["rag_search", "file_write"]), step("s2", "analysis", ["db_query", "file_write"]),
                 deliverables={"summary.md": "s1", "extra.csv": "s2"})

    assert lint(shape, registry(), 4, request) == [UNNAMED.format("extra.csv")]


def test_file_write_in_a_plan_without_deliverables_is_refused():
    shape = plan(step("s1", "research", ["rag_search", "file_write"]), step("s2", "code", ["code_exec", "file_write"]),
                 step("s3", "analysis", ["db_query"]))

    assert lint(shape, registry(), 4, "Explain the option and give a worked example.") == [
        NO_FILE.format("s1"), NO_FILE.format("s2"),
    ]


@pytest.mark.parametrize(("request_text", "path"), [
    ("Save it as `notes.md`.", "notes.md"),
    ("Put it in notes.md.", "notes.md"),
    ("notes.md: one step per line", "notes.md"),
    ('Write the steps (to "notes.md").', "notes.md"),
    ("Save the steps as `docs/notes.md`.", "docs/notes.md"),
])
def test_a_whole_name_in_the_request_names_the_file(request_text, path):
    assert named_in(request_text, path)


@pytest.mark.parametrize("request_text", [
    "Update `old_notes.md` with the new steps.",
    "Save the steps as `docs/notes.md`.",
    "Keep a copy in notes.md.bak.",
    "Write to `notes.mdx`.",
    "Write the notes in Markdown.",
    "Save them as `Notes.md`.",  # the path is matched exactly, case included
])
def test_a_name_inside_a_longer_one_or_in_another_form_does_not_name_the_file(request_text):
    assert not named_in(request_text, "notes.md")


def test_without_the_switch_unnamed_deliverables_and_file_write_are_allowed():
    unnamed = plan(step("s1", "analysis", ["db_query", "file_write"]), deliverables={"counts.txt": "s1"})
    bare = plan(step("s1", "analysis", ["db_query", "file_write"]))

    for shape in (unnamed, bare):
        assert lint(shape, registry(), 4, "List the three largest regions.", only_requested=False) == []


def test_a_deliverables_producer_must_list_file_write():
    shape = plan(step("s1", "research", ["rag_search"]), deliverables={"notes.md": "s1"})

    assert lint(shape, registry(), 4, "Put the notes in `notes.md`.") == [
        "s1: it is to write `notes.md`; list file_write in its tools."
    ]


def test_a_deliverables_producer_must_own_file_write():
    shape = plan(step("s1", "research", ["rag_search"]), deliverables={"notes.md": "s1"})

    (error,) = lint(shape, registry(without=("file_write",)), 4, "Put the notes in `notes.md`.")
    assert error == (
        "s1: it is to write `notes.md`, but research cannot call file_write (owned here by: nobody); "
        "give the file to a subtask whose specialist owns it."
    )


def test_without_the_deliverables_switch_deliverables_are_not_linted():
    shape = plan(step("s1", "research", ["rag_search"]), deliverables={"notes.md": "s1"})

    assert lint(shape, registry(), 4, "Save it as `other.md`.", deliverables=False) == []


@pytest.mark.anyio
async def test_the_planner_is_asked_for_deliverables_and_retried_without_them(tmp_path):
    request = "Rank the databases by stars and save the ranking as `ranking.md`."
    unlisted = plan(step("s1", "analysis", ["db_query", "file_write"], "Rank and save as ranking.md"))
    listed = plan(step("s1", "analysis", ["db_query", "file_write"], "Rank and save as ranking.md"),
                  deliverables={"ranking.md": "s1"})
    llm = fixtures(tmp_path, unlisted, listed)

    planned = await decompose(llm, request, registry())

    assert planned.plan == listed
    first, retry = llm.prompts
    assert "In deliverables, list each file the task names and asks to be saved or written,\n   and no other" in first
    assert "The task asks to save `ranking.md`" in retry


@pytest.mark.anyio
async def test_a_plan_with_a_file_the_request_does_not_name_is_retried_without_it(tmp_path):
    request = "Count the rows per region and list the three largest."
    extra = plan(step("s1", "analysis", ["db_query", "file_write"], "Count and save"), deliverables={"counts.txt": "s1"})
    clean = plan(step("s1", "analysis", ["db_query"], "Count and report the three largest"))
    llm = fixtures(tmp_path, extra, clean)

    planned = await decompose(llm, request, registry())

    assert planned.plan == clean
    first, retry = llm.prompts
    assert "leave deliverables empty and list file_write in no subtask's tools" in first
    assert f"Validation error:\n{UNNAMED.format('counts.txt')}\n{NO_FILE.format('s1')}" in retry


@pytest.mark.anyio
async def test_a_plan_under_rules_before_named_files_keeps_its_unnamed_file(tmp_path):
    extra = plan(step("s1", "analysis", ["db_query", "file_write"]), deliverables={"counts.txt": "s1"})
    llm = fixtures(tmp_path, extra, extra)

    planned = await decompose(llm, "List the three largest regions.", registry(), only_requested_files=False)

    assert planned.plan == extra and planned.lint_errors == []
    assert len(llm.prompts) == 1


@pytest.mark.anyio
async def test_a_plan_under_rules_without_deliverables_is_not_retried_for_them(tmp_path):
    request = "Rank the databases by stars and save the ranking as `ranking.md`."
    unlisted = plan(step("s1", "analysis", ["db_query", "file_write"]))
    llm = fixtures(tmp_path, unlisted, unlisted)

    planned = await decompose(llm, request, registry(), check_deliverables=False)

    assert planned.plan == unlisted and planned.lint_errors == []
    assert len(llm.prompts) == 1
