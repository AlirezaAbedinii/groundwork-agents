"""The capability manifest and the plan lint (planning/capabilities.py), and how planning uses them.

The failing plans are the shapes the planner produced when it was asked only to
decompose and saw no tools: work given to a specialist that does not own the tool
it needs, tool-less research and analysis steps, code written in one subtask and
run in another, and four to eight subtasks that open with research. Each must
fail. A plan with one subtask, or one subtask per owner of the tools it needs,
must pass.
"""

from dataclasses import replace

import pytest

from orchestrator.config import get_settings
from orchestrator.planning.capabilities import lint, manifest, render_manifest, summary
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


def plan(*subtasks: dict) -> ExecutionPlan:
    return ExecutionPlan.model_validate({"subtasks": list(subtasks), "confidence": 0.9})


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
    shape = plan(*(step(f"s{n}", "research", ["web_search"]) for n in range(1, 7)))

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


@pytest.mark.parametrize("shape", [
    pytest.param(plan(step("s1", "analysis", ["db_query", "code_exec", "file_write"], "Rank and save as top.md")),
                 id="one analysis subtask"),
    pytest.param(plan(step("s1", "research", ["rag_search", "file_write"], "Summarise the docs into notes.md")),
                 id="docs to a file in one research subtask"),
    pytest.param(plan(step("s1", "analysis", ["db_query"]), step("s2", "research", ["web_search", "file_write"])),
                 id="one subtask per owner"),
    pytest.param(plan(step("s1", "code", ["code_exec"]), step("s2", "writing", [], depends_on=["s1"])),
                 id="writing from its inputs"),
])
def test_a_minimal_plan_passes(shape):
    assert lint(shape, registry(), 4) == []


def test_the_lint_changes_neither_the_plan_nor_the_registry():
    tools, shape = registry(), plan(step("s1", "research", ["db_query"]))
    before = (shape.model_dump(), render_manifest(manifest(tools)))

    lint(shape, tools, 1)

    assert (shape.model_dump(), render_manifest(manifest(tools))) == before

