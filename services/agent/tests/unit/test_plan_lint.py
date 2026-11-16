"""The capability manifest (planning/capabilities.py): each specialist's tools as this
deployment registers them."""

from dataclasses import replace

from orchestrator.config import get_settings
from orchestrator.planning.capabilities import manifest, render_manifest, summary
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

