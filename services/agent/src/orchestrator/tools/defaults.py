"""Default tool registry wiring: the local tools, plus the RAG service's tools over MCP."""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import replace

from orchestrator.config import get_settings
from orchestrator.db.session import get_engine
from orchestrator.tools import api_call, code_exec, db_query, file_io, web_search
from orchestrator.tools.base import InvocationStore
from orchestrator.tools.mcp_models import McpToolPolicy, McpUnavailableError
from orchestrator.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# The retrieval tools the RAG stack's MCP server offers, as the agent allows them.
# Only tools listed here are registered (deny by default), under the rag_ prefix;
# the server's own annotations decide nothing. Search hits and an answer with its
# citations are the evidence the specialist works from, so they get twice the
# default result budget: a search with a larger top_k still arrives as whole hits.
RAG_TOOL_PREFIX = "rag_"
RAG_RESULT_BUDGET_CHARS = 4000
RAG_MCP_POLICY: dict[str, McpToolPolicy] = {
    "search": McpToolPolicy(owners=frozenset({"research"}), rate_limit=10, result_budget_chars=RAG_RESULT_BUDGET_CHARS),
    "ask": McpToolPolicy(  # spends the RAG service's LLM budget
        owners=frozenset({"research"}), rate_limit=5, result_budget_chars=RAG_RESULT_BUDGET_CHARS
    ),
    "list_sources": McpToolPolicy(owners=frozenset({"research"}), rate_limit=3),
}

# Registries that already hold the RAG tools. Discovery awaits, so it runs outside
# the lock; the lock covers only the quick, sync registration, so two first runs
# in different threads or loops can't both register the same tools.
_with_rag_tools: weakref.WeakSet[ToolRegistry] = weakref.WeakSet()
_register_lock = threading.Lock()


def build_default_registry(store: InvocationStore) -> ToolRegistry:
    """The local tools. ``db_query``'s description lists the demo schema's tables, read
    from the database now (the static text if it can't be read). A tool with nothing
    behind it is left out: ``code_exec`` without a sandbox, and ``web_search`` without a
    Tavily key (MOCK_LLM keeps it, answered with canned results)."""
    settings = get_settings()
    registry = ToolRegistry(store)
    for spec in (
        web_search.SPEC,
        file_io.READ_SPEC,
        file_io.WRITE_SPEC,
        code_exec.SPEC,
        db_query.SPEC,
        api_call.SPEC,
    ):
        if spec is code_exec.SPEC and settings.code_exec_backend == "disabled":
            continue  # no sandbox to run model-written code in
        if spec is web_search.SPEC and not (settings.tavily_api_key or settings.mock_llm):
            continue  # no search provider to answer it
        if spec is db_query.SPEC:
            spec = replace(spec, description=db_query.describe(get_engine()))
        registry.register(spec)
    return registry


async def build_registry(store: InvocationStore) -> ToolRegistry:
    """A new registry with the local tools, plus the RAG tools when MCP_RAG_URL is set."""
    registry = build_default_registry(store)
    await add_rag_tools(registry)
    return registry


async def add_rag_tools(registry: ToolRegistry) -> None:
    """Add the RAG MCP tools to ``registry`` if they aren't there yet.

    A no-op when MCP_RAG_URL is empty or the tools are already registered, so the
    runner calls it before every run: the first run that reaches the server adds
    them, and while the server is down each run goes on with the local tools and
    the next one tries again.
    """
    settings = get_settings()
    if not settings.mcp_rag_url or registry in _with_rag_tools:
        return
    # Imported here: the MCP SDK takes most of a second to import, and runs without MCP don't need it.
    from orchestrator.tools.mcp_client import discover_tools

    try:
        specs = await discover_tools(
            settings.mcp_rag_url, RAG_MCP_POLICY, prefix=RAG_TOOL_PREFIX, timeout_s=settings.mcp_call_timeout_s
        )
    except McpUnavailableError as exc:
        logger.warning("RAG tools unavailable, running without them until a later run reaches the server: %s", exc)
        return
    with _register_lock:
        if registry in _with_rag_tools:
            return  # a concurrent first run registered them
        for spec in specs:
            registry.register(spec)
        _with_rag_tools.add(registry)
    logger.info("Registered RAG tools from %s: %s", settings.mcp_rag_url, ", ".join(spec.name for spec in specs))
