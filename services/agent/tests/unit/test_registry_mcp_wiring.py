"""The RAG tools join a registry over MCP on the first run that reaches the server.

The runner keeps one registry per process and calls add_rag_tools before every
run; build_registry (replay, fixture recording) makes a fresh one. Discovery
reaches the in-process stub from conftest.py instead of a URL.
"""

import asyncio
import logging
import threading

import pytest

from orchestrator.config import get_settings
from orchestrator.llm.router import SPECIALISTS
from orchestrator.tools import mcp_client
from orchestrator.tools.base import InMemoryInvocationStore
from orchestrator.tools.defaults import add_rag_tools, build_default_registry, build_registry

RAG = {"rag_ask", "rag_list_sources", "rag_search"}
UNREACHABLE = "http://127.0.0.1:9/mcp"  # nothing listens on the discard port


def names(registry) -> set[str]:
    return {spec.name for specialist in SPECIALISTS for spec in registry.tools_for(specialist)}


LOCAL = names(build_default_registry(InMemoryInvocationStore()))


def reach_stub(monkeypatch, stub) -> None:
    """Discovery goes to the in-process stub, whatever MCP_RAG_URL says."""
    discover = mcp_client.discover_tools

    async def via_stub(target, policy, **options):
        return await discover(stub.server, policy, **options)

    monkeypatch.setattr(mcp_client, "discover_tools", via_stub)


@pytest.fixture()
def rag_up(monkeypatch, rag_stub):
    monkeypatch.setattr(get_settings(), "mcp_rag_url", "http://rag-mcp.test/mcp")
    reach_stub(monkeypatch, rag_stub)
    return rag_stub


def test_without_an_mcp_url_only_the_local_tools_are_registered_and_nothing_is_dialled(monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_rag_url", "")

    def must_not_run(*args, **options):
        raise AssertionError("discovery ran without MCP_RAG_URL")

    monkeypatch.setattr(mcp_client, "discover_tools", must_not_run)

    registry = asyncio.run(build_registry(InMemoryInvocationStore()))

    assert names(registry) == LOCAL
    assert len(LOCAL) == 6


def test_with_the_server_up_the_rag_tools_join_under_the_agents_policy(rag_up):
    registry = asyncio.run(build_registry(InMemoryInvocationStore()))

    assert names(registry) == LOCAL | RAG
    rag = {name: registry.get(name) for name in RAG}
    assert {name: spec.rate_limit for name, spec in rag.items()} == {"rag_search": 10, "rag_ask": 5, "rag_list_sources": 3}
    assert all(spec.owners == frozenset({"research"}) and not spec.sensitive for spec in rag.values())


def test_a_registry_that_has_the_tools_doesnt_discover_again(rag_up, monkeypatch):
    registry = build_default_registry(InMemoryInvocationStore())
    asyncio.run(add_rag_tools(registry))

    def must_not_run(*args, **options):
        raise AssertionError("discovered again")

    monkeypatch.setattr(mcp_client, "discover_tools", must_not_run)
    asyncio.run(add_rag_tools(registry))  # the next run

    assert names(registry) == LOCAL | RAG


def test_two_first_runs_at_once_register_the_tools_once(rag_up, monkeypatch):
    registry = build_default_registry(InMemoryInvocationStore())
    discover = mcp_client.discover_tools
    both_discovered = threading.Barrier(2, timeout=10)

    async def discover_then_wait(*args, **options):
        specs = await discover(*args, **options)
        both_discovered.wait()  # neither run registers before both have discovered
        return specs

    monkeypatch.setattr(mcp_client, "discover_tools", discover_then_wait)
    errors: list[BaseException] = []

    def first_run() -> None:  # like two Celery worker threads, each with its own loop
        try:
            asyncio.run(add_rag_tools(registry))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=first_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)

    assert errors == []  # without the lock and flag, the second run's register() raises ValueError
    assert names(registry) == LOCAL | RAG


def test_an_unreachable_server_is_retried_by_a_later_run(monkeypatch, rag_stub, caplog):
    monkeypatch.setattr(get_settings(), "mcp_rag_url", UNREACHABLE)
    registry = build_default_registry(InMemoryInvocationStore())

    asyncio.run(add_rag_tools(registry))

    assert names(registry) == LOCAL  # this run goes on with the local tools
    assert any(r.levelno == logging.WARNING and "RAG tools unavailable" in r.getMessage() for r in caplog.records)

    reach_stub(monkeypatch, rag_stub)  # the server is up by the next run
    asyncio.run(add_rag_tools(registry))

    assert names(registry) == LOCAL | RAG
