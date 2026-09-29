"""smoke.py prints each tool's outcome against a running server, tool errors included."""

import pytest

import smoke
from conftest import RAG_URL
from server import build_server

pytestmark = pytest.mark.anyio


async def test_smoke_prints_every_tool(rag_http, capsys):
    await smoke.main(build_server(RAG_URL, http=rag_http))

    out = capsys.readouterr().out
    assert "protocol 2026-07-28" in out
    assert "tools ['ask', 'list_sources', 'search']" in out
    assert "list_sources: 7 sources, 37 chunks" in out
    assert "search 'FERRY-429' (hybrid):" in out
    assert "  1. " in out and "04-error-codes.md  Request errors" in out
    assert "ask (confidence 0.81, $0.000261): FERRY-429 means" in out
    assert "  [1] 04-error-codes.md  Request errors  supported=True" in out


async def test_smoke_prints_tool_errors_instead_of_raising(rag_http, fake_rag, capsys):
    fake_rag.fail = (503, {"detail": "Missing required configuration: OPENAI_API_KEY."})

    await smoke.main(build_server(RAG_URL, http=rag_http))

    out = capsys.readouterr().out
    assert "ask: tool error: " in out
    assert "retrieval service returned 503: Missing required configuration: OPENAI_API_KEY." in out
