"""The SDK import isn't shadowed, and tool results fit the agent loop's budget."""

import json
from pathlib import Path

from tool_models import (
    DEFAULT_TOP_K,
    RESULT_BUDGET_CHARS,
    SNIPPET_CHARS,
    AskResult,
    Citation,
    SearchHit,
    SearchResult,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

# The costliest headings of the Ferry corpus as JSON (quotes and dashes escaped),
# and snippets denser in non-ASCII than any chunk there: json.dumps writes each
# such character as six, and the loop measures the escaped JSON.
HEADINGS = [
    'Sample Corpus — "Ferry" Internal Docs (synthetic)',
    "Why this corpus is shaped the way it is",
    "What happens when you exceed the limit",
    "Ferry — Configuration Reference",
    "Ferry — Architecture & Glossary",
]
SNIPPET = ("Ferry retries a job with backoff — see ferry.retry.* → the DLQ. " * 10)[:SNIPPET_CHARS]


def loop_view(result) -> int:
    """Characters of JSON the agent loop sees: the registry wraps results in {"content": …}."""
    return len(json.dumps({"content": result.model_dump(mode="json")}))


def test_import_mcp_resolves_to_the_sdk_not_this_directory():
    # This directory is named `mcp` too. Given an __init__.py, or with the SDK
    # missing, `import mcp` would resolve here (or to a namespace package).
    import mcp

    assert mcp.__file__ is not None
    sdk_dir = Path(mcp.__file__).resolve().parent
    assert sdk_dir.parent.name == "site-packages"
    assert sdk_dir != PROJECT_DIR
    assert hasattr(mcp, "Client")


def test_search_result_at_the_default_top_k_fits_the_loop_budget():
    hit = SearchHit(
        chunk_id="9d2b7c5e0f1a4836",
        source_file="03-configuration.md",
        section_heading=HEADINGS[0],
        score=1 / 61,
        snippet=SNIPPET,
    )
    result = SearchResult(
        query="How should a client retry after FERRY-429 — Rate Limit Exceeded?",
        mode="hybrid",
        hits=[hit] * DEFAULT_TOP_K,
    )

    assert len(SNIPPET) == SNIPPET_CHARS
    assert loop_view(result) <= RESULT_BUDGET_CHARS


def test_ask_result_with_five_citations_leaves_room_for_the_answer():
    citations = [
        Citation(
            index=n,
            chunk_id="9d2b7c5e0f1a4836",
            source_file="03-configuration.md",
            section_heading=heading,
            supported=True,
        )
        for n, heading in enumerate(HEADINGS, start=1)
    ]
    result = AskResult(
        answer="",
        refused=False,
        confidence=0.8123456789012345,
        citations=citations,
        cost_usd=0.000261,
    )

    # The answer's length is up to the generator; five citations must leave it
    # a thousand characters of the budget.
    assert RESULT_BUDGET_CHARS - loop_view(result) >= 1000
