"""The SDK import isn't shadowed, and tool results are sized for the agent loop.

``fit_hits`` and ``fit_sources`` take the RAG API's dicts and return what the
tools send. Sizes are measured the way the agent loop measures them: the JSON of
the result as ``json.dumps`` writes it, every non-ASCII character as six.
"""

import json
from pathlib import Path

import pytest

from tool_models import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    QUERY_ECHO_CHARS,
    RESULT_BUDGET_CHARS,
    AskResult,
    Citation,
    SearchResult,
    SourcesResult,
    excerpt,
    fit_hits,
    fit_sources,
    query_terms,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
# The agent loop's budget for these tools (services/agent/.../tools/defaults.py:
# RAG_RESULT_BUDGET_CHARS); it measures the result wrapped as {"content": …}.
AGENT_BUDGET_CHARS = 4000

# The costliest headings of the Ferry corpus as JSON (quotes and dashes escaped).
HEADINGS = [
    'Sample Corpus — "Ferry" Internal Docs (synthetic)',
    "Why this corpus is shaped the way it is",
    "What happens when you exceed the limit",
    "Ferry — Configuration Reference",
    "Ferry — Architecture & Glossary",
]
# The worst case: chunks at the RAG's chunk_size (800) denser in non-ASCII than any
# chunk of either corpus, the longest source path of the public corpus, scores at
# full float precision, and a query longer than any echoed whole.
DENSE_LINE = "Ferry retries a job with backoff — see ferry.retry.* → the DLQ “now”.\n"
WORST_TEXT = (DENSE_LINE * 20)[:800]
WORST_PATH = "fastapi/advanced/path-operation-advanced-configuration-and-more-options.md"
WORST_QUERY = "How should a client retry after FERRY-429 — Rate Limit Exceeded? " * 6


def hit(n: int, text: str, heading: str | None = None, path: str = "04-error-codes.md") -> dict:
    """One hit as the RAG API sends it."""
    return {
        "chunk_id": f"a3f9c2e17b4d8e{n:02d}",
        "text": text,
        "score": 1 / (61 + n),
        "source_file": path,
        "section_heading": heading if heading is not None else "",
    }


def worst_hits(k: int) -> list[dict]:
    return [hit(n, WORST_TEXT, HEADINGS[n % len(HEADINGS)], WORST_PATH) for n in range(k)]


def size(result: dict) -> int:
    return len(json.dumps(result))


def loop_view(result: dict) -> int:
    """Characters of JSON the agent loop sees: the registry wraps results in {"content": …}."""
    return len(json.dumps({"content": result}))


def test_import_mcp_resolves_to_the_sdk_not_this_directory():
    # This directory is named `mcp` too. Given an __init__.py, or with the SDK
    # missing, `import mcp` would resolve here (or to a namespace package).
    import mcp

    assert mcp.__file__ is not None
    sdk_dir = Path(mcp.__file__).resolve().parent
    assert sdk_dir.parent.name == "site-packages"
    assert sdk_dir != PROJECT_DIR
    assert hasattr(mcp, "Client")


# --- search --------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", range(1, MAX_TOP_K + 1))
def test_a_search_result_never_exceeds_the_budget(top_k):
    fitted = fit_hits(WORST_QUERY, "hybrid", worst_hits(top_k))

    # What the tool sends is the model's JSON, so measure that, not the dict.
    sent = SearchResult.model_validate(fitted).model_dump(mode="json")
    assert sent == fitted
    assert size(sent) <= RESULT_BUDGET_CHARS
    assert loop_view(sent) <= AGENT_BUDGET_CHARS
    assert len(sent["hits"]) == top_k  # every hit keeps its place
    assert all(h["text"] for h in sent["hits"][:3])


@pytest.mark.parametrize("top_k", range(1, MAX_TOP_K + 1))
def test_the_budget_holds_for_chunks_far_longer_than_the_rags(top_k):
    hits = [hit(n, WORST_TEXT * 4, HEADINGS[0], WORST_PATH) for n in range(top_k)]

    assert size(fit_hits(WORST_QUERY, "dense", hits)) <= RESULT_BUDGET_CHARS


def test_hits_are_whole_in_rank_order_while_they_fit_then_excerpts():
    texts = [f"Passage {n}: " + "the retry policy backs off. " * 25 for n in range(6)]

    fitted = fit_hits("retry policy", "hybrid", [hit(n, t) for n, t in enumerate(texts)])

    complete = [h["complete"] for h in fitted["hits"]]
    whole = complete.index(False)
    assert whole >= 3 and not any(complete[whole:])  # whole first, then only excerpts
    for h, text in zip(fitted["hits"], texts, strict=True):
        assert (h["text"] == text) is h["complete"]
        assert text.startswith(h["text"]) or h["text"] in text
    assert size(fitted) <= RESULT_BUDGET_CHARS
    assert RESULT_BUDGET_CHARS - size(fitted) < 8  # the excerpts use what is left


def test_the_default_top_k_sends_the_rags_chunks_whole():
    # 800 characters is the RAG's chunk_size; the public corpus's median chunk is 600.
    texts = [(f"Line {n} of a passage about retries and backoff.\n" * 20)[:800] for n in range(3)]
    hits = [hit(n, text, HEADINGS[n]) for n, text in enumerate(texts)]

    fitted = fit_hits("How do retries back off?", "hybrid", hits)

    assert DEFAULT_TOP_K == 3
    assert [h["complete"] for h in fitted["hits"]] == [True, True, True]
    assert [h["text"] for h in fitted["hits"]] == texts


def test_a_short_hit_after_a_long_one_can_still_be_whole():
    long_text, short = "word " * 2000, "FERRY-429 means you exceeded the quota."

    fitted = fit_hits("FERRY-429", "dense", [hit(0, long_text), hit(1, short)])

    assert [h["complete"] for h in fitted["hits"]] == [False, True]
    assert fitted["hits"][1]["text"] == short


def test_the_metadata_passes_through_and_an_empty_heading_is_null():
    fitted = fit_hits("FERRY-429", "hybrid", [hit(0, "FERRY-429", "Request errors"), hit(1, "x")])

    first, second = fitted["hits"]
    assert (first["chunk_id"], first["source_file"], first["section_heading"]) == (
        "a3f9c2e17b4d8e00",
        "04-error-codes.md",
        "Request errors",
    )
    assert first["score"] == 1 / 61  # full precision, unchanged
    assert second["section_heading"] is None
    assert (fitted["query"], fitted["mode"]) == ("FERRY-429", "hybrid")


def test_a_long_query_is_echoed_cut():
    fitted = fit_hits(WORST_QUERY, "hybrid", [])

    assert len(WORST_QUERY) > QUERY_ECHO_CHARS
    assert fitted["query"] == WORST_QUERY[:QUERY_ECHO_CHARS] + "…"
    assert fit_hits("short", "hybrid", [])["query"] == "short"


def test_metadata_too_long_for_every_hit_drops_hits_from_the_end():
    heading = "Ö" * 300  # 1800 characters as JSON

    fitted = fit_hits("x", "dense", [hit(n, "text", heading) for n in range(4)])

    assert size(fitted) <= RESULT_BUDGET_CHARS
    assert [h["chunk_id"] for h in fitted["hits"]] == ["a3f9c2e17b4d8e00"]


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
    assert RESULT_BUDGET_CHARS - size(result.model_dump(mode="json")) >= 1000


# --- excerpts ------------------------------------------------------------------------

CHUNK = "\n".join(
    [
        "# Rate limits",
        "Every project has a submission quota.",
        "Quotas reset at midnight UTC.",
        "Unrelated notes about the dashboard colours.",
        "When you exceed the limit the API answers FERRY-429.",
        "The response carries a Retry-After header in seconds.",
        "Clients retry according to their ferry.retry.* settings.",
        "More notes about the dashboard.",
        "Even more notes about the dashboard and its themes.",
    ]
)


def test_query_terms_are_lower_cased_words_longer_than_two_characters():
    assert query_terms("What does FERRY-429 mean, and is it OK?") == {
        "what",
        "does",
        "ferry",
        "429",
        "mean",
        "and",
    }


def test_an_excerpt_is_the_window_of_lines_with_the_most_query_terms():
    terms = query_terms("What header comes with FERRY-429 when I exceed the limit?")

    got = excerpt(CHUNK, terms, 120)

    assert len(json.dumps(got)) - 2 <= 120
    assert "When you exceed the limit the API answers FERRY-429." in got.split("\n")
    assert "Retry-After header" in got
    assert got in CHUNK and not got.startswith("# Rate limits")


def test_an_excerpt_counts_distinct_terms_not_repeats():
    text = "ferry ferry ferry ferry\nfiller\nfiller\nferry retry quota"

    assert excerpt(text, query_terms("ferry retry quota"), 20) == "ferry retry quota"


def test_an_excerpt_takes_the_earliest_of_equal_windows():
    text = "alpha one\nfiller\nalpha two"

    assert excerpt(text, query_terms("alpha"), 12) == "alpha one"


def test_an_excerpt_without_query_terms_is_the_start():
    got = excerpt(CHUNK, query_terms("kubernetes helm"), 60)

    # The longest start within 60 characters as JSON, where each line break is two.
    assert CHUNK.startswith(got)
    assert len(json.dumps(got)) - 2 <= 60 < len(json.dumps(CHUNK[: len(got) + 1])) - 2


def test_a_line_too_long_for_the_limit_falls_back_to_the_start():
    text = "FERRY-429 " * 30  # one line, 300 characters

    assert excerpt(text, query_terms("FERRY-429"), 50) == text[:50]


def test_an_excerpt_measures_text_as_json():
    text = "→" * 40 + "\nFERRY-429 here"  # each arrow is 6 characters as JSON

    got = excerpt(text, query_terms("FERRY-429"), 30)

    assert got == "FERRY-429 here"
    assert excerpt("→" * 40, frozenset(), 30) == "→" * 5


def test_a_text_that_fits_is_returned_whole_and_a_zero_limit_gives_nothing():
    assert excerpt(CHUNK, frozenset(), 10_000) == CHUNK
    assert excerpt(CHUNK, query_terms("FERRY-429"), 0) == ""


# --- list_sources --------------------------------------------------------------------


def documents(n: int) -> list[dict]:
    """``n`` documents in nested directories, as the RAG API lists them."""
    dirs = ["fastapi/tutorial/", "fastapi/advanced/security/", "uv/guides/", "pydantic/"]
    return [
        {"source_file": f"{dirs[i % len(dirs)]}document-number-{i:03d}.md", "chunks": 1 + i % 7}
        for i in range(n)
    ] + [{"source_file": "README.md", "chunks": 2}]


def test_a_list_that_fits_is_sent_whole():
    docs = documents(10)

    fitted = fit_sources(docs, 99)

    assert fitted == {"sources": docs, "truncated": None, "total_chunks": 99, "directories": None}
    assert SourcesResult.model_validate(fitted).model_dump(mode="json") == fitted


def test_a_long_list_becomes_counts_per_directory_and_the_first_documents():
    docs = documents(300)

    fitted = fit_sources(docs, 1234)

    sent = SourcesResult.model_validate(fitted).model_dump(mode="json")
    assert sent == fitted and size(sent) <= RESULT_BUDGET_CHARS
    assert loop_view(sent) <= AGENT_BUDGET_CHARS
    shown = fitted["truncated"]["shown"]
    assert fitted["truncated"] == {"shown": shown, "of": 301} and 0 < shown < 301
    assert fitted["sources"] == docs[:shown]
    assert fitted["total_chunks"] == 1234
    by_dir = {d["directory"]: (d["documents"], d["chunks"]) for d in fitted["directories"]}
    assert list(by_dir) == [
        "",
        "fastapi/advanced/security/",
        "fastapi/tutorial/",
        "pydantic/",
        "uv/guides/",
    ]
    assert by_dir[""] == (1, 2)  # README.md, at the top level
    tutorial = [d for d in docs if d["source_file"].startswith("fastapi/tutorial/")]
    assert by_dir["fastapi/tutorial/"] == (len(tutorial), sum(d["chunks"] for d in tutorial))


def test_more_directories_than_fit_are_cut_and_the_budget_holds():
    docs = [{"source_file": f"area-{i:04d}/sub/page.md", "chunks": 1} for i in range(400)]

    fitted = fit_sources(docs, 400)

    assert size(fitted) <= RESULT_BUDGET_CHARS
    listed = [d["directory"] for d in fitted["directories"]]
    assert 0 < len(listed) < 400 and listed == sorted(listed)
    assert listed[0] == "area-0000/sub/"
