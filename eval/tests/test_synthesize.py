"""Chunk selection, draft checks and the drafting run, with a fake RAG and a fake LLM."""

import json
import random
import re

import httpx
import pytest

from budget import Budget, BudgetExceeded
from dataset.synthesize import (
    Chunk,
    allocate,
    code_share,
    eligible,
    flag_draft,
    jaccard,
    run,
    select_chunks,
    shared_run,
)
from llm import LLMError, Usage

PROSE = (
    "The resolver picks the newest version that satisfies every constraint in the "
    "project, and it records each choice in the lockfile so that later installs are "
    "reproducible across machines and operating systems without any network lookups. "
) * 2  # 2 x 170 characters: long enough to draft from
CODE = "Intro line.\n```python\n" + "x = compute_something(1, 2, 3)\n" * 12 + "```\n"
SHORT = "Too short to draft from."


def corpus() -> dict[str, list[str]]:
    return {
        "uv/a.md": [SHORT, PROSE + "A1"],
        "uv/b.md": [CODE, PROSE + "B1"],
        "uv/c.md": [PROSE + "C1", PROSE + "C2"],
        "pydantic/a.md": [PROSE + "PA"],
        "pydantic/b.md": [PROSE + "GOLDEN QUOTE HERE"],
        "fastapi/a.md": [PROSE + "FA"],
    }


def fake_rag(chunks: dict[str, list[str]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            docs = [{"source_file": s, "chunks": len(c)} for s, c in chunks.items()]
            return httpx.Response(200, json={"documents": docs, "total_chunks": 0})
        source = request.url.params["source_file"]
        body = [
            {"chunk_id": f"{source}#{i}", "ordinal": i, "section_heading": "Sec", "text": t}
            for i, t in enumerate(chunks[source])
        ]
        return httpx.Response(200, json={"source_file": source, "chunks": body})

    return httpx.Client(base_url="http://rag.test", transport=httpx.MockTransport(handler))


class FakeLLM:
    """Drafts a fixed question and quotes the chunk's last line, or fails on request."""

    model = "fake-model"

    def __init__(
        self,
        *,
        budget=None,
        question="Why does the resolver write a lockfile?",
        quote=None,
        fail_every=0,
    ) -> None:
        self.budget = budget or Budget(1.0)
        self.question, self.quote, self.fail_every = question, quote, fail_every
        self.calls = 0

    def complete(self, system, user, schema, *, name="answer"):
        self.budget.check(0.001)
        self.calls += 1
        self.budget.record(0.001)
        if self.fail_every and self.calls % self.fail_every == 0:
            raise LLMError("the model refused: no")
        text = re.search(r"<<<\n(.*)\n>>>", user, re.S).group(1)
        quote = self.quote or text.strip()[-60:]
        draft = {
            "question": self.question,
            "reference_answer": "Reproducible installs.",
            "quote": quote,
        }
        return schema.model_validate(draft), Usage(10, 5, 0.001)


# --- selection ----------------------------------------------------------------------


def test_code_share_counts_fenced_lines():
    assert code_share(PROSE) == 0
    assert code_share(CODE) > 0.9
    # a chunk that starts inside a fence: the larger reading wins
    tail = "x = 1\n" * 20 + "```\nshort prose"
    assert code_share(tail) > 0.8


def test_allocate_is_proportional_and_sums_to_n():
    share = allocate({"uv": 997, "pydantic": 1246, "fastapi": 1534}, 40)
    assert sum(share.values()) == 40
    assert share == {"uv": 11, "pydantic": 13, "fastapi": 16}


def test_eligibility_rules():
    golden = {"GOLDEN QUOTE HERE"}
    assert not eligible(Chunk("1", "a.md", None, SHORT), golden)
    assert not eligible(Chunk("2", "a.md", None, CODE), golden)
    assert not eligible(Chunk("3", "a.md", None, PROSE + "GOLDEN QUOTE HERE"), golden)
    assert eligible(Chunk("4", "a.md", None, PROSE), golden)


def test_select_takes_at_most_one_eligible_chunk_per_file():
    chunks = select_chunks(fake_rag(corpus()), 6, random.Random(0), set(), {"GOLDEN QUOTE HERE"})
    sources = [c.source for c in chunks]
    assert len(sources) == len(set(sources))
    assert "pydantic/b.md" not in sources  # its only chunk holds a golden quote
    for c in chunks:
        assert eligible(c, {"GOLDEN QUOTE HERE"})


def test_select_honours_skipped_files():
    chunks = select_chunks(fake_rag(corpus()), 6, random.Random(0), {"uv/a.md", "uv/c.md"}, set())
    assert {c.source for c in chunks}.isdisjoint({"uv/a.md", "uv/c.md"})


def test_select_is_reproducible_for_a_seed():
    a = select_chunks(fake_rag(corpus()), 4, random.Random(7), set(), set())
    b = select_chunks(fake_rag(corpus()), 4, random.Random(7), set(), set())
    assert a == b


# --- checks -------------------------------------------------------------------------


def test_shared_run_finds_six_word_overlap():
    chunk = "Set the cache directory with the UV_CACHE_DIR variable."
    assert shared_run("How do I set the cache directory with uv?", chunk) is None  # 5 words
    assert shared_run("Can I set the cache directory with the CLI?", chunk) == (
        "set the cache directory with the"
    )


def test_jaccard():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert jaccard("Where is the cache?", "where is THE cache") == 1.0


def test_flags():
    chunk = "uv keeps a cache. Set UV_CACHE_DIR to move it."
    assert (
        flag_draft("Where does uv put downloads?", "Set UV_CACHE_DIR to move it.", chunk, {}) == []
    )
    flags = flag_draft(
        "where does uv put downloads",
        "not in the chunk at all",
        chunk,
        {"q001": "Where does uv put downloads?"},
    )
    assert "quote not in chunk" in flags
    assert any(f.startswith("near-duplicate of q001") for f in flags)
    assert flag_draft("Q?", "short", chunk + " short", {}) == [
        "quote is 5 characters (12-120 allowed)"
    ]


# --- the run ------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "What is in the lockfile?",
                "reference_answer": "x",
                "category": "lookup",
                "origin": "targeted",
                "split": "test",
                "verified": True,
                "evidence": [
                    {"quotes": [{"source": "pydantic/b.md", "text": "GOLDEN QUOTE HERE"}]}
                ],
            }
        )
        + "\n"
    )
    return golden, tmp_path / "candidates.jsonl", tmp_path / "rejects.jsonl"


def test_run_writes_flagged_candidates(files, capsys):
    golden, out, rejects = files
    code = run(n=4, llm=FakeLLM(), rag=fake_rag(corpus()), golden=golden, out=out, rejects=rejects)
    assert code == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["candidate_id"] for r in rows] == ["c001", "c002", "c003", "c004"]
    first = rows[0]
    assert set(first) >= {
        "source",
        "heading",
        "chunk_id",
        "chunk_text",
        "question",
        "reference_answer",
        "quote",
        "flags",
        "model",
        "cost_usd",
    }
    assert first["model"] == "fake-model" and first["cost_usd"] == 0.001
    # the same question each time: every later draft is a near-duplicate of an earlier one
    assert rows[0]["flags"] == []
    assert any("near-duplicate of c001" in f for f in rows[1]["flags"])
    assert "4 drafts written" in capsys.readouterr().out


def test_a_second_run_appends_and_avoids_used_files(files):
    golden, out, rejects = files
    run(n=2, llm=FakeLLM(), rag=fake_rag(corpus()), golden=golden, out=out, rejects=rejects)
    first = [json.loads(line) for line in out.read_text().splitlines()]
    run(
        n=2,
        llm=FakeLLM(question="Something else entirely?"),
        rag=fake_rag(corpus()),
        golden=golden,
        out=out,
        rejects=rejects,
        seed=1,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[: len(first)] == first
    assert [r["candidate_id"] for r in rows[len(first) :]] == ["c003", "c004"][
        : len(rows) - len(first)
    ]
    sources = [r["source"] for r in rows]
    assert len(sources) == len(set(sources))


def test_the_budget_stops_the_run_and_keeps_paid_drafts(files, capsys):
    golden, out, rejects = files
    llm = FakeLLM(budget=Budget(0.0025))  # room for two $0.001 calls
    code = run(n=4, llm=llm, rag=fake_rag(corpus()), golden=golden, out=out, rejects=rejects)
    assert code == 1
    assert len(out.read_text().splitlines()) == 2
    assert "stopped:" in capsys.readouterr().out
    assert llm.budget.spent_usd <= 0.0025


def test_failed_drafts_are_counted_not_written(files, capsys):
    golden, out, rejects = files
    run(
        n=4,
        llm=FakeLLM(fail_every=2),
        rag=fake_rag(corpus()),
        golden=golden,
        out=out,
        rejects=rejects,
    )
    assert len(out.read_text().splitlines()) == 2
    assert "2 failed" in capsys.readouterr().out


def test_budget_exceeded_is_the_llm_contract():
    # the fake mirrors LLM.complete: check before the call, record after it
    llm = FakeLLM(budget=Budget(0.0))
    with pytest.raises(BudgetExceeded):
        llm.complete("", "<<<\nx\n>>>", object)
