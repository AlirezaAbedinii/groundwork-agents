"""The review tool, driven by scripted keys and a scripted editor."""

import json
from pathlib import Path

import pytest

from dataset.review import quote_lines, review, review_targeted

CHUNK = (
    "uv keeps downloaded archives in a cache. Set UV_CACHE_DIR to move the cache "
    "somewhere else, or pass --cache-dir on the command line."
)


def candidate(cid: str, source: str, **overrides) -> dict:
    base = {
        "candidate_id": cid,
        "source": source,
        "heading": "Cache directory",
        "chunk_id": f"{source}#3",
        "chunk_text": CHUNK,
        "question": "How can I keep uv's downloads on another disk?",
        "reference_answer": "Set UV_CACHE_DIR or pass --cache-dir.",
        "quote": "Set UV_CACHE_DIR to move the cache",
        "flags": [],
        "model": "gpt-4o-mini",
        "cost_usd": 0.0002,
        "drafted_on": "2026-10-05",
    }
    return base | overrides


@pytest.fixture
def files(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "s003",
                "question": "Older synthetic row?",
                "reference_answer": "x",
                "category": "lookup",
                "origin": "synthetic",
                "split": "dev",
                "verified": True,
                "evidence": [{"quotes": [{"source": "uv/x.md", "text": "an older quote here"}]}],
            }
        )
        + "\n"
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(
            json.dumps(c) + "\n"
            for c in [
                candidate("c001", "uv/concepts/cache.md"),
                candidate("c002", "uv/b.md"),
                candidate("c003", "uv/c.md"),
            ]
        )
    )
    return golden, candidates, tmp_path / "rejects.jsonl"


def keys(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_accept_reject_skip(files):
    golden, candidates, rejects = files
    out: list[str] = []
    tally = review(
        candidates,
        golden,
        rejects,
        ask=keys("a", "r", "", "r", "not a user's question", "s"),
        edit=None,
        out=out.append,
    )
    assert tally == {"accepted": 1, "rejected": 1, "skipped": 1}

    rows = lines(golden)
    assert [r["id"] for r in rows] == ["s003", "s004"]
    row = rows[1]
    assert row["origin"] == "synthetic" and row["verified"] is True and row["category"] == "lookup"
    assert row["split"] in ("dev", "test")
    assert row["evidence"] == [
        {
            "quotes": [
                {"source": "uv/concepts/cache.md", "text": "Set UV_CACHE_DIR to move the cache"}
            ]
        }
    ]
    assert "drafted by gpt-4o-mini" in row["notes"] and "uv/concepts/cache.md" in row["notes"]

    assert [(r["candidate_id"], r["reason"]) for r in lines(rejects)] == [
        ("c002", "not a user's question")
    ]
    assert "a reject needs a reason" in out
    assert [c["candidate_id"] for c in lines(candidates)] == ["c003"]  # the skipped one stays


def test_quit_leaves_the_rest_pending(files):
    golden, candidates, rejects = files
    tally = review(candidates, golden, rejects, ask=keys("q"), edit=None, out=lambda s: None)
    assert tally == {"accepted": 0, "rejected": 0, "skipped": 0}
    assert len(lines(candidates)) == 3
    assert len(lines(golden)) == 1


def test_edit_rechecks_the_flags(files):
    golden, candidates, rejects = files

    def editor(text):
        fields = json.loads(text)
        fields["quote"] = "a quote that isn't in the chunk"
        return json.dumps(fields)

    out: list[str] = []
    review(candidates, golden, rejects, ask=keys("e", "q"), edit=editor, out=out.append)
    edited = lines(candidates)[0]
    assert edited["quote"] == "a quote that isn't in the chunk"
    assert "quote not in chunk" in edited["flags"]
    assert "FLAG: quote not in chunk" in out


def test_a_draft_the_schema_rejects_cannot_be_accepted(files):
    golden, candidates, rejects = files
    candidates.write_text(json.dumps(candidate("c001", "uv/a.md", quote="x" * 121)) + "\n")
    out: list[str] = []
    review(
        candidates, golden, rejects, ask=keys("a", "r", "quote too long"), edit=None, out=out.append
    )
    assert any(line.startswith("not accepted; edit the draft first") for line in out)
    assert len(lines(golden)) == 1
    assert lines(rejects)[0]["reason"] == "quote too long"


def test_a_bad_edit_is_ignored(files):
    golden, candidates, rejects = files
    out: list[str] = []
    review(
        candidates,
        golden,
        rejects,
        ask=keys("e", "q"),
        edit=lambda text: "{not json",
        out=out.append,
    )
    assert any(line.startswith("edit ignored") for line in out)
    assert lines(candidates)[0]["quote"] == "Set UV_CACHE_DIR to move the cache"


# --- targeted candidates ------------------------------------------------------------

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


QUOTE = "Set `TOOL_CACHE_DIR` to move the cache, or pass `--cache-dir` on the"


def targeted(rid: str, **overrides) -> dict:
    base = {
        "id": rid,
        "question": "How do I move the tool's cache?",
        "reference_answer": "Set TOOL_CACHE_DIR or pass --cache-dir.",
        "category": "lookup",
        "evidence": [{"quotes": [{"source": "tool/cache.md", "text": QUOTE}]}],
        "origin": "targeted",
        "verified": False,
        "phrasing": "exact_token",
    }
    return base | overrides


@pytest.fixture
def targeted_files(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({"id": "q050"}) + "\n")
    candidates = tmp_path / "candidates.jsonl"
    rows = [targeted("q051"), targeted("q052", verified=True), targeted("q053"), targeted("q054")]
    candidates.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return golden, candidates, tmp_path / "rejects.jsonl"


def test_targeted_accept_reject_skip(targeted_files):
    golden, candidates, rejects = targeted_files
    out: list[str] = []
    tally = review_targeted(
        candidates,
        golden,
        rejects,
        CORPUS,
        ask=keys("a", "r", "", "r", "the answer is incomplete", "s"),
        edit=None,
        out=out.append,
    )
    assert tally == {"accepted": 1, "rejected": 1, "skipped": 1}
    rows = lines(candidates)
    # q051 verified in place, q052 was verified already (not shown), q053 rejected, q054 skipped
    assert [(r["id"], r["verified"]) for r in rows] == [
        ("q051", True),
        ("q052", True),
        ("q054", False),
    ]
    assert rows[0] == targeted("q051", verified=True)  # nothing else changed, no split added
    assert [(r["id"], r["reason"]) for r in lines(rejects)] == [
        ("q053", "the answer is incomplete")
    ]
    assert "a reject needs a reason" in out
    assert not any(line.startswith("== q052") for line in out)
    assert lines(golden) == [{"id": "q050"}]  # nothing is promoted


def test_targeted_shows_each_quote_with_its_lines(targeted_files):
    golden, candidates, rejects = targeted_files
    out: list[str] = []
    review_targeted(candidates, golden, rejects, CORPUS, ask=keys("q"), edit=None, out=out.append)
    assert "== q051  lookup × exact_token" in out
    assert "-- evidence[0] quote 1: tool/cache.md" in out
    assert "       7 | " in out  # the line before the quote (empty)
    assert "  >    8 | Set `TOOL_CACHE_DIR` to move the cache, or pass `--cache-dir` on the" in out
    assert "       9 | command line." in out  # the quote ends on line 8


def test_targeted_marks_a_quote_that_isnt_in_the_file(tmp_path):
    golden = tmp_path / "golden.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    row = targeted("q051")
    row["evidence"][0]["quotes"][0]["text"] = "a sentence the file doesn't have"
    candidates.write_text(json.dumps(row) + "\n")
    out: list[str] = []
    review_targeted(
        candidates, golden, tmp_path / "r.jsonl", CORPUS, ask=keys("q"), edit=None, out=out.append
    )
    assert "   NOT FOUND in tool/cache.md" in out


def test_quote_lines_finds_the_shortest_run():
    text = ["alpha beta", "gamma delta", "epsilon", "gamma delta"]
    assert quote_lines(text, "beta gamma") == (1, 2)
    assert quote_lines(text, "delta") == (2, 2)
    assert quote_lines(text, "zeta") is None


def test_targeted_edit_is_saved_only_if_it_still_validates(targeted_files):
    golden, candidates, rejects = targeted_files

    def editor(answers):
        it = iter(answers)

        def edit(text):
            fields = json.loads(text)
            assert "id" not in fields and "verified" not in fields
            return next(it)(fields)

        return edit

    edits = [
        lambda f: json.dumps(f | {"phrasing": "loose"}),  # not a phrasing
        lambda f: json.dumps(  # a quote the file doesn't have
            f
            | {"evidence": [{"quotes": [{"source": "tool/cache.md", "text": "no such text here"}]}]}
        ),
        lambda f: "{broken",
        lambda f: json.dumps(f | {"question": "Where can the cache go?", "phrasing": "paraphrase"}),
    ]
    out: list[str] = []
    review_targeted(
        candidates,
        golden,
        rejects,
        CORPUS,
        ask=keys("e", "e", "e", "e", "q"),
        edit=editor(edits),
        out=out.append,
    )
    saved = [m for m in out if m.startswith("edit not saved")]
    assert len(saved) == 2
    assert "phrasing" in saved[0] and "not in the file" in saved[1]
    assert any(m.startswith("edit ignored") for m in out)
    row = lines(candidates)[0]
    assert row["id"] == "q051" and row["verified"] is False
    assert row["question"] == "Where can the cache go?" and row["phrasing"] == "paraphrase"
