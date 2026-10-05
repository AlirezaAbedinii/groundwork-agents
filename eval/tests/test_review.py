"""The review tool, driven by scripted keys and a scripted editor."""

import json

import pytest

from dataset.review import review

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
