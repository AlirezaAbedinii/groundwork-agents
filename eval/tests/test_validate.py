"""The validator, offline, against a corpus on disk and against a fake RAG API."""

import json
from pathlib import Path

import httpx
import pytest

from dataset.validate import main

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus"
GOLDEN = FIXTURES / "golden_3rows.jsonl"

# What a RAG API serving the fixture corpus returns from /v1/chunks, per source.
CHUNKS = {
    "tool/cache.md": [
        "The tool keeps downloaded archives in a cache directory, so a repeated\n"
        "install never fetches the same archive twice.",
        "Set `TOOL_CACHE_DIR` to move the cache, or pass `--cache-dir` on the\n"
        "command line.\n\n```bash\n# remove every cached archive\ntool cache clean\n```",
        "Run `tool cache clean` to remove every cached archive.",
    ],
    "web/errors.md": [
        "Raise `HTTPError` with a status code and a detail message; the server turns it\n"
        "into a JSON response with that status.",
        "A request body that fails validation gets a 422 response that lists every\ninvalid field.",
    ],
}


def fake_rag(chunks: dict[str, list[str]] = CHUNKS, *, index: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/config":
            info = {
                "embedding_model": "fake-embedder",
                "dim": 8,
                "chunk_strategy": "fixed",
                "chunk_size": 800,
                "chunk_overlap": 120,
                "chunks": sum(map(len, chunks.values())),
            }
            return httpx.Response(
                200, json={"collection": "docs_test", "index": info if index else None}
            )
        if request.url.path == "/v1/documents":
            docs = [{"source_file": s, "chunks": len(c)} for s, c in chunks.items()]
            return httpx.Response(200, json={"documents": docs, "total_chunks": 0})
        if request.url.path == "/v1/chunks":
            source = request.url.params["source_file"]
            if source not in chunks:
                return httpx.Response(404, json={"detail": "unknown"})
            body = [
                {"chunk_id": f"{source}-{i}", "ordinal": i, "section_heading": None, "text": t}
                for i, t in enumerate(chunks[source])
            ]
            return httpx.Response(200, json={"source_file": source, "chunks": body})
        return httpx.Response(404)

    return httpx.Client(base_url="http://rag.test", transport=httpx.MockTransport(handler))


def golden_rows() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines()]


def write_rows(path: Path, rows: list[dict | str]) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return path


def run(capsys, *args: str, rag: httpx.Client | None = None) -> tuple[int, str]:
    code = main([str(a) for a in args], rag_client=rag)
    return code, capsys.readouterr().out


def with_quote(text: str, source: str = "tool/cache.md") -> list[dict]:
    rows = golden_rows()
    rows[0]["evidence"] = [{"quotes": [{"source": source, "text": text}]}]
    return rows


# --- offline ------------------------------------------------------------------------


def test_the_fixture_passes_with_the_corpus_and_rag(capsys):
    code, out = run(capsys, GOLDEN, "--corpus", CORPUS, "--rag", "http://rag.test", rag=fake_rag())
    assert code == 0, out
    assert (
        "3 rows · targeted 3 / synthetic 0 · lookup 1, multi_hop 1, no_answer 1, ambiguous 0" in out
    )
    assert "corpus: 3 quotes in 2 files, all found" in out
    assert "rag: 3 quotes checked against docs_test (fake-embedder; fixed 800/120; 5 chunks)" in out
    assert out.rstrip().endswith("0 errors, 1 warning")  # the composition warning only
    assert "WARN  composition: lookup 1/37" in out


def test_schema_errors_are_reported_per_row(tmp_path, capsys):
    rows = golden_rows()
    no_answer_with_evidence = rows[2] | {"id": "q004", "evidence": rows[0]["evidence"]}
    too_long = rows[0] | {"id": "q005"}
    too_long["evidence"] = [{"quotes": [{"source": "tool/cache.md", "text": "x" * 121}]}]
    unverified = rows[0] | {"id": "q006", "verified": False}
    no_split = {k: v for k, v in rows[0].items() if k != "split"} | {"id": "q007"}
    misspelled = rows[0] | {"id": "q008", "note": "typo"}
    path = write_rows(
        tmp_path / "g.jsonl",
        [
            *rows,
            "{not json",
            "[1, 2]",
            rows[0],
            no_answer_with_evidence,
            too_long,
            unverified,
            no_split,
            misspelled,
        ],
    )
    code, out = run(capsys, path)
    assert code == 1
    assert "ERROR line 4: invalid JSON" in out
    assert "ERROR line 5: not a JSON object" in out
    assert "ERROR q001 (line 6): duplicate id (first on line 1)" in out
    assert "ERROR q004 (line 7): a no_answer question has no evidence" in out
    assert "ERROR q005 (line 8): evidence[0].quotes[0].text: String should have at most 120" in out
    assert "ERROR q006: verified is false" in out
    assert "ERROR q007 (line 10): split is missing (run with --assign-splits)" in out
    assert "ERROR q008 (line 11): note: Extra inputs are not permitted" in out
    assert "11 rows (5 parsed)" in out


def test_an_empty_file_is_valid(tmp_path, capsys):
    path = tmp_path / "g.jsonl"
    path.write_text("")
    code, out = run(capsys, path)
    assert code == 0
    assert "0 rows · targeted 0 / synthetic 0" in out


def test_split_imbalance_is_a_warning(tmp_path, capsys):
    rows = []
    for n in range(1, 6):
        rows.append(golden_rows()[2] | {"id": f"q{n:03}", "split": "test"})
    code, out = run(capsys, write_rows(tmp_path / "g.jsonl", rows))
    assert code == 0
    assert "WARN  splits: no_answer has 0 dev / 5 test, 2 dev expected" in out


# --- --corpus -----------------------------------------------------------------------


def test_a_near_miss_points_at_the_closest_line(tmp_path, capsys):
    path = write_rows(tmp_path / "g.jsonl", with_quote("Set TOOL_CACHE_DIR to move the cache"))
    code, out = run(capsys, path, "--corpus", CORPUS)
    assert code == 1
    assert (
        "ERROR q001 evidence[0].quotes[0] (tool/cache.md): not in the file; closest is line 8:"
        in out
    )
    assert "corpus: 3 quotes in 2 files, 1 not found" in out


def test_a_quote_across_a_heading_is_an_error(tmp_path, capsys):
    path = write_rows(tmp_path / "g.jsonl", with_quote("the same archive twice. ## Cache location"))
    code, out = run(capsys, path, "--corpus", CORPUS)
    assert code == 1
    assert "on or across a heading line, which no chunk holds" in out


def test_a_hash_line_inside_a_code_fence_isnt_a_heading(tmp_path, capsys):
    path = write_rows(
        tmp_path / "g.jsonl", with_quote("# remove every cached archive tool cache clean")
    )
    code, out = run(capsys, path, "--corpus", CORPUS)
    assert code == 0, out


def test_a_missing_source_file_is_an_error(tmp_path, capsys):
    path = write_rows(
        tmp_path / "g.jsonl", with_quote("a quote from nowhere", source="tool/gone.md")
    )
    code, out = run(capsys, path, "--corpus", CORPUS)
    assert code == 1
    assert "(tool/gone.md): no such file under" in out


# --- --rag --------------------------------------------------------------------------


def test_a_quote_no_chunk_contains_is_an_error(tmp_path, capsys):
    chunks = CHUNKS | {"tool/cache.md": ["Set `TOOL_CACHE_DIR` to move the cache, or pass"]}
    path = write_rows(tmp_path / "g.jsonl", golden_rows())
    code, out = run(capsys, path, "--rag", "http://rag.test", rag=fake_rag(chunks))
    assert code == 1
    assert (
        "ERROR q001 evidence[0].quotes[0] (tool/cache.md): no chunk of docs_test contains it" in out
    )


def test_an_unindexed_source_is_an_error(tmp_path, capsys):
    chunks = {"web/errors.md": CHUNKS["web/errors.md"]}
    code, out = run(capsys, GOLDEN, "--rag", "http://rag.test", rag=fake_rag(chunks))
    assert code == 1
    assert "(tool/cache.md): source isn't indexed in docs_test" in out


def test_a_quote_in_many_chunks_is_not_specific_enough(tmp_path, capsys):
    chunks = CHUNKS | {
        "web/more.md": ["Run `tool cache clean` to remove every cached archive."] * 2
    }
    code, out = run(capsys, GOLDEN, "--rag", "http://rag.test", rag=fake_rag(chunks))
    assert code == 0
    assert (
        "WARN  q002 evidence[0].quotes[0] (tool/cache.md): in 3 chunks "
        "(web/more.md ×2, tool/cache.md ×1): not specific enough"
    ) in out


def test_a_quote_found_only_in_another_source_is_a_warning(capsys):
    chunks = CHUNKS | {
        "tool/cache.md": CHUNKS["tool/cache.md"][:2],
        "tool/copy.md": ["Run `tool cache clean` to remove every cached archive."],
    }
    code, out = run(capsys, GOLDEN, "--rag", "http://rag.test", rag=fake_rag(chunks))
    assert code == 0
    assert "only chunks of other sources contain it (tool/copy.md)" in out


def test_an_unseeded_collection_is_an_error(capsys):
    code, out = run(capsys, GOLDEN, "--rag", "http://rag.test", rag=fake_rag(index=False))
    assert code == 1
    assert "ERROR rag: collection 'docs_test' isn't indexed" in out


def test_an_unreachable_rag_is_an_error(capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(base_url="http://rag.test", transport=httpx.MockTransport(refuse))
    code, out = run(capsys, GOLDEN, "--rag", "http://rag.test", rag=client)
    assert code == 1
    assert "ERROR rag: http://rag.test: connection refused" in out


# --- --assign-splits ----------------------------------------------------------------


def unsplit(n: int, category: str = "no_answer", start: int = 1) -> list[dict]:
    template = {k: v for k, v in golden_rows()[2].items() if k != "split"}
    return [template | {"id": f"q{i:03}", "category": category} for i in range(start, start + n)]


def test_assign_splits_fills_toward_40_60_per_category(tmp_path, capsys):
    path = write_rows(tmp_path / "g.jsonl", unsplit(18) + unsplit(5, "ambiguous", start=19))
    code, out = run(capsys, path, "--assign-splits")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    no_answer = [r["split"] for r in rows if r["category"] == "no_answer"]
    assert (no_answer.count("dev"), no_answer.count("test")) == (7, 11)
    # ambiguous rows here have no evidence, so they fail the schema, but still get a split
    ambiguous = [r["split"] for r in rows if r["category"] == "ambiguous"]
    assert (ambiguous.count("dev"), ambiguous.count("test")) == (2, 3)
    assert "assigned splits to 23 rows" in out
    assert list(rows[0]).index("split") == list(rows[0]).index("origin") + 1


def test_assign_splits_never_moves_a_row_and_leaves_other_lines_alone(tmp_path, capsys):
    kept = (
        '{"id": "q001",   "question": "kept verbatim", "reference_answer": "a", '
        '"category": "no_answer", "evidence": [], "origin": "targeted", "split": "test", '
        '"verified": true}'
    )
    path = write_rows(tmp_path / "g.jsonl", [kept, *unsplit(4, start=2)])
    run(capsys, path, "--assign-splits")
    lines = path.read_text().splitlines()
    assert lines[0] == kept
    splits = [json.loads(line)["split"] for line in lines]
    assert splits[0] == "test"
    assert splits.count("dev") == 2  # 40 % of 5, around the row that was already test

    before = path.read_text()
    code, out = run(capsys, path, "--assign-splits")
    assert code == 0
    assert "assigned" not in out
    assert path.read_text() == before


def test_assign_splits_is_deterministic(tmp_path, capsys):
    a = write_rows(tmp_path / "a.jsonl", unsplit(10))
    b = write_rows(tmp_path / "b.jsonl", unsplit(10))
    run(capsys, a, "--assign-splits")
    run(capsys, b, "--assign-splits")
    assert a.read_text() == b.read_text()


# --- --tasks ------------------------------------------------------------------------


def task(tid: str, category: str, **overrides) -> dict:
    base = {
        "id": tid,
        "request": "Look it up in the docs and write notes.md",
        "category": category,
        "must_call": [{"tool": "rag_search"}, {"tool": "file_write", "args": {"path": "notes.md"}}],
        "rubric": [],
    }
    return base | overrides


def test_tasks_mode(tmp_path, capsys):
    path = write_rows(tmp_path / "t.jsonl", [task("t01", "docs"), task("t02", "code")])
    code, out = run(capsys, path, "--tasks")
    assert code == 0, out
    assert "2 tasks · code 1, docs 1" in out


def test_tasks_mode_reports_bad_rows(tmp_path, capsys):
    missing = {k: v for k, v in task("t02", "docs").items() if k != "must_call"}
    path = write_rows(tmp_path / "t.jsonl", [task("t01", "docs"), missing, task("t01", "code")])
    code, out = run(capsys, path, "--tasks")
    assert code == 1
    assert "ERROR t02 (line 2): must_call: Field required" in out
    assert "ERROR t01 (line 3): duplicate id (first on line 1)" in out


def test_tasks_mode_takes_no_golden_options(tmp_path):
    path = write_rows(tmp_path / "t.jsonl", [task("t01", "docs")])
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--tasks", "--assign-splits"])
    assert exc.value.code == 2


def test_the_committed_files_are_valid(capsys):
    root = Path(__file__).parent.parent
    assert main([str(root / "golden" / "golden_set.jsonl")]) == 0
    assert main([str(root / "tasks" / "agent_tasks.jsonl"), "--tasks"]) == 0
