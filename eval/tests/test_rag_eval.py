"""The retrieval stage of the RAG evaluation, end to end against a fake RAG service.

The fake answers ``/v1/config`` and ``/v1/search`` over ``httpx.MockTransport`` with
canned hits for a 6-question golden set, so every number below can be worked out by
hand. The committed fixture under ``tests/fixtures/rag_report/`` is that run's records
file and the report scored from it; ``python -m rag_eval score --records
tests/fixtures/rag_report/rag-records.jsonl`` rewrites the report files in place.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest

import rag_eval
from schemas import EvidenceItem, GoldenQuestion, Quote

FIXTURE = Path(__file__).parent / "fixtures" / "rag_report"
CONFIG = {
    "collection": "fixture",
    "embedding_provider": "sentence_transformers",
    "embedding_model": "fake-embedder",
    "llm_provider": "openai",
    "generation_model": "gpt-4o-mini",
    "default_mode": "hybrid",
    "top_k": 10,
    "rerank_top_k": 5,
    "thresholds": {"dense": 0.3, "hybrid": 0.3},
    "citation_verification": False,
    "index": {
        "embedding_model": "fake-embedder",
        "dim": 8,
        "chunk_strategy": "fixed",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "chunks": 5,
    },
}

QUOTE_A = "Set `TOOL_CACHE_DIR` to move the cache"
QUOTE_B = "Run `tool cache clean` to remove"
QUOTE_C = "an invalid body gets a 422 response"
QUOTE_D = "listens on port 8080 by default"
QUOTE_E = "the lockfile is `tool.lock`"
TEXT = {
    "A": f"{QUOTE_A}, or pass `--cache-dir`.",
    "B": f"{QUOTE_B} every cached archive.",
    "C": f"Requests are validated: {QUOTE_C} listing each field.",
    "BC": f"{QUOTE_B} every cached archive. Also, {QUOTE_C}.",
    "D": f"The server {QUOTE_D}.",
    # E sits past character 150 but inside the first 300: a 300-char snippet keeps it,
    # a 150-char one cuts it off.
    "E": "Some words about projects. " * 6 + f"In short, {QUOTE_E} in the root.",
    "X": "Unrelated text about logging.",
}


def q(id_, question, category, split, *quotes, origin="targeted"):
    evidence = [EvidenceItem(quotes=[Quote(source="tool/x.md", text=t)]) for t in quotes]
    return GoldenQuestion(
        id=id_,
        question=question,
        reference_answer="ref",
        category=category,
        evidence=evidence,
        origin=origin,
        split=split,
        verified=True,
    )


GOLDEN = [
    q("q001", "Where is the cache?", "lookup", "test", QUOTE_A),
    q(
        "q002",
        "How do I clear it, and what does a bad body get?",
        "multi_hop",
        "dev",
        QUOTE_B,
        QUOTE_C,
    ),
    q("q003", "How much does hosting cost?", "no_answer", "test"),
    q("q004", "Is there a GUI?", "no_answer", "dev"),
    q("s001", "What is the default port?", "lookup", "dev", QUOTE_D, origin="synthetic"),
    q("s002", "Which file holds the lock?", "lookup", "test", QUOTE_E, origin="synthetic"),
]
# (question, mode) -> [(text key, score)], best first.
HITS = {
    ("Where is the cache?", "dense"): [("X", 0.62), ("A", 0.60)],
    ("Where is the cache?", "hybrid"): [("A", 0.95), ("X", 0.40)],
    ("How do I clear it, and what does a bad body get?", "dense"): [
        ("B", 0.55),
        ("X", 0.5),
        ("C", 0.45),
    ],
    ("How do I clear it, and what does a bad body get?", "hybrid"): [("BC", 0.90)],
    ("How much does hosting cost?", "dense"): [("X", 0.58)],
    ("How much does hosting cost?", "hybrid"): [("X", 0.10)],
    ("Is there a GUI?", "dense"): [("X", 0.50)],
    ("Is there a GUI?", "hybrid"): [("X", 0.05)],
    ("What is the default port?", "dense"): [("D", 0.70)],
    ("What is the default port?", "hybrid"): [("D", 0.97)],
    ("Which file holds the lock?", "dense"): [("E", 0.66)],
    ("Which file holds the lock?", "hybrid"): [("E", 0.90)],
}


class FakeRag:
    def __init__(self, config=CONFIG, fail=()):
        self.config, self.fail, self.searches = config, set(fail), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/config":
            return httpx.Response(200, json=self.config)
        body = json.loads(request.content)
        self.searches.append(body)
        if (body["query"], body["mode"]) in self.fail:
            return httpx.Response(503, json={"detail": "busy"})
        hits = [
            {
                "chunk_id": f"c-{key}",
                "text": TEXT[key],
                "score": score,
                "source_file": "tool/x.md",
                "section_heading": None,
            }
            for key, score in HITS[(body["query"], body["mode"])]
        ]
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "mode": body["mode"],
                "hits": hits,
                "timings_ms": {"total_ms": 12.5},
            },
        )


def client(fake: FakeRag) -> httpx.Client:
    return httpx.Client(base_url="http://rag", transport=httpx.MockTransport(fake))


@pytest.fixture
def golden(tmp_path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(row.model_dump_json() + "\n" for row in GOLDEN))
    return path


def collect(tmp_path, golden, fake, run_id="run1") -> int:
    args = [
        "collect",
        "--stage",
        "retrieval",
        "--run-id",
        run_id,
        "--golden",
        str(golden),
        "--runs",
        str(tmp_path / "runs"),
    ]
    return rag_eval.main(args, rag_client=client(fake))


def score_run(tmp_path, golden, out, run_id="run1") -> None:
    args = [
        "score",
        "--run-id",
        run_id,
        "--golden",
        str(golden),
        "--runs",
        str(tmp_path / "runs"),
        "--out",
        str(out),
    ]
    assert rag_eval.main(args) == 0


# --- collect ------------------------------------------------------------------------


def test_collect_records_the_config_and_every_search(tmp_path, golden):
    fake = FakeRag()
    assert collect(tmp_path, golden, fake) == 0
    run_dir = tmp_path / "runs" / "run1"
    assert json.loads((run_dir / "run.json").read_text())["config"] == CONFIG
    records = [json.loads(line) for line in (run_dir / "retrieval.jsonl").read_text().splitlines()]
    assert len(records) == len(fake.searches) == 12  # 6 questions x 2 modes
    assert {s["top_k"] for s in fake.searches} == {10}
    assert records[0]["hits"][0] == {
        "chunk_id": "c-X",
        "source_file": "tool/x.md",
        "section_heading": None,
        "score": 0.62,
        "text": TEXT["X"],
    }


def test_resume_skips_what_was_recorded_and_retries_errors(tmp_path, golden):
    assert collect(tmp_path, golden, FakeRag(fail={("Is there a GUI?", "hybrid")})) == 1
    again = FakeRag()
    assert collect(tmp_path, golden, again) == 0
    assert again.searches == [{"query": "Is there a GUI?", "mode": "hybrid", "top_k": 10}]


def test_resume_refuses_a_service_whose_config_changed(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    with pytest.raises(SystemExit, match="config changed"):
        collect(tmp_path, golden, FakeRag(config={**CONFIG, "collection": "other"}))


# --- score --------------------------------------------------------------------------


def scored(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    records = rag_eval.build_records(tmp_path / "runs" / "run1", golden)
    return records, rag_eval.score_records(records)


def test_records_say_which_items_each_hit_covers_in_full_and_as_snippets(tmp_path, golden):
    records, _ = scored(tmp_path, golden)
    header, rows = records[0], records[1:]
    assert header["golden"]["rows"] == 6 and header["missing"] == 0
    s002 = next(r for r in rows if r["id"] == "s002" and r["mode"] == "dense")
    assert s002["hits"][0]["covered"] == [0]
    assert s002["hits"][0]["covered_300"] == [0]
    assert s002["hits"][0]["covered_150"] == []  # the quote starts past character 150
    q003 = next(r for r in rows if r["id"] == "q003" and r["mode"] == "hybrid")
    assert (q003["n_items"], q003["top_score"]) == (0, 0.10)


def test_retrieval_scores_by_mode(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    dense, hybrid = scores["dense"]["retrieval"], scores["hybrid"]["retrieval"]
    # Dense, the 4 answerable questions: q001 finds its item at rank 2, q002 its two at
    # ranks 1 and 3, s001 and s002 theirs at rank 1.
    # MRR = (1/2 + 1 + 1 + 1) / 4 = 0.875 · R@1 = (0 + 1/2 + 1 + 1) / 4 = 0.625
    assert dense["all"]["n"] == 4
    assert dense["all"]["MRR@10"] == pytest.approx(0.875)
    assert dense["all"]["R@1"] == pytest.approx(0.625)
    assert dense["all"]["R@3"] == pytest.approx(1.0)
    # Hybrid finds everything at rank 1 (q002's two items in one chunk).
    assert hybrid["all"]["nDCG@10"] == pytest.approx(1.0)
    assert set(dense["by_category"]) == {"lookup", "multi_hop"}
    assert dense["by_origin"]["synthetic"]["n"] == 2
    assert dense["by_split"]["dev"]["n"] == 2


def test_the_snippet_check_cuts_hits_like_mcp(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    snippets = scores["dense"]["snippets"]  # test split: q001 and s002
    assert snippets["n"] == 2
    assert snippets["R@3 full"] == snippets["R@3 300 chars"] == pytest.approx(1.0)
    assert snippets["R@5 150 chars"] == pytest.approx(0.5)  # s002's quote is cut off


def test_refusal_thresholds_are_chosen_on_dev_and_measured_on_test(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    dense, hybrid = scores["dense"]["refusal"], scores["hybrid"]["refusal"]
    # Dense dev scores: q002 0.55, q004 (no_answer) 0.50, s001 0.70. At 0.55 only q004 is
    # refused: F1 = 1, the lowest threshold that gets it.
    assert dense["chosen_threshold"] == 0.55
    # Dense test: q001 0.62, q003 (no_answer) 0.58, s002 0.66 -> nothing is below 0.55.
    assert dense["chosen"]["test"]["recall"] == 0.0
    assert dense["chosen"]["test"]["precision"] is None
    # Hybrid dev: q004 0.05 against 0.90 and 0.97 -> 0.90; on test only q003 (0.10) is
    # below it, and s002's 0.90 is not (refuse strictly below).
    assert hybrid["chosen_threshold"] == 0.90
    test = hybrid["chosen"]["test"]
    assert (test["tp"], test["fp"], test["fn"], test["tn"]) == (1, 0, 0, 2)
    assert test["recall_ci"][1] == 1.0
    assert hybrid["configured_threshold"] == 0.3
    assert dense["configured"]["tp"] == 0  # 0.3 refuses nothing in dense mode


def test_dev_with_nothing_to_refuse_reports_no_threshold(tmp_path):
    golden = tmp_path / "golden.jsonl"
    rows = [row for row in GOLDEN if row.id != "q004"]  # dev loses its no_answer row
    golden.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    collect(tmp_path, golden, FakeRag())
    score_run(tmp_path, golden, tmp_path / "out")
    report = (tmp_path / "out" / "rag.md").read_text()
    assert "| dense | n/a (chosen on dev) |" in report


def test_missing_searches_are_reported_not_scored(tmp_path, golden):
    collect(tmp_path, golden, FakeRag(fail={("Where is the cache?", "dense")}))
    score_run(tmp_path, golden, tmp_path / "out")
    report = (tmp_path / "out" / "rag.md").read_text()
    assert "**1 question-mode pairs have no successful search**" in report
    assert "| dense | 3 |" in report  # 3 answerable questions left in dense mode


# --- the report ---------------------------------------------------------------------


def test_score_from_records_reproduces_score_from_the_run(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    first, second = tmp_path / "first", tmp_path / "second"
    score_run(tmp_path, golden, first)
    assert (
        rag_eval.main(
            ["score", "--records", str(first / "rag-records.jsonl"), "--out", str(second)]
        )
        == 0
    )
    names = ["rag.md", "rag.json", "refusal_dense.svg", "refusal_hybrid.svg"]
    for name in names:
        assert (second / name).read_bytes() == (first / name).read_bytes(), name
    assert not (second / "rag-records.jsonl").exists()  # the input isn't copied


def test_the_refusal_charts_are_well_formed_svg(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    score_run(tmp_path, golden, tmp_path / "out")
    for mode in ("dense", "hybrid"):
        root = ET.parse(tmp_path / "out" / f"refusal_{mode}.svg").getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.get("aria-label").startswith(f"Refusal gate, {mode}")


def test_the_committed_fixture_rescores_byte_for_byte(tmp_path):
    records = FIXTURE / "rag-records.jsonl"
    assert rag_eval.main(["score", "--records", str(records), "--out", str(tmp_path)]) == 0
    for name in ["rag.md", "rag.json", "refusal_dense.svg", "refusal_hybrid.svg"]:
        assert (tmp_path / name).read_bytes() == (FIXTURE / name).read_bytes(), name
