"""The RAG evaluation end to end, against a fake RAG service and a scripted judge.

The fake answers ``/v1/config``, ``/v1/search`` and ``/v1/ask`` over
``httpx.MockTransport`` with canned hits and answers for a 6-question golden set, and
the judge's verdicts are scripted by answer text, so every number below can be worked
out by hand. The committed fixture under ``tests/fixtures/rag_report/`` is that run's
records file (both stages, two human grades) and the report scored from it;
``python -m rag_eval score --records tests/fixtures/rag_report/rag-records.jsonl``
rewrites the report files in place.
"""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import httpx
import pytest

import rag_eval
import report
from judge_fake import FakeJudge
from llm import LLMError
from schemas import (
    ClaimVerdict,
    CorrectnessVerdict,
    EvidenceItem,
    FaithfulnessVerdict,
    GoldenQuestion,
    Quote,
)

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


def q(id_, question, category, split, *quotes, origin="targeted", notes=""):
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
        notes=notes,
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
    q("q003", "How much does hosting cost?", "no_answer", "test", notes="out-of-scope: pricing."),
    q(
        "q004",
        "Is there a GUI?",
        "no_answer",
        "dev",
        notes="near-miss: a GUI; the docs cover the CLI.",
    ),
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


def context(key: str) -> dict:
    return {
        "chunk_id": f"c-{key}",
        "text": TEXT[key],
        "score": 0.9,
        "metadata": {"source_file": "tool/x.md"},
    }


def answer(text, *, refused_by=None, cost=0.0004, ms=900.0, keys=("A",), supported=()):
    """A /v1/ask response: an answer, or a refusal by the gate (no contexts) or the model."""
    refusal = "I don't know based on the provided documentation."
    return {
        "answer": refusal if refused_by else text,
        "refused": refused_by is not None,
        "refused_by": refused_by,
        "retrieval_confidence": 0.5,
        "contexts": [] if refused_by == "gate" else [context(k) for k in keys],
        "citations": [
            {"index": i + 1, "resolved": True, "supported": v} for i, v in enumerate(supported)
        ],
        "cost_usd": cost,
        "timings_ms": {"total_ms": ms},
    }


# (question, mode) -> the /v1/ask response. Dense answers q003 (no_answer) and refuses
# s002 at the gate and q002 itself; hybrid refuses only q003 and q004, both at the gate.
ASKS = {
    ("Where is the cache?", "dense"): answer("Set TOOL_CACHE_DIR [1]."),
    ("Where is the cache?", "hybrid"): answer(
        "TOOL_CACHE_DIR moves it [1].", supported=(True, False)
    ),
    ("How do I clear it, and what does a bad body get?", "dense"): answer(
        "", refused_by="model", cost=0.0003
    ),
    ("How do I clear it, and what does a bad body get?", "hybrid"): answer(
        "Clean it; 422 [1].", keys=("BC",)
    ),
    ("How much does hosting cost?", "dense"): answer("It costs $5 a month.", keys=("X",)),
    ("How much does hosting cost?", "hybrid"): answer("", refused_by="gate", cost=0.00001, ms=40.0),
    ("Is there a GUI?", "dense"): answer("", refused_by="gate", cost=0.00001, ms=40.0),
    ("Is there a GUI?", "hybrid"): answer("", refused_by="gate", cost=0.00001, ms=40.0),
    ("What is the default port?", "dense"): answer("Port 80 [1].", keys=("D",)),
    ("What is the default port?", "hybrid"): answer("Port 8080 [1].", keys=("D",)),
    ("Which file holds the lock?", "dense"): answer("", refused_by="gate", cost=0.00001, ms=40.0),
    ("Which file holds the lock?", "hybrid"): answer("A lockfile [1].", keys=("E",)),
}
# The scripted judge, by answer text: correctness ratings and per-claim support.
RATINGS = {
    "Set TOOL_CACHE_DIR [1].": 5,
    "TOOL_CACHE_DIR moves it [1].": 5,
    "Clean it; 422 [1].": 4,
    "Port 80 [1].": 3,
    "Port 8080 [1].": 5,
    "A lockfile [1].": 2,
}
SUPPORT = {
    "Set TOOL_CACHE_DIR [1].": [True, True],
    "TOOL_CACHE_DIR moves it [1].": [True],
    "Clean it; 422 [1].": [True, True],
    "It costs $5 a month.": [False],
    "Port 80 [1].": [True, False],
    "Port 8080 [1].": [True],
    "A lockfile [1].": [True, False],
}


def scripted_judge(**overrides) -> FakeJudge:
    def correctness(question, reference, answer):
        return CorrectnessVerdict(reasoning="scripted", rating=RATINGS[answer])

    def faithfulness(contexts, answer):
        claims = [
            ClaimVerdict(claim=f"c{i}", reasoning="r", supported=v)
            for i, v in enumerate(SUPPORT[answer])
        ]
        return FaithfulnessVerdict(claims=claims)

    return FakeJudge(**({"correctness": correctness, "faithfulness": faithfulness} | overrides))


class FakeRag:
    def __init__(self, config=CONFIG, fail=()):
        self.config, self.fail, self.searches, self.asks = config, set(fail), [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/config":
            return httpx.Response(200, json=self.config)
        body = json.loads(request.content)
        if request.url.path == "/v1/ask":
            self.asks.append(body)
            if (body["question"], body["mode"]) in self.fail:
                return httpx.Response(503, json={"detail": "no generation key"})
            return httpx.Response(200, json=ASKS[(body["question"], body["mode"])])
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


def collect_answers(tmp_path, golden, fake, *extra, judge=None, cap="1.00") -> int:
    args = ["collect", "--stage", "answers", "--run-id", "run1", "--golden", str(golden)]
    args += ["--runs", str(tmp_path / "runs"), "--max-cost-usd", cap, *extra]
    judge = judge if judge is not None else scripted_judge()
    return rag_eval.main(args, rag_client=client(fake), judge=judge)


def score_run(tmp_path, golden, out, run_id="run1", grades=None) -> None:
    # An explicit --grades path, so a real golden/human_grades.jsonl never leaks in.
    grades = grades or tmp_path / "no-grades.jsonl"
    args = ["score", "--run-id", run_id, "--golden", str(golden), "--grades", str(grades)]
    args += ["--runs", str(tmp_path / "runs"), "--out", str(out)]
    assert rag_eval.main(args) == 0


# --- collect ------------------------------------------------------------------------


def test_collect_records_the_config_and_every_search(tmp_path, golden):
    fake = FakeRag()
    assert collect(tmp_path, golden, fake) == 0
    run_dir = tmp_path / "runs" / "run1"
    assert json.loads((run_dir / "run.json").read_text())["stages"]["retrieval"]["config"] == CONFIG
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
    assert header["golden"]["rows"] == 6 and header["retrieval"]["missing"] == 0
    assert header["answers"] is None  # no answer stage in this run
    s002 = next(r for r in rows if r["id"] == "s002" and r["mode"] == "dense")
    assert s002["hits"][0]["covered"] == [0]
    assert s002["hits"][0]["covered_300"] == [0]
    assert s002["hits"][0]["covered_150"] == []  # the quote starts past character 150
    q003 = next(r for r in rows if r["id"] == "q003" and r["mode"] == "hybrid")
    assert (q003["n_items"], q003["top_score"]) == (0, 0.10)


def test_retrieval_scores_by_mode(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    dense, hybrid = (
        scores["retrieval"]["dense"]["retrieval"],
        scores["retrieval"]["hybrid"]["retrieval"],
    )
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
    snippets = scores["retrieval"]["dense"]["snippets"]  # test split: q001 and s002
    assert snippets["n"] == 2
    assert snippets["R@3 full"] == snippets["R@3 300 chars"] == pytest.approx(1.0)
    assert snippets["R@5 150 chars"] == pytest.approx(0.5)  # s002's quote is cut off


def test_refusal_thresholds_are_chosen_on_dev_and_measured_on_test(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    dense, hybrid = (scores["retrieval"][m]["refusal"] for m in ("dense", "hybrid"))
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
    collect_answers(tmp_path, golden, FakeRag())
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


# --- the answer stage ---------------------------------------------------------------


def answers_scored(tmp_path, golden, grades=None):
    collect_answers(tmp_path, golden, FakeRag())
    records = rag_eval.build_records(tmp_path / "runs" / "run1", golden, grades)
    return records, rag_eval.score_records(records)


def test_answers_go_to_ask_with_five_contexts_in_both_modes(tmp_path, golden):
    fake = FakeRag()
    assert collect_answers(tmp_path, golden, fake) == 0
    assert len(fake.asks) == 12
    assert {(a["mode"], a["top_k"]) for a in fake.asks} == {("dense", 5), ("hybrid", 5)}


def test_decided_outcomes_make_no_judge_call(tmp_path, golden):
    judge = scripted_judge()
    collect_answers(tmp_path, golden, FakeRag(), judge=judge)
    asked = Counter(kind for kind, _ in judge.requests)
    # Correctness only for answered, answerable questions: dense q001 and s001; hybrid
    # q001, q002, s001 and s002. Faithfulness for every answer: those 6 plus dense q003.
    assert asked == {"correctness": 6, "faithfulness": 7}
    judged = [args[2] for kind, args in judge.requests if kind == "correctness"]
    assert "It costs $5 a month." not in judged  # a no_answer question is never graded


def test_answer_scores(tmp_path, golden):
    _, scores = answers_scored(tmp_path, golden)
    dense, hybrid = scores["answers"]["modes"]["dense"], scores["answers"]["modes"]["hybrid"]
    # Dense: q001 passes (5), q002 refused by the model (fail), q003 answered a no_answer
    # question (fail), q004 refused it (correct), s001 rated 3 (fail), s002 refused (fail).
    assert (dense["correct"]["k"], dense["correct"]["n"]) == (2, 6)
    # Hybrid: every one correct except s002, rated 2.
    assert (hybrid["correct"]["k"], hybrid["correct"]["n"]) == (5, 6)
    assert dense["mean_rating"] == pytest.approx(4.0)  # (5 + 3) / 2
    # Faithfulness: dense q001 2/2, q003 0/1, s001 1/2 -> mean 0.5; fully supported 1 of 3.
    assert dense["faithfulness"] == pytest.approx(0.5)
    assert dense["fully_supported"]["k"] == 1 and dense["fully_supported"]["n"] == 3
    # Hybrid: 1, 1, 1 and 1/2 -> 0.875.
    assert hybrid["faithfulness"] == pytest.approx(0.875)
    # Only hybrid q001 was verified by the pipeline: 1 of its 2 citations held.
    assert hybrid["citation_accuracy_self"] == 0.5 and dense["citation_accuracy_self"] is None
    # Dense cost: 2 answers at $0.0004, the model refusal $0.0003, 2 gate refusals at
    # $0.00001 and q003 at $0.0004 -> $0.00152 over 6.
    assert dense["cost_per_query"] == pytest.approx(0.00152 / 6)
    assert (dense["p50_ms"], dense["p95_ms"]) == (900.0, 900.0)
    assert scores["answers"]["modes"]["dense"]["by_category"]["no_answer"]["correct"]["k"] == 1


def test_refusals_end_to_end_count_the_gate_and_the_model(tmp_path, golden):
    _, scores = answers_scored(tmp_path, golden)
    dense = scores["answers"]["modes"]["dense"]["refusal"]
    hybrid = scores["answers"]["modes"]["hybrid"]["refusal"]
    # Test split: q001, q003 (no_answer) and s002. Dense answers q003 and refuses s002.
    assert (dense["tp"], dense["fp"], dense["fn"], dense["tn"]) == (0, 1, 1, 1)
    assert (dense["by_gate"], dense["by_model"]) == (1, 0)
    # Hybrid refuses q003 at the gate and answers the other two.
    assert (hybrid["tp"], hybrid["fp"], hybrid["fn"], hybrid["tn"]) == (1, 0, 0, 2)


def test_the_cap_covers_asks_and_stops_before_one_could_pass_it(tmp_path, golden):
    # gpt-4o-mini reserves 3000 x $0.15/M + 1000 x $0.60/M = $0.00105 per ask. With a
    # $0.0015 cap: ask 1 (spent $0.0004), ask 2 ($0.0008), then $0.0008 + $0.00105 > cap.
    fake = FakeRag()
    assert collect_answers(tmp_path, golden, fake, cap="0.0015") == 1
    assert len(fake.asks) == 2
    resumed = FakeRag()
    assert collect_answers(tmp_path, golden, resumed) == 0
    assert len(resumed.asks) == 10  # the 2 already paid for aren't asked again


def test_the_worst_case_counts_the_citation_checks():
    one = rag_eval.ask_worst_case_usd(CONFIG)
    assert one == pytest.approx(3000 * 0.15e-6 + 1000 * 0.60e-6)
    assert rag_eval.ask_worst_case_usd({**CONFIG, "citation_verification": True}) == pytest.approx(
        6 * one
    )


def test_an_answer_missing_a_verdict_is_judged_again_without_asking_again(tmp_path, golden):
    def flaky(contexts, answer):
        raise LLMError("the reply hit max_tokens")

    fake = FakeRag()
    assert collect_answers(tmp_path, golden, fake, judge=scripted_judge(faithfulness=flaky)) == 1
    records = rag_eval.build_records(tmp_path / "runs" / "run1", golden)
    assert records[0]["answers"]["unjudged"] == 7
    again, judge = FakeRag(), scripted_judge()
    assert collect_answers(tmp_path, golden, again, judge=judge) == 0
    assert again.asks == []
    assert {kind for kind, _ in judge.requests} == {"faithfulness"}


def test_a_judge_from_the_generators_provider_needs_a_flag(tmp_path, golden):
    same = scripted_judge()
    same.provider = "openai"  # the fake service's generator is OpenAI's
    with pytest.raises(SystemExit, match="allow-same-provider-judge"):
        collect_answers(tmp_path, golden, FakeRag(), judge=same)
    assert (
        collect_answers(tmp_path, golden, FakeRag(), "--allow-same-provider-judge", judge=same) == 0
    )
    score_run(tmp_path, golden, tmp_path / "out")
    assert (
        "**The judge comes from the generator's provider.**"
        in (tmp_path / "out" / "rag.md").read_text()
    )


def test_the_answer_stage_needs_a_cap(tmp_path, golden):
    args = ["collect", "--stage", "answers", "--run-id", "r", "--golden", str(golden)]
    with pytest.raises(SystemExit):
        rag_eval.main(args, rag_client=client(FakeRag()), judge=scripted_judge())


def test_failed_asks_are_errors_not_scores(tmp_path, golden):
    fake = FakeRag(fail={("Where is the cache?", "dense"), ("Is there a GUI?", "hybrid")})
    assert collect_answers(tmp_path, golden, fake) == 1
    records, scores = answers_scored(tmp_path, golden)  # resumes: retries both, which pass now
    assert records[0]["answers"]["missing"] == 0
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    collect_answers(fresh, golden, FakeRag(fail={("Where is the cache?", "dense")}))
    score_run(fresh, golden, fresh / "out")
    report = (fresh / "out" / "rag.md").read_text()
    assert "**1 question-mode pairs have no successful answer**" in report
    assert "| dense | 5 |" in report


def test_kappa_against_human_grades_of_the_same_answers(tmp_path, golden):
    run = tmp_path / "runs" / "run1" / "answers.jsonl"
    collect_answers(tmp_path, golden, FakeRag())
    texts = {
        (r["id"], r["mode"]): r["answer"] for r in map(json.loads, run.read_text().splitlines())
    }
    # The judge passes dense q001, fails dense s001 (3), passes hybrid q001, q002 and s001,
    # fails hybrid s002 (2). The human disagrees on dense s001 and hybrid q002:
    # p_o = 4/6; both pass 4 of 6, so p_e = (4/6)^2 + (2/6)^2 = 5/9;
    # kappa = (2/3 - 5/9) / (1 - 5/9) = (1/9) / (4/9) = 0.25.
    human = {
        ("q001", "dense"): True,
        ("s001", "dense"): True,
        ("q001", "hybrid"): True,
        ("q002", "hybrid"): False,
        ("s001", "hybrid"): True,
        ("s002", "hybrid"): False,
    }
    grades = tmp_path / "grades.jsonl"
    rows = [
        {
            "id": i,
            "mode": m,
            "answer_sha": rag_eval.answer_sha(texts[(i, m)]),
            "pass": v,
            "note": "",
        }
        for (i, m), v in human.items()
    ]
    # A grade of a different answer to the same question doesn't count.
    rows.append(
        {
            "id": "q001",
            "mode": "dense",
            "answer_sha": rag_eval.answer_sha("old"),
            "pass": False,
            "note": "",
        }
    )
    grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, scores = answers_scored(tmp_path, golden, grades)
    agreement = scores["answers"]["agreement"]
    assert agreement["n"] == 6
    assert agreement["raw"] == pytest.approx(4 / 6)
    assert agreement["kappa"] == pytest.approx(0.25)


# --- the threshold to configure -----------------------------------------------------


def test_the_configured_value_truncates_without_moving_a_decision():
    # The hybrid threshold chosen on the published run sits on q017's score: rounding it
    # to 0.9636 would refuse q017; truncating to 0.963599 keeps every decision.
    chosen, scores = 0.9635999851538614, [0.9635999851538614, 0.98, 0.5]
    assert rag_eval.config_value(chosen, scores) == 0.963599
    assert [s < 0.9636 for s in scores] != [s < chosen for s in scores]


def test_the_configured_value_takes_more_digits_when_a_score_sits_in_between():
    chosen, scores = 0.5000005, [0.5000005, 0.5000002, 0.4]
    value = rag_eval.config_value(chosen, scores)
    assert value != 0.5  # truncating to 6 decimals would keep 0.5000002 too
    assert [s < value for s in scores] == [s < chosen for s in scores]
    assert value <= chosen


def test_reports_print_the_value_to_configure_and_keep_thresholds_exact(tmp_path):
    assert report._rounded({"chosen_threshold": 0.9635999851538614, "f1": 0.12345678}) == {
        "chosen_threshold": 0.9635999851538614,
        "f1": 0.123457,
    }
    assert report.threshold_text(0.963599) == "0.963599"
    assert report.threshold_text(0.3) == "0.3"


# --- no_answer kinds ----------------------------------------------------------------


def test_the_kind_comes_from_the_notes_and_travels_in_the_records(tmp_path, golden):
    assert rag_eval.no_answer_kind(GOLDEN[2]) == "out-of-scope"
    assert rag_eval.no_answer_kind(GOLDEN[0]) is None  # answerable
    unlabelled = GOLDEN[3].model_copy(update={"notes": "No prefix here."})
    assert rag_eval.no_answer_kind(unlabelled) == "unlabelled"
    records, _ = scored(tmp_path, golden)
    kinds = {(r["id"], r["mode"]): r["no_answer_kind"] for r in records[1:]}
    assert kinds[("q003", "dense")] == "out-of-scope" and kinds[("q001", "dense")] is None


def test_gate_refusals_by_kind(tmp_path, golden):
    _, scores = scored(tmp_path, golden)
    dense = scores["retrieval"]["dense"]["refusal"]["by_kind"]
    hybrid = scores["retrieval"]["hybrid"]["refusal"]["by_kind"]
    # Dense: q004 (near-miss, dev) scores 0.50 < 0.55; q003 (out-of-scope, test) 0.58 isn't,
    # nor below the configured 0.3.
    assert list(dense) == ["near-miss", "out-of-scope"]
    assert dense["near-miss"]["dev"] == {"n": 1, "refused_chosen": 1}
    assert dense["out-of-scope"]["test"] == {"n": 1, "refused_chosen": 0, "refused_configured": 0}
    # Hybrid: q003 scores 0.10, below both 0.9 and 0.3.
    assert hybrid["out-of-scope"]["test"] == {"n": 1, "refused_chosen": 1, "refused_configured": 1}
    assert hybrid["out-of-scope"]["median_top_score"] == 0.10


def test_who_declined_each_no_answer_question(tmp_path, golden):
    _, scores = answers_scored(tmp_path, golden)
    dense = scores["answers"]["modes"]["dense"]["no_answer_by_kind"]
    # Dense answered q003 (out-of-scope, test) and refused q004 (near-miss, dev) at the gate.
    assert dense["out-of-scope"]["test"] == {"n": 1, "gate": 0, "model": 0, "answered": 1}
    assert dense["near-miss"]["dev"] == {"n": 1, "gate": 1, "model": 0, "answered": 0}
    assert dense["near-miss"]["test"]["n"] == 0


# --- one config snapshot per stage --------------------------------------------------

CALIBRATED = {**CONFIG, "thresholds": {"dense": 0.55, "hybrid": 0.9}, "citation_verification": True}


def run_json(tmp_path) -> dict:
    return json.loads((tmp_path / "runs" / "run1" / "run.json").read_text())


def test_each_stage_keeps_its_own_snapshot(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    retrieval_before = run_json(tmp_path)["stages"]["retrieval"]
    # The answer stage runs with calibrated thresholds and citation checks on.
    assert collect_answers(tmp_path, golden, FakeRag(config=CALIBRATED)) == 0
    stages = run_json(tmp_path)["stages"]
    assert stages["retrieval"] == retrieval_before  # untouched
    assert stages["answers"]["config"] == CALIBRATED
    records = rag_eval.build_records(tmp_path / "runs" / "run1", golden)
    assert records[0]["retrieval"]["config"]["thresholds"] == {"dense": 0.3, "hybrid": 0.3}
    assert records[0]["answers"]["config"]["thresholds"] == {"dense": 0.55, "hybrid": 0.9}
    # The gate section compares with the threshold the retrieval stage ran with.
    scores = rag_eval.score_records(records)
    assert scores["retrieval"]["hybrid"]["refusal"]["configured_threshold"] == 0.3
    score_run(tmp_path, golden, tmp_path / "out")
    report_md = (tmp_path / "out" / "rag.md").read_text()
    assert "| refusal thresholds (dense / hybrid) | 0.3 / 0.3 | 0.55 / 0.9 |" in report_md
    assert "| citation verification | off | on |" in report_md


def test_a_stage_resumes_only_against_its_own_snapshot(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    collect_answers(tmp_path, golden, FakeRag(config=CALIBRATED))
    with pytest.raises(SystemExit, match="answers stage"):
        collect_answers(tmp_path, golden, FakeRag(config=CONFIG))
    again = FakeRag(config=CONFIG)
    assert collect(tmp_path, golden, again) == 0  # retrieval still matches its own
    assert again.searches == []


def test_every_stage_measures_the_same_collection(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    for change in (
        {"collection": "other"},
        {"embedding_model": "other-model"},
        {"index": {**CONFIG["index"], "chunk_size": 400}},
    ):
        with pytest.raises(SystemExit, match="start a new run"):
            collect_answers(tmp_path, golden, FakeRag(config={**CONFIG, **change}))
    assert "answers" not in run_json(tmp_path)["stages"]


def test_runs_recorded_before_per_stage_snapshots_still_work(tmp_path, golden):
    collect(tmp_path, golden, FakeRag())
    path = tmp_path / "runs" / "run1" / "run.json"
    legacy = {"run_id": "run1", "started_at": "2026-10-01T10:00:00+00:00", "config": CONFIG}
    path.write_text(json.dumps(legacy, indent=2) + "\n")
    before = path.read_bytes()
    assert collect(tmp_path, golden, FakeRag()) == 0  # resuming retrieval rewrites nothing
    assert path.read_bytes() == before
    assert collect_answers(tmp_path, golden, FakeRag(config=CALIBRATED)) == 0
    stages = run_json(tmp_path)["stages"]
    assert stages["retrieval"] == {"started_at": legacy["started_at"], "config": CONFIG}
    assert stages["answers"]["config"] == CALIBRATED
    # A records file written before per-stage snapshots still scores and renders.
    records = rag_eval.build_records(tmp_path / "runs" / "run1", golden)
    for stage in ("retrieval", "answers"):
        del records[0][stage]["config"]
    report.rag_markdown(records[0], rag_eval.score_records(records))


def test_ids_pick_the_questions_for_a_pilot(tmp_path, golden):
    fake = FakeRag()
    assert collect_answers(tmp_path, golden, fake, "--ids", "q001,q003") == 0
    assert sorted({a["question"] for a in fake.asks}) == [
        "How much does hosting cost?",
        "Where is the cache?",
    ]
    assert len(fake.asks) == 4
    with pytest.raises(SystemExit):
        collect_answers(tmp_path, golden, FakeRag(), "--ids", "q001,q999")
