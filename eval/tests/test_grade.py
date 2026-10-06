"""The blind grading tool: what it samples, what it hides, what it records."""

import json

import pytest

from dataset.grade import allocate, grade, judged_answers, sample
from rag_eval import answer_sha
from schemas import EvidenceItem, GoldenQuestion, Quote

SECRET = "SECRET-REASONING"


def golden_row(qid: str, category: str) -> GoldenQuestion:
    n_items = {"lookup": 1, "multi_hop": 2, "no_answer": 0}[category]
    evidence = [
        EvidenceItem(quotes=[Quote(source="tool/x.md", text=f"quote number {i} here")])
        for i in range(n_items)
    ]
    return GoldenQuestion(
        id=qid,
        question=f"Question {qid}?",
        reference_answer=f"Reference {qid}.",
        category=category,
        evidence=evidence,
        origin="targeted",
        split="test",
        verified=True,
    )


def judged(qid: str, mode: str, rating: int = 5) -> dict:
    verdict = {"verdict": {"reasoning": SECRET, "rating": rating}, "provider": "fake"}
    return {
        "id": qid,
        "mode": mode,
        "answer": f"Answer to {qid}, {mode[0]}.",
        "refused": False,
        "correctness": verdict,
    }


@pytest.fixture
def run(tmp_path):
    """16 judged answers: per mode, 6 lookup and 2 multi_hop; plus 3 that weren't judged."""
    lookups = [f"q{i:03}" for i in range(1, 7)]
    multis = ["q011", "q012"]
    rows = [golden_row(q, "lookup") for q in lookups] + [golden_row(q, "multi_hop") for q in multis]
    rows.append(golden_row("q020", "no_answer"))
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    records = [judged(q, mode) for mode in ("dense", "hybrid") for q in lookups + multis]
    records += [
        {
            "id": "q020",
            "mode": "dense",
            "answer": "Made up.",
            "refused": False,
            "correctness": None,
            "faithfulness": {},
        },  # a no_answer question is never rated
        {"id": "q001", "mode": "other", "error": "HTTPStatusError: 503"},
        {"id": "q002", "mode": "other", "answer": "I don't know.", "refused": True},
    ]
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir, golden


def test_only_answers_the_judge_rated_are_candidates(run):
    items = judged_answers(*run)
    assert len(items) == 16
    assert {i["mode"] for i in items} == {"dense", "hybrid"}
    assert all(i["answer_sha"] == answer_sha(i["answer"]) for i in items)


def test_allocate_splits_in_proportion_with_largest_remainders():
    assert allocate({"a": 6, "b": 2, "c": 6, "d": 2}, 8) == {"a": 3, "b": 1, "c": 3, "d": 1}
    # 30 of 10 + 7 + 3: exact 15, 10.5, 4.5 -> 15, 10, 4 with one left; the .5 tie goes
    # to the first key in order.
    assert allocate({"a": 20, "b": 14, "c": 6}, 30) == {"a": 15, "b": 11, "c": 4}
    assert allocate({"a": 2, "b": 1}, 5) == {"a": 2, "b": 1}  # fewer than asked: all


def test_the_sample_is_stratified_and_repeatable(run):
    items = judged_answers(*run)
    picked = sample(items, 8, seed=0)
    strata = sorted((i["mode"], i["category"]) for i in picked)
    assert strata.count(("dense", "lookup")) == 3 and strata.count(("hybrid", "multi_hop")) == 1
    assert sample(items, 8, seed=0) == picked
    assert sample(items, 8, seed=1) != picked


def test_grading_is_blind_and_records_the_answer_it_saw(run, tmp_path):
    items = sample(judged_answers(*run), 4, seed=0)
    keys = iter(["p", "fine", "f", "", "x", "s", "q"])
    shown: list[str] = []
    grades = tmp_path / "grades.jsonl"
    tally = grade(items, grades, "r1", ask=lambda _: next(keys), out=shown.append)
    assert tally == {"passed": 1, "failed": 1, "skipped": 1}
    text = "\n".join(shown)
    assert SECRET not in text and "rating" not in text
    assert "dense" not in text and "hybrid" not in text
    rows = [json.loads(line) for line in grades.read_text().splitlines()]
    assert [(r["id"], r["pass"], r["note"]) for r in rows] == [
        (items[0]["id"], True, "fine"),
        (items[1]["id"], False, ""),
    ]
    assert rows[0]["answer_sha"] == items[0]["answer_sha"] and rows[0]["run_id"] == "r1"


def test_resuming_skips_what_was_graded(run, tmp_path):
    items = sample(judged_answers(*run), 4, seed=0)
    grades = tmp_path / "grades.jsonl"
    first = iter(["p", "", "q"])
    grade(items, grades, "r1", ask=lambda _: next(first), out=lambda _: None)
    shown: list[str] = []
    grade(items, grades, "r1", ask=lambda _: "q", out=shown.append)
    assert shown[0].startswith("1 of 4 already graded")
    assert f"== 1/3  {items[1]['id']}" in shown
