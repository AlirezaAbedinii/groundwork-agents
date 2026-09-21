"""Phase 4 tests for the eval metrics + harness (judge mocked, no network)."""
from __future__ import annotations

from eval.metrics import (
    parse_rating,
    refusal_correctness,
    score_citation_accuracy,
    score_citation_accuracy_judged,
    score_correctness,
    score_faithfulness,
    score_retrieval_relevance,
)
from eval.run_eval import evaluate, load_golden

from rag.config import Settings
from rag.generation.citations import Citation
from rag.generation.llm_client import ChatResult
from rag.indexing.vector_store import ScoredChunk
from rag.observability.metrics import TokenUsage
from rag.pipeline import AnswerResult


class ScriptedJudge:
    """Returns a fixed rating string; records the last prompt it graded."""

    def __init__(self, rating_text: str) -> None:
        self.rating_text = rating_text
        self.calls = 0

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        self.last_user = user
        return ChatResult(self.rating_text, TokenUsage(10, 2))


# --- rating parser --------------------------------------------------------
def test_parse_rating_explicit() -> None:
    assert parse_rating("Rating: 5\nReason: perfect") == 5
    assert parse_rating("score = 3") == 3


def test_parse_rating_fallback_and_failclosed() -> None:
    assert parse_rating("I'd say a 4 overall") == 4
    assert parse_rating("no number here") == 1  # fails closed to worst


# --- correctness ----------------------------------------------------------
def test_correctness_high_when_judge_rates_high() -> None:
    judge = ScriptedJudge("Rating: 5")
    r = score_correctness("Q?", "expected", "answer", judge)
    assert r.name == "correctness"
    assert r.rating == 5 and r.score == 1.0 and r.passed is True
    # The reference answer is actually shown to the judge.
    assert "expected" in judge.last_user


def test_correctness_low_when_judge_rates_low() -> None:
    r = score_correctness("Q?", "expected", "wrong", ScriptedJudge("Rating: 2"))
    assert r.score == 0.25 and r.passed is False


# --- faithfulness ---------------------------------------------------------
def test_faithfulness_scores_and_sees_context() -> None:
    judge = ScriptedJudge("Rating: 4")
    r = score_faithfulness("the answer", "the retrieved context", judge)
    assert r.name == "faithfulness" and r.passed is True
    assert "the retrieved context" in judge.last_user


# --- no_answer refusal check (deterministic, no judge) --------------------
def test_refusal_correctness_rewards_refusing() -> None:
    assert refusal_correctness(refused=True).passed is True
    assert refusal_correctness(refused=False).passed is False
    assert refusal_correctness(refused=True).rating is None  # no judge call


# --- retrieval relevance (deterministic) -----------------------------------
def _chunk_from(source: str) -> ScoredChunk:
    return ScoredChunk("c1", "text", 0.9, {"source_file": source})


def test_retrieval_relevance_hit_and_miss() -> None:
    contexts = [_chunk_from("04-error-codes.md"), _chunk_from("01-overview.md")]
    hit = score_retrieval_relevance(["04-error-codes.md"], contexts)
    assert hit.passed is True and hit.score == 1.0

    miss = score_retrieval_relevance(["05-rate-limits.md"], contexts)
    assert miss.passed is False and miss.score == 0.0


def test_retrieval_relevance_handles_file_section_form() -> None:
    # SCHEMA allows "file#section"; matching is on the bare filename.
    contexts = [_chunk_from("03-configuration.md")]
    r = score_retrieval_relevance(["03-configuration.md#worker-settings"], contexts)
    assert r.passed is True


def test_retrieval_relevance_none_for_no_answer_records() -> None:
    assert score_retrieval_relevance([], [_chunk_from("a.md")]) is None


# --- citation accuracy (from pipeline verification) --------------------------
def _cit(index: int, resolved: bool = True, supported: bool | None = None) -> Citation:
    return Citation(index=index, resolved=resolved, chunk_id=f"c{index}", supported=supported)


def test_citation_accuracy_share_of_supported() -> None:
    r = score_citation_accuracy([_cit(1, supported=True), _cit(2, supported=False)])
    assert r.score == 0.5 and r.passed is False
    perfect = score_citation_accuracy([_cit(1, supported=True)])
    assert perfect.score == 1.0 and perfect.passed is True


def test_citation_accuracy_unresolved_counts_against() -> None:
    # Citing a chunk that was never retrieved is an accuracy failure.
    r = score_citation_accuracy([_cit(1, supported=True), _cit(9, resolved=False)])
    assert r.score == 0.5


def test_citation_accuracy_none_when_unjudged_or_empty() -> None:
    assert score_citation_accuracy([]) is None
    assert score_citation_accuracy([_cit(1, supported=None)]) is None


def test_citation_accuracy_name_is_overridable_for_the_self_tally() -> None:
    r = score_citation_accuracy([_cit(1, supported=True)], name="citation_accuracy_self")
    assert r.name == "citation_accuracy_self" and r.score == 1.0


# --- citation accuracy re-verified by the eval judge ----------------------
class VerdictJudge:
    """Answers citation checks from a scripted verdict list; records every prompt."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> ChatResult:
        self.prompts.append((system, user))
        return ChatResult(self.verdicts.pop(0), TokenUsage(10, 2))


_CTX = [
    ScoredChunk("c1", "Ferry retries failed jobs three times.", 0.9, {"source_file": "a.md"}),
    ScoredChunk("c2", "The worker timeout defaults to 30 seconds.", 0.8, {"source_file": "b.md"}),
]
_ANSWER = "Ferry retries failed jobs [1]. The timeout is 45 seconds [2]."


def test_judged_all_supported() -> None:
    judge = VerdictJudge(["SUPPORTED", "SUPPORTED"])
    r = score_citation_accuracy_judged(_ANSWER, [_cit(1), _cit(2)], _CTX, judge)
    assert r.name == "citation_accuracy"
    assert r.score == 1.0 and r.passed is True
    assert r.detail == "2/2 citations supported"
    # One judge call per resolved citation, each shown its claim and its cited chunk.
    assert len(judge.prompts) == 2
    assert "Ferry retries failed jobs [1]." in judge.prompts[0][1]
    assert "three times" in judge.prompts[0][1]
    assert "45 seconds [2]" in judge.prompts[1][1]
    assert "30 seconds" in judge.prompts[1][1]


def test_judged_all_unsupported_ignores_pipeline_flags() -> None:
    # The pipeline marked both supported; the independent judge disagrees.
    cits = [_cit(1, supported=True), _cit(2, supported=True)]
    r = score_citation_accuracy_judged(_ANSWER, cits, _CTX, VerdictJudge(["UNSUPPORTED"] * 2))
    assert r.score == 0.0 and r.passed is False
    # ...and the self tally still reports the pipeline's own view, so the gap is visible.
    assert score_citation_accuracy(cits, name="citation_accuracy_self").score == 1.0
    # The input citations are not mutated by re-verification.
    assert [c.supported for c in cits] == [True, True]


def test_judged_mixed_counts_unresolved_against_and_skips_judging_them() -> None:
    cits = [_cit(1), _cit(2), _cit(9, resolved=False)]
    judge = VerdictJudge(["SUPPORTED", "UNSUPPORTED — the source says 30 seconds"])
    r = score_citation_accuracy_judged(_ANSWER, cits, _CTX, judge)
    assert r.score == 1 / 3 and r.passed is False
    assert r.detail == "1/3 citations supported"
    assert len(judge.prompts) == 2  # the unresolved [9] has no chunk to check


def test_judged_fails_closed_on_a_non_verdict_reply() -> None:
    # A rating-style reply (what the other metrics expect) is not a verdict.
    r = score_citation_accuracy_judged(_ANSWER, [_cit(1)], _CTX, VerdictJudge(["Rating: 5"]))
    assert r.score == 0.0


def test_judged_none_without_citations_and_makes_no_call() -> None:
    judge = VerdictJudge([])
    assert score_citation_accuracy_judged("no citations here", [], _CTX, judge) is None
    assert judge.prompts == []


# --- harness aggregation --------------------------------------------------
class _Answerer:
    """Maps question -> canned AnswerResult (mirrors pipeline output)."""

    def __init__(self, mapping: dict[str, AnswerResult]) -> None:
        self.mapping = mapping

    def answer(self, question: str) -> AnswerResult:
        return self.mapping[question]


def _answered(q: str, text: str) -> AnswerResult:
    ctx = ScoredChunk("c1", "ctx", 0.9, {"source_file": "d.md"})
    return AnswerResult(
        question=q, answer=text, mode="dense", refused=False,
        retrieval_confidence=0.9, contexts=[ctx], cost_usd=0.001,
        timings_ms={"total_ms": 4.0},
    )


def _refused(q: str) -> AnswerResult:
    return AnswerResult(
        question=q, answer="I don't know", mode="dense", refused=True,
        retrieval_confidence=0.0, timings_ms={"total_ms": 1.0},
    )


def test_evaluate_aggregates_and_skips_faithfulness_on_refusal(tmp_path) -> None:
    from eval.run_eval import GoldenRecord

    records = [
        GoldenRecord("q1", "lookup q", "exp", ["a.md"], "lookup"),
        GoldenRecord("q2", "missing q", "", [], "no_answer"),
    ]
    answerer = _Answerer(
        {"lookup q": _answered("lookup q", "good [1]"), "missing q": _refused("missing q")}
    )
    judge = ScriptedJudge("Rating: 5")

    report = evaluate(records, answerer, judge)

    assert report.n == 2
    assert report.aggregates["answered"] == 1
    assert report.aggregates["refused"] == 1
    # Refused record contributes correctness (refusal-based) but not faithfulness.
    faith = [r.faithfulness for r in report.records]
    assert faith[0] is not None and faith[1] is None
    # no_answer correctness is judged deterministically (no judge call for it);
    # only the answered lookup triggers correctness + faithfulness judging = 2 calls.
    assert judge.calls == 2
    assert report.per_category["no_answer"]["correctness_mean"] == 1.0


class _RoutingJudge:
    """Rates every rating prompt 5; answers citation checks from a verdict list."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        text = self.verdicts.pop(0) if "UNSUPPORTED" in system else "Rating: 5"
        return ChatResult(text, TokenUsage(10, 2))


def test_evaluate_reports_judge_and_self_citation_accuracy_side_by_side() -> None:
    import json

    from eval.run_eval import GoldenRecord

    answer = AnswerResult(
        question="q", answer=_ANSWER, mode="hybrid", refused=False,
        retrieval_confidence=0.9, contexts=_CTX, cost_usd=0.001,
        citations=[_cit(1, supported=True), _cit(2, supported=True)],  # pipeline: 2/2
        timings_ms={"total_ms": 4.0},
    )
    judge = _RoutingJudge(["SUPPORTED", "UNSUPPORTED"])  # eval judge: 1/2

    records = [GoldenRecord("q1", "q", "exp", ["a.md"], "lookup")]
    report = evaluate(records, _Answerer({"q": answer}), judge)

    rec = report.records[0]
    assert rec.citation_accuracy.score == 0.5 and rec.citation_accuracy_self.score == 1.0
    assert report.aggregates["citation_accuracy_mean"] == 0.5
    assert report.aggregates["citation_accuracy_self_mean"] == 1.0
    assert report.aggregates["citation_accuracy_n"] == 1
    assert report.aggregates["citation_accuracy_self_n"] == 1
    # correctness + faithfulness + one re-verification per citation.
    assert judge.calls == 4
    # Both land in the JSON report.
    dumped = json.loads(report.to_json())["records"][0]
    assert dumped["citation_accuracy"]["score"] == 0.5
    assert dumped["citation_accuracy_self"]["score"] == 1.0


def test_load_golden_reads_the_real_set() -> None:
    records = load_golden(Settings(_env_file=None).golden_set_path)
    assert len(records) >= 50  # V1 target: 50+ hand-verified questions
    assert {r.category for r in records} == {"lookup", "multi_hop", "no_answer", "ambiguous"}
    assert all(r.id and r.question and r.category for r in records)
    assert len({r.id for r in records}) == len(records)  # ids stay unique
    # no_answer records must have no supporting sources (SCHEMA rule).
    assert all(not r.supporting_sources for r in records if r.category == "no_answer")
