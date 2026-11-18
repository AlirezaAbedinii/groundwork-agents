"""The retrieval ablation end to end, against fake RAG services, one per arm.

Two arms (dense, the reference, and hybrid) search a five-question golden set whose
hits are canned per arm, so every metric and difference below is worked out by hand.
"""

import json
import math
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import metrics
import rag_eval
from schemas import AblationPlan, EvidenceItem, GoldenQuestion, Quote

QUOTE_A = "Set `TOOL_CACHE_DIR` to move the cache"
QUOTE_B = "Run `tool cache clean` to remove"
QUOTE_C = "an invalid body gets a 422 response"
QUOTE_D = "listens on port 8080 by default"
TEXT = {
    "A": f"{QUOTE_A}, or pass `--cache-dir`.",
    "B": f"{QUOTE_B} every cached archive.",
    "C": f"Requests are validated: {QUOTE_C}.",
    "D": f"The server {QUOTE_D}.",
    "X": "Unrelated text about logging.",
}


def q(id_, split, category, *quotes, phrasing=None):
    evidence = [EvidenceItem(quotes=[Quote(source="tool/x.md", text=t)]) for t in quotes]
    return GoldenQuestion(
        id=id_,
        question=f"question {id_}?",
        reference_answer="ref",
        category=category,
        evidence=evidence,
        origin="targeted",
        split=split,
        verified=True,
        phrasing=phrasing,
    )


GOLDEN = [
    q("q001", "dev", "lookup", QUOTE_A, phrasing="exact_token"),
    q("q002", "dev", "lookup", QUOTE_B, phrasing="paraphrase"),
    q("q003", "dev", "multi_hop", QUOTE_A, QUOTE_C, phrasing="exact_token"),
    q("q004", "dev", "no_answer"),
    q("q005", "test", "lookup", QUOTE_D, phrasing="paraphrase"),
    q("q006", "test", "ambiguous", QUOTE_A, QUOTE_D),
]

HYBRID_SETTINGS = {
    "sparse_enabled": True,
    "rerank_enabled": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_candidates": 20,
    "rrf_k": 60,
    "rrf_dense_weight": 0.7,
    "rrf_sparse_weight": 0.3,
}
CONFIG = {
    "collection": "fixture",
    "embedding_model": "fake-embedder",
    "default_mode": "hybrid",
    "top_k": 10,
    "hybrid": HYBRID_SETTINGS,
    "index": {"chunk_strategy": "fixed", "chunk_size": 800, "chunk_overlap": 120},
}
# Each arm's hits per question (chunk keys, best first).
HITS = {
    "A0": {
        "q001": "XA",  # A at rank 2
        "q002": "B",
        "q003": "AX",  # A at 1; C never
        "q005": "X",
        "q006": "DA",
    },
    "A1": {
        "q001": "A",
        "q002": "XB",
        "q003": "AC",
        "q005": "D",
        "q006": "D",
    },
}
RERANK_MS = {"q001": 100.0, "q002": 300.0, "q003": 200.0, "q005": 150.0, "q006": 250.0}
NDCG_2ND = 1 / math.log2(3)  # nDCG@10 of one item found at rank 2
NDCG_ONE_OF_TWO = 1 / (1 + 1 / math.log2(3))  # one of two items, at rank 1


class FakeArmRag:
    def __init__(self, arm: str, mode: str, config: dict = CONFIG):
        self.arm, self.mode, self.config = arm, mode, config
        self.searches: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/config":
            return httpx.Response(200, json=self.config)
        body = json.loads(request.content)
        assert body["mode"] == self.mode and body["top_k"] == 10
        self.searches.append(body["query"])
        qid = body["query"].removeprefix("question ").removesuffix("?")
        keys = HITS[self.arm].get(qid, "")
        hits = [
            {"chunk_id": f"c-{k}", "text": TEXT[k], "score": 0.5, "source_file": "tool/x.md"}
            for k in keys
        ]
        timings = {"embed": 10.0, "dense": 2.0, "total_ms": 12.0}
        if self.mode == "hybrid" and qid in RERANK_MS:
            timings |= {"rerank": RERANK_MS[qid], "total_ms": 12.0 + RERANK_MS[qid]}
        return httpx.Response(200, json={"hits": hits, "timings_ms": timings})


@pytest.fixture
def setup(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(row.model_dump_json() + "\n" for row in GOLDEN))
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "reference": "A0",
                "arms": [
                    {"name": "A0", "change": "dense", "mode": "dense", "rag": "http://dense"},
                    {
                        "name": "A1",
                        "change": "hybrid",
                        "mode": "hybrid",
                        "rag": "http://hybrid",
                        "expect": {
                            "hybrid.rerank_enabled": True,
                            "hybrid.rrf_dense_weight": 0.7,
                            "index.chunk_strategy": "fixed",
                        },
                    },
                ],
            }
        )
    )
    fakes = {"http://dense": FakeArmRag("A0", "dense"), "http://hybrid": FakeArmRag("A1", "hybrid")}

    def connect(url: str) -> httpx.Client:
        return httpx.Client(base_url=url, transport=httpx.MockTransport(fakes[url]))

    def run(*extra: str) -> int:
        argv = ["ablate", "--arms", str(arms), "--run-id", "abl", "--runs", str(tmp_path / "runs")]
        return rag_eval.main([*argv, "--golden", str(golden), *extra], connect=connect)

    return {"run": run, "fakes": fakes, "arms": arms, "out": tmp_path / "runs" / "abl"}


def _scores(out: Path) -> dict:
    return json.loads((out / "ablation.json").read_text())["scores"]


def test_ablate_searches_answerable_dev_questions_after_a_warm_up(setup) -> None:
    assert setup["run"]() == 0

    for fake in setup["fakes"].values():
        # The warm-up loads the service's models before anything is timed; it isn't recorded.
        assert fake.searches == [
            rag_eval.WARM_UP_QUERY,
            "question q001?",
            "question q002?",
            "question q003?",
        ]  # dev only, and no no_answer question: it has no evidence to find
    scores = _scores(setup["out"])
    assert list(scores["splits"]) == ["dev"]
    records = rag_eval.read_jsonl(setup["out"] / "ablation-records.jsonl")
    assert {(r["arm"], r["id"]) for r in records[1:]} == {
        (arm, qid) for arm in ("A0", "A1") for qid in ("q001", "q002", "q003")
    }


def test_ablate_scores_each_arm_and_its_difference_to_the_reference(setup) -> None:
    setup["run"]("--splits", "dev,test")

    dev = _scores(setup["out"])["splits"]["dev"]
    a0, a1 = dev["A0"], dev["A1"]
    # A0: q001 A at 2; q002 B at 1; q003 one of two items at 1.
    assert a0["R@1"] == pytest.approx((0 + 1 + 0.5) / 3)
    assert a0["R@3"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert a0["MRR@10"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert a0["nDCG@10"] == pytest.approx((NDCG_2ND + 1 + NDCG_ONE_OF_TWO) / 3)
    # A1: q001 A at 1; q002 B at 2; q003 both items at 1 and 2.
    assert a1["R@1"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert a1["nDCG@10"] == pytest.approx((1 + NDCG_2ND + 1) / 3)
    assert a0["vs_reference"] is None

    diff = a1["vs_reference"]["nDCG@10"]
    assert diff["all"]["n"] == 3
    assert diff["all"]["difference"] == pytest.approx((1 - NDCG_ONE_OF_TWO) / 3, abs=1e-6)
    assert diff["by_category"]["lookup"]["difference"] == pytest.approx(0.0, abs=1e-6)
    assert diff["by_category"]["multi_hop"]["difference"] == pytest.approx(
        1 - NDCG_ONE_OF_TWO, abs=1e-6
    )
    assert diff["by_phrasing"]["paraphrase"]["difference"] == pytest.approx(
        NDCG_2ND - 1, abs=1e-6
    )
    assert a1["vs_reference"]["R@3"]["all"]["difference"] == pytest.approx(0.5 / 3, abs=1e-6)
    # The interval is the paired bootstrap of the per-question scores, seed 0.
    expected = metrics.paired_bootstrap([0.5, 1.0, 1.0], [1.0, 0.5, 1.0])
    assert a1["vs_reference"]["MRR@10"]["all"]["low"] == pytest.approx(expected.low, abs=1e-6)
    assert a1["vs_reference"]["MRR@10"]["all"]["high"] == pytest.approx(expected.high, abs=1e-6)

    test = _scores(setup["out"])["splits"]["test"]
    assert test["A0"]["R@5"] == pytest.approx((0 + 1) / 2)
    assert test["A1"]["R@5"] == pytest.approx((1 + 0.5) / 2)
    assert test["A1"]["vs_reference"]["R@5"]["by_category"]["ambiguous"]["n"] == 1


def test_ablate_reports_the_service_latency_and_the_rerank_stage(setup) -> None:
    setup["run"]()

    dev = _scores(setup["out"])["splits"]["dev"]
    assert dev["A0"]["latency_ms"] == {"p50": 12.0, "p95": 12.0, "n": 3}
    assert dev["A0"]["rerank_ms"] is None  # dense has no rerank stage
    # Nearest rank over 100, 200, 300: the 2nd and the 3rd smallest.
    assert dev["A1"]["rerank_ms"] == {"p50": 200.0, "p95": 300.0, "n": 3}
    assert dev["A1"]["latency_ms"]["p50"] == 212.0


def test_the_table_names_each_arm_its_settings_and_the_strata(setup) -> None:
    setup["run"]()

    md = (setup["out"] / "ablation.md").read_text()
    assert "| A1 | hybrid | hybrid | fixture (fixed, 800/120) | on | " in md
    assert "`cross-encoder/ms-marco-MiniLM-L-6-v2` | 20 | 0.7/0.3 |" in md
    assert "## dev: 3 answerable questions" in md
    assert "Difference to A0, arm − A0 per question:" in md
    assert "nDCG@10, arm − A0, per stratum" in md
    assert "category multi_hop (1)" in md and "phrasing paraphrase (1)" in md
    assert "## test" not in md  # test is searched only when asked


def test_an_arm_its_service_does_not_serve_is_refused_before_any_search(setup) -> None:
    setup["fakes"]["http://hybrid"].config = CONFIG | {
        "hybrid": HYBRID_SETTINGS | {"rerank_enabled": False}
    }

    with pytest.raises(SystemExit, match=r"A1: .*hybrid.rerank_enabled: served False"):
        setup["run"]()

    assert setup["fakes"]["http://hybrid"].searches == []


def test_an_expected_key_the_service_does_not_report_is_refused(setup) -> None:
    setup["fakes"]["http://hybrid"].config = {k: v for k, v in CONFIG.items() if k != "hybrid"}

    with pytest.raises(SystemExit, match="hybrid.rerank_enabled: not served"):
        setup["run"]()


def test_a_finished_arm_contacts_no_service_and_rescoring_is_identical(setup, tmp_path) -> None:
    setup["run"]()
    first = {p.name: p.read_bytes() for p in setup["out"].glob("ablation*")}

    def unreachable(url: str) -> httpx.Client:
        raise AssertionError(f"contacted {url}")

    argv = ["ablate", "--arms", str(setup["arms"]), "--run-id", "abl"]
    argv += ["--runs", str(tmp_path / "runs"), "--golden", str(tmp_path / "golden.jsonl")]
    assert rag_eval.main(argv, connect=unreachable) == 0
    assert {p.name: p.read_bytes() for p in setup["out"].glob("ablation*")} == first

    out = tmp_path / "rescored"
    records = setup["out"] / "ablation-records.jsonl"
    assert rag_eval.main(["ablate", "--records", str(records), "--out", str(out)]) == 0
    for name in ("ablation.md", "ablation.json"):
        assert (out / name).read_bytes() == first[name]
    assert not (out / "ablation-records.jsonl").exists()


def test_an_arm_recorded_against_another_config_is_refused(setup) -> None:
    setup["run"]()
    setup["fakes"]["http://hybrid"].config = CONFIG | {"collection": "other"}

    with pytest.raises(SystemExit, match="arm A1 was recorded with another mode or /v1/config"):
        setup["run"]("--splits", "test")


def test_only_collects_the_named_arms(setup) -> None:
    setup["run"]("--only", "A0")

    assert setup["fakes"]["http://hybrid"].searches == []
    assert list(_scores(setup["out"])["splits"]["dev"]) == ["A0"]


def test_a_search_error_is_recorded_and_retried(setup) -> None:
    fake = setup["fakes"]["http://hybrid"]
    real = fake.__call__

    class Flaky:
        failed = False

        def __call__(self, request):
            if request.url.path == "/v1/search" and b"q002" in request.content and not self.failed:
                self.failed = True
                return httpx.Response(503, json={"detail": "busy"})
            return real(request)

    setup["fakes"]["http://hybrid"] = Flaky()
    assert setup["run"]() == 1
    assert _scores(setup["out"])["splits"]["dev"]["A1"]["n"] == 2

    setup["fakes"]["http://hybrid"] = fake
    assert setup["run"]() == 0
    assert fake.searches[-1] == "question q002?"  # only the failed one is searched again
    assert _scores(setup["out"])["splits"]["dev"]["A1"]["n"] == 3


def test_the_arms_file_needs_unique_names_and_a_reference_among_them() -> None:
    arm = {"name": "A0", "change": "dense", "mode": "dense"}
    with pytest.raises(ValidationError, match="repeat"):
        AblationPlan.model_validate({"reference": "A0", "arms": [arm, arm]})
    with pytest.raises(ValidationError, match="not an arm"):
        AblationPlan.model_validate({"reference": "A9", "arms": [arm]})
    with pytest.raises(ValidationError):
        AblationPlan.model_validate({"reference": "A0", "arms": [arm | {"name": "../x"}]})


def test_test_split_must_be_asked_for_by_name(setup) -> None:
    with pytest.raises(SystemExit):
        setup["run"]("--splits", "train")
