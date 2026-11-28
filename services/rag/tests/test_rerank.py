"""The reranker maps each model's output into [0, 1] once, on as many threads as CPUs."""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import pytest

from rag.indexing.vector_store import ScoredChunk
from rag.retrieval.rerank import CrossEncoderScorer, Reranker, cpu_limit


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FakeCrossEncoder:
    """Like sentence-transformers' CrossEncoder.predict: the model's own activation
    applies unless the caller passes another."""

    def __init__(self, logits: dict[str, float], activation) -> None:
        self.logits = logits
        self.activation_fn = activation

    def predict(self, pairs, activation_fn=None):
        fn = activation_fn or self.activation_fn
        return [fn(self.logits[text]) for _, text in pairs]


LOGITS = {"best": 3.0, "middle": 0.5, "worst": -2.0}


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param(_sigmoid, id="returns-probabilities"),
        pytest.param(lambda x: x, id="returns-logits"),
    ],
)
def test_each_model_is_mapped_into_0_1_exactly_once(activation) -> None:
    model = FakeCrossEncoder(LOGITS, activation)
    candidates = [ScoredChunk(cid, cid, 0.0, {}) for cid in ("worst", "best", "middle")]

    hits = Reranker(scorer=CrossEncoderScorer("m", model=model), top_k=3).rerank("q", candidates)

    assert [h.chunk_id for h in hits] == ["best", "middle", "worst"]
    assert [h.score for h in hits] == pytest.approx([_sigmoid(LOGITS[h.chunk_id]) for h in hits])


def test_a_model_that_returns_probabilities_keeps_its_own_scale() -> None:
    """A second sigmoid would squeeze 0.95 into 0.72 and fail every calibrated gate."""
    model = FakeCrossEncoder(LOGITS, _sigmoid)
    own = model.predict([("q", "best")])[0]

    (hit,) = Reranker(scorer=CrossEncoderScorer("m", model=model), top_k=1).rerank(
        "q", [ScoredChunk("best", "best", 0.0, {})]
    )

    assert hit.score == pytest.approx(own)
    assert hit.score > 0.95


def _cgroup(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"cpu.max": "100000 100000\n"}, 1),  # docker --cpus=1, Azure's 1 vCPU
        ({"cpu.max": "400000 100000\n"}, 4),
        ({"cpu.max": "150000 100000\n"}, 1),  # whole CPUs only
        ({"cpu.max": "50000 100000\n"}, 1),  # at least one thread
        ({"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 2),
    ],
)
def test_cpu_limit_follows_the_cgroup_quota(tmp_path, monkeypatch, files, expected) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    assert cpu_limit(_cgroup(tmp_path, files)) == expected


@pytest.mark.parametrize(
    "files",
    [
        {"cpu.max": "max 100000\n"},
        {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"},
        {},
    ],
)
def test_cpu_limit_without_a_quota_is_the_cores_it_may_run_on(tmp_path, monkeypatch, files) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6})
    assert cpu_limit(_cgroup(tmp_path, files)) == 3


def test_loading_the_model_sets_torch_threads_to_the_cpu_limit(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: calls.append(("threads", n))
    st = types.ModuleType("sentence_transformers")

    def cross_encoder(name: str) -> FakeCrossEncoder:
        calls.append(("load", name))
        return FakeCrossEncoder(LOGITS, _sigmoid)

    st.CrossEncoder = cross_encoder
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr("rag.retrieval.rerank.cpu_limit", lambda: 2)

    scorer = CrossEncoderScorer("org/model")
    assert scorer.score_pairs("q", ["best"]) == [3.0]
    scorer.score_pairs("q", ["worst"])

    assert calls == [("threads", 2), ("load", "org/model")]  # threads before the model, once
