"""Cross-encoder reranker: score top-N candidates -> keep top-k.

A cross-encoder reads the *(query, chunk)* pair jointly, so it is far more
precise than embedding similarity — and far slower, which is why it only
reranks the short fused candidate list (default top-20 -> top-5) instead of
searching the corpus.

The model (``gte-reranker-modernbert-base`` by default) runs locally via
``sentence-transformers`` (the ``rerank`` extra) — no LLM cost. Each model's raw
logits are mapped into [0, 1] by one sigmoid, here, so the refusal gate compares
the same kind of score whichever model is configured. The scorer sits behind a
one-method protocol, so tests inject a deterministic fake and never load torch.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..indexing.vector_store import ScoredChunk

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup")


class SupportsScorePairs(Protocol):
    """Scores (query, text) pairs with unbounded logits; higher = more relevant."""

    def score_pairs(self, query: str, texts: list[str]) -> list[float]: ...


def _quota_cpus(root: Path) -> float | None:
    """The CPU quota of this process's cgroup, in CPUs; None when there is none."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = (root / "cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1: a quota of -1 means none
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def cpu_limit(cgroup_root: Path = CGROUP_ROOT) -> int:
    """The CPUs this process may use: the cores it may run on, capped by its cgroup's CPU
    quota (whole CPUs, at least 1).

    A container limited by quota (``docker --cpus``, Azure Container Apps) still sees
    every host core, and torch starts one thread per core by default; on a 1-CPU quota
    on an 8-core host that ran the reranker about four times slower than one thread.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        cores = os.cpu_count() or 1
    quota = _quota_cpus(cgroup_root)
    if quota is not None:
        cores = min(cores, math.floor(quota))
    return max(1, cores)


def _logits(scores: Any) -> Any:
    return scores


class CrossEncoderScorer:
    """Local sentence-transformers cross-encoder (lazy import, cached model).

    Returns raw logits. sentence-transformers applies an activation of its own choosing
    in ``predict``: a sigmoid for a one-label model whose config names none (such as
    gte-reranker-modernbert-base), the identity for one whose config says so (such as
    ms-marco-MiniLM-L-6-v2). The scorer asks for the logits in every case, so the
    :class:`Reranker` maps them once.
    """

    def __init__(self, model_name: str, model: Any = None) -> None:
        self.model_name = model_name
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "Reranking requires 'sentence-transformers'. "
                    'Install: pip install -e ".[rerank]"'
                ) from exc
            threads = cpu_limit()
            torch.set_num_threads(threads)
            self._model = CrossEncoder(self.model_name)
            logger.info("reranker %s loaded, %d torch threads", self.model_name, threads)
        return self._model

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        model = self._ensure_model()
        scores = model.predict([(query, t) for t in texts], activation_fn=_logits)
        return [float(s) for s in scores]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Reranker:
    """Rerank candidate chunks by cross-encoder relevance."""

    scorer: SupportsScorePairs
    top_k: int = 5

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        """Return the best ``top_k`` candidates re-scored and re-ordered.

        Result scores are the sigmoid of the scorer's logits, in [0, 1] (the
        original retrieval scores served their purpose upstream). Ties break by
        chunk_id for determinism.
        """
        if not candidates:
            return []
        raw = self.scorer.score_pairs(query, [c.text for c in candidates])
        rescored = [
            ScoredChunk(
                chunk_id=c.chunk_id, text=c.text, score=_sigmoid(s), metadata=c.metadata
            )
            for c, s in zip(candidates, raw, strict=True)
        ]
        rescored.sort(key=lambda c: (-c.score, c.chunk_id))
        return rescored[: top_k or self.top_k]
