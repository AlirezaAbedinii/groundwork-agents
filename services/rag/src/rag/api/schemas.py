"""Pydantic request/response models for the API.

These define the public contract of the service (and drive the OpenAPI docs at
``/docs``). ``AskResponse`` mirrors :class:`rag.pipeline.AnswerResult`: the
answer text, the parsed ``[n]`` citations, the ranked retrieved contexts, the
confidence signal, and the latency/cost metadata recorded for every request.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of ``POST /v1/ask``."""

    question: str = Field(min_length=1, max_length=2000, description="The user question.")
    mode: Literal["dense", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode. Defaults to the server's configured mode "
        "(dense). 'hybrid' returns 501 until the V1 retrieval stack lands.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Chunks to retrieve (default from config)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What does FERRY-429 mean?", "mode": "dense", "top_k": 5}]
        }
    }


class CitationModel(BaseModel):
    """One ``[n]`` citation parsed from the answer, mapped to its source chunk."""

    index: int = Field(description="The n in [n], 1-based, as written in the answer.")
    resolved: bool = Field(description="False if [n] points outside the retrieved set.")
    chunk_id: str = ""
    source_file: str = ""
    section_heading: str | None = None
    supported: bool | None = Field(
        default=None,
        description="LLM-judge verification verdict: does the cited chunk support "
        "the claim? None when verification did not run.",
    )


class ContextModel(BaseModel):
    """One retrieved chunk, ranked by similarity."""

    chunk_id: str
    text: str
    score: float = Field(description="Similarity score (1 - cosine distance).")
    metadata: dict = Field(default_factory=dict)


class UsageModel(BaseModel):
    """Token usage for the generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    """Response of ``POST /v1/ask``."""

    question: str
    answer: str
    mode: str
    refused: bool = Field(description="True when the system declined to answer.")
    refused_by: Literal["gate", "model"] | None = Field(
        default=None,
        description="Who declined: the retrieval gate (before generating) or the model "
        "(after reading the context). None for an answer.",
    )
    retrieval_confidence: float = Field(
        description="The top retrieved chunk's score, which the gate compares with the "
        "mode's threshold (cosine in dense mode, reranker score in hybrid mode)."
    )
    confidence: float = Field(
        description="Composite confidence in [0,1]: weighted retrieval score + "
        "citation coverage + answer completeness."
    )
    confidence_breakdown: dict = Field(
        default_factory=dict,
        description="Per-component confidence inputs (retrieval, citation_coverage, "
        "completeness) and whether verification ran.",
    )
    citations: list[CitationModel] = Field(default_factory=list)
    contexts: list[ContextModel] = Field(
        default_factory=list, description="Retrieved chunks, ranked by score."
    )
    usage: UsageModel = Field(default_factory=UsageModel)
    cost_usd: float = Field(description="Embedding + generation cost of this request.")
    timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Per-stage latency (embed, dense, generate, total_ms)."
    )


class SearchRequest(BaseModel):
    """Body of ``POST /v1/search``."""

    query: str = Field(min_length=1, max_length=2000, description="What to search for.")
    mode: Literal["dense", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode. Defaults to the server's configured mode.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Chunks to return (default from config)."
    )

    model_config = {
        "json_schema_extra": {"examples": [{"query": "FERRY-429", "mode": "hybrid", "top_k": 5}]}
    }


class SearchHit(BaseModel):
    """One retrieved chunk."""

    chunk_id: str
    text: str
    score: float = Field(
        description="Higher is more relevant: 1 - cosine distance (dense) or the "
        "sigmoid-normalized cross-encoder score (hybrid)."
    )
    source_file: str
    section_heading: str | None = None


class SearchResponse(BaseModel):
    """Response of ``POST /v1/search``: retrieval only, nothing generated."""

    query: str
    mode: str
    hits: list[SearchHit] = Field(default_factory=list, description="Best match first.")
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage latency (embed, dense, sparse, fusion, rerank, total_ms).",
    )


class IngestRequest(BaseModel):
    """Body of ``POST /v1/ingest``."""

    path: str | None = Field(
        default=None,
        description="File or directory to ingest. Defaults to the sample corpus.",
    )


class IngestResponse(BaseModel):
    """Result of an ingest run."""

    files: int
    chunks_indexed: int
    total_chunks_in_store: int
    embedding_cost_usd: float
    chunks_skipped_duplicates: int = Field(
        default=0, description="Near-duplicate chunks skipped by dedup (cosine > threshold)."
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """One indexed source document."""

    source_file: str
    chunks: int


class DocumentsResponse(BaseModel):
    """Response of ``GET /v1/documents``."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total_chunks: int = 0


class StatsResponse(BaseModel):
    """Response of ``GET /v1/stats`` — cost/latency summary over all traces."""

    requests: int
    refused: int
    refusal_rate: float
    total_cost_usd: float
    mean_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    latency_ms: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-stage P50/P95/P99 (+ sample count n), e.g. {'generate': {'p50': ...}}.",
    )


class ChunkInfo(BaseModel):
    """One stored chunk of a source document."""

    chunk_id: str
    ordinal: int
    section_heading: str | None = None
    text: str


class ChunksResponse(BaseModel):
    """Response of ``GET /v1/chunks``: one source's chunks in document order."""

    source_file: str
    chunks: list[ChunkInfo] = Field(default_factory=list)


class IndexInfo(BaseModel):
    """What built the served collection, from its registry row."""

    embedding_model: str
    dim: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    chunks: int


class ConfigResponse(BaseModel):
    """Response of ``GET /v1/config``: what this service runs with. No secrets."""

    collection: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    generation_model: str
    default_mode: str
    top_k: int
    rerank_top_k: int
    thresholds: dict[str, float] = Field(
        description="The refusal gate's threshold per retrieval mode (dense, hybrid)."
    )
    citation_verification: bool
    index: IndexInfo | None = Field(
        default=None, description="The served collection's registry row; null until it's seeded."
    )
