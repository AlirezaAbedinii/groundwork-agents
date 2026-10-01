"""FastAPI app + routes.

``POST /v1/ask`` answers a question with citations, confidence, and latency/cost
metadata, and logs a full trace per request. ``POST /v1/search`` runs retrieval
alone, so it works without a generation key. OpenAPI docs are served at ``/docs``.

The app is built by :func:`create_app`, which accepts injectable factories so
tests can swap the pipeline/retriever/trace store for fakes. Real provider clients are
constructed **lazily on first use** — importing this module, serving ``/docs``,
and running ``/health`` all work with no API key configured; a missing key
surfaces as a clear 503 on the endpoints that need it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from fastapi import FastAPI, HTTPException

from ..config import ConfigError, Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from ..observability.metrics import Stopwatch
from ..observability.trace_store import TraceStore
from ..pipeline import AnswerResult, RAGPipeline, SupportsRetrieve
from .schemas import (
    AskRequest,
    AskResponse,
    ChunkInfo,
    ChunksResponse,
    CitationModel,
    ConfigResponse,
    ContextModel,
    DocumentInfo,
    DocumentsResponse,
    IndexInfo,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UsageModel,
)

PipelineFactory = Callable[[str], RAGPipeline]
RetrieverFactory = Callable[[str], SupportsRetrieve]
# indexer(path) -> IndexSummary-like; store_factory() -> object with list_sources/count.
Indexer = Callable[[str], object]
StoreFactory = Callable[[], object]


def _default_pipeline_factory(settings: Settings) -> PipelineFactory:
    def factory(mode: str) -> RAGPipeline:
        return RAGPipeline.from_settings(settings, mode=mode)

    return factory


def _default_retriever_factory(settings: Settings) -> RetrieverFactory:
    def factory(mode: str) -> SupportsRetrieve:
        from ..retrieval import build_retriever

        return build_retriever(mode, settings=settings)

    return factory


_Built = TypeVar("_Built")


@contextmanager
def _missing_pieces_as_http() -> Iterator[None]:
    """A missing piece becomes a 501/503, whether building or serving a request."""
    try:
        yield
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ConfigError as exc:
        # e.g. no API key, or a collection embedded by another model.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        # e.g. a collection or BM25 index that hasn't been seeded yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail=f"Server missing a dependency: {exc}"
        ) from exc


def _build_for_mode(factory: Callable[[str], _Built], mode: str) -> _Built:
    """Build ``mode``'s pipeline or retriever; a missing piece becomes a 501/503."""
    with _missing_pieces_as_http():
        return factory(mode)


def _to_hit(chunk: ScoredChunk) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=chunk.score,
        source_file=str(chunk.metadata.get("source_file", "")),
        # Chunk metadata reports "" for "no heading" (see Chunk.metadata).
        section_heading=chunk.metadata.get("section_heading") or None,
    )


def _to_response(result: AnswerResult) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        mode=result.mode,
        refused=result.refused,
        refused_by=result.refused_by,
        retrieval_confidence=result.retrieval_confidence,
        confidence=result.confidence,
        confidence_breakdown=result.confidence_breakdown,
        citations=[
            CitationModel(
                index=c.index,
                resolved=c.resolved,
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                section_heading=c.section_heading,
                supported=c.supported,
            )
            for c in result.citations
        ],
        contexts=[
            ContextModel(chunk_id=c.chunk_id, text=c.text, score=c.score, metadata=c.metadata)
            for c in result.contexts
        ],
        usage=UsageModel(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_usd=round(result.cost_usd, 6),
        timings_ms=result.timings_ms,
    )


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_factory: PipelineFactory | None = None,
    retriever_factory: RetrieverFactory | None = None,
    trace_store: TraceStore | None = None,
    indexer: Indexer | None = None,
    store_factory: StoreFactory | None = None,
) -> FastAPI:
    """Build the FastAPI app with injectable dependencies (fakes in tests)."""
    settings = settings or get_settings()

    app = FastAPI(
        title="RAG Hybrid Search",
        version="0.1.0",
        description=(
            "Retrieval-Augmented Generation over technical docs: grounded answers "
            "with [n] citations, a confidence-gated refusal path, and per-request "
            "latency/cost metadata."
        ),
    )
    app.state.settings = settings
    app.state.pipeline_factory = pipeline_factory or _default_pipeline_factory(settings)
    app.state.trace_store = trace_store  # created lazily so imports touch no disk
    app.state.pipelines = {}  # mode -> RAGPipeline, built on first use
    app.state.retriever_factory = retriever_factory or _default_retriever_factory(settings)
    app.state.retrievers = {}  # mode -> retriever for /v1/search, built on first use
    app.state.indexer = indexer
    app.state.store_factory = store_factory
    app.state.store = None  # built on first use and reused: it holds a connection pool

    def _get_indexer() -> Indexer:
        if app.state.indexer is None:
            from ..indexing import index_path

            app.state.indexer = lambda path: index_path(path, settings=settings)
        return app.state.indexer

    def _get_store():
        if app.state.store is None:
            if app.state.store_factory is None:
                from ..indexing.vector_store import VectorStore

                app.state.store_factory = lambda: VectorStore.from_settings(settings)
            app.state.store = app.state.store_factory()
        return app.state.store

    def _get_trace_store() -> TraceStore:
        if app.state.trace_store is None:
            app.state.trace_store = TraceStore(settings.trace_store_path)
        return app.state.trace_store

    def _get_pipeline(mode: str) -> RAGPipeline:
        if mode not in app.state.pipelines:
            app.state.pipelines[mode] = _build_for_mode(app.state.pipeline_factory, mode)
        return app.state.pipelines[mode]

    def _get_retriever(mode: str) -> SupportsRetrieve:
        # Only the retriever: no chat client, so no generation key is needed.
        if mode not in app.state.retrievers:
            app.state.retrievers[mode] = _build_for_mode(app.state.retriever_factory, mode)
        return app.state.retrievers[mode]

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=AskResponse, tags=["query"])
    def ask(body: AskRequest) -> AskResponse:
        """Answer a question from the indexed docs, with citations + metadata."""
        mode = body.mode or settings.default_mode
        pipeline = _get_pipeline(mode)
        with _missing_pieces_as_http():
            result = pipeline.answer(body.question, top_k=body.top_k)

        # Log the full trace (latency, tokens, cost) for /v1/stats and analysis.
        _get_trace_store().record(
            {
                "question": result.question,
                "mode": result.mode,
                "refused": result.refused,
                "confidence": result.retrieval_confidence,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cost_usd": result.cost_usd,
                "timings_ms": result.timings_ms,
            }
        )
        return _to_response(result)

    @app.post("/v1/search", response_model=SearchResponse, tags=["query"])
    def search(body: SearchRequest) -> SearchResponse:
        """Return the best-matching chunks, without generating an answer."""
        mode = body.mode or settings.default_mode
        retriever = _get_retriever(mode)
        sw = Stopwatch()
        with _missing_pieces_as_http():
            chunks = retriever.retrieve(body.query, top_k=body.top_k, stopwatch=sw)
        # No trace record: /v1/stats summarizes answered questions.
        return SearchResponse(
            query=body.query, mode=mode, hits=[_to_hit(c) for c in chunks], timings_ms=sw.as_dict()
        )

    @app.post("/v1/ingest", response_model=IngestResponse, tags=["index"])
    def ingest(body: IngestRequest) -> IngestResponse:
        """Index a file or directory (defaults to the sample corpus)."""
        from pathlib import Path

        target = Path(body.path) if body.path else settings.corpus_dir
        if not target.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {target}")
        with _missing_pieces_as_http():
            summary = _get_indexer()(str(target))
        return IngestResponse(
            files=summary.files,
            chunks_indexed=summary.chunks_indexed,
            total_chunks_in_store=summary.total_chunks_in_store,
            embedding_cost_usd=summary.embedding_cost_usd,
            chunks_skipped_duplicates=getattr(summary, "chunks_skipped_duplicates", 0),
            timings_ms=summary.timings_ms,
        )

    @app.get("/v1/documents", response_model=DocumentsResponse, tags=["index"])
    def documents() -> DocumentsResponse:
        """List indexed source documents and their chunk counts."""
        with _missing_pieces_as_http():
            store = _get_store()
            sources = store.list_sources()
            total = store.count()
        return DocumentsResponse(
            documents=[DocumentInfo(source_file=s, chunks=n) for s, n in sources.items()],
            total_chunks=total,
        )

    @app.get("/v1/chunks", response_model=ChunksResponse, tags=["index"])
    def chunks(source_file: str) -> ChunksResponse:
        """One indexed source's chunks in document order (paths as ``/v1/documents`` lists them)."""
        with _missing_pieces_as_http():
            stored = _get_store().chunks_for_source(source_file)
        if not stored:
            raise HTTPException(status_code=404, detail=f"No indexed source {source_file!r}")
        return ChunksResponse(
            source_file=source_file,
            chunks=[
                ChunkInfo(
                    chunk_id=c.chunk_id,
                    ordinal=c.ordinal,
                    section_heading=c.section_heading or None,
                    text=c.text,
                )
                for c in stored
            ],
        )

    @app.get("/v1/config", response_model=ConfigResponse, tags=["ops"])
    def config() -> ConfigResponse:
        """The models, retrieval settings and thresholds this service runs with, and what
        built the served collection; evaluation reports record it. No keys or URLs."""
        with _missing_pieces_as_http():
            store = _get_store()
            info = store.collection_info()
            index = None
            if info is not None:
                index = IndexInfo(
                    embedding_model=info.embedding_model,
                    dim=info.dim,
                    chunk_strategy=info.chunking.strategy,
                    chunk_size=info.chunking.size,
                    chunk_overlap=info.chunking.overlap,
                    chunks=store.count(),
                )
        return ConfigResponse(
            collection=settings.collection,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            llm_provider=settings.llm_provider,
            generation_model=settings.generation_model,
            default_mode=settings.default_mode,
            top_k=settings.top_k,
            rerank_top_k=settings.rerank_top_k,
            thresholds={
                mode: settings.refusal_threshold(mode) for mode in ("dense", "hybrid")
            },
            citation_verification=settings.citation_verification,
            index=index,
        )

    @app.get("/v1/stats", response_model=StatsResponse, tags=["ops"])
    def stats() -> StatsResponse:
        """Cost/latency summary (P50/P95/P99 per stage) over all logged requests."""
        return StatsResponse(**_get_trace_store().aggregates())

    return app


# uvicorn entrypoint: `uvicorn rag.api.main:app`
app = create_app()
