"""FastAPI application factory for the orchestration API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.api.routes.approvals import router as approvals_router
from orchestrator.api.routes.memory import router as memory_router
from orchestrator.api.routes.replay import router as replay_router
from orchestrator.api.routes.tasks import router as tasks_router
from orchestrator.api.routes.traces import router as traces_router
from orchestrator.config import get_settings
from orchestrator.llm.router import route
from orchestrator.observability.tracing import setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The provider from the start, so request spans reach Azure Monitor before the first run.
    provider = setup_tracing()
    yield
    # Export what the batch still holds before the process goes (scale-in, a new revision).
    provider.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Agent Orchestration System",
        version="0.1.0",
        lifespan=lifespan,
        # Container Apps probes /health every few seconds; their spans would only be noise.
        telemetry={"exclude": lambda scope: scope.get("path") == "/health"},
    )
    app.include_router(tasks_router)
    app.include_router(memory_router)
    app.include_router(approvals_router)
    app.include_router(traces_router)
    app.include_router(replay_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "run_mode": settings.run_mode}

    @app.get("/config")
    def config() -> dict:
        """The models as routed, the reviewer policy, the run mode and the backends. No keys or URLs.

        The reviewer is shown as routed for the specialists' output, which is what
        it reviews.
        """
        current = get_settings()

        def spec(agent: str, producer_provider: str | None = None) -> str:
            choice = route(agent, producer_provider)
            return f"{choice.provider}:{choice.model}"

        return {
            "models": {
                "supervisor": spec("supervisor"),
                "specialist": spec("research"),
                "reviewer": spec("reviewer", route("research").provider),
                "memory": spec("memory"),
            },
            "reviewer_provider": current.reviewer_provider,
            "mcp_rag": bool(current.mcp_rag_url),
            "mock_llm": current.mock_llm,
            "run_mode": current.run_mode,
            "working_memory": current.working_memory_backend,
            "code_exec": current.code_exec_backend,
            "telemetry": bool(current.applicationinsights_connection_string),
        }

    return app


app = create_app()
