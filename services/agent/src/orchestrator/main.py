"""FastAPI application factory for the orchestration API."""

from fastapi import FastAPI

from orchestrator.api.routes.approvals import router as approvals_router
from orchestrator.api.routes.memory import router as memory_router
from orchestrator.api.routes.replay import router as replay_router
from orchestrator.api.routes.tasks import router as tasks_router
from orchestrator.api.routes.traces import router as traces_router
from orchestrator.config import get_settings
from orchestrator.llm.router import route


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Agent Orchestration System", version="0.1.0")
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
        """The models as routed, the reviewer policy and the run mode. No keys or URLs.

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
        }

    return app


app = create_app()
