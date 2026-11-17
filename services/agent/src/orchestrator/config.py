"""Central configuration.

Every runtime flag and threshold lives here.
Values come from the environment or a local .env file; defaults target host-run
development against `make infra` services.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    tavily_api_key: str = ""  # optional; without it web_search is left out of the registry (except under MOCK_LLM)

    # Storage
    database_url: str = "postgresql+psycopg://orchestrator:orchestrator@localhost:5432/orchestrator"
    redis_url: str = "redis://localhost:6379/0"
    # Task working memory: Redis, or a Postgres table where there is no Redis (memory/working.py)
    working_memory_backend: Literal["redis", "postgres"] = "redis"

    # Execution
    run_mode: Literal["inline", "celery"] = "inline"
    mock_llm: bool = False
    llm_fixtures_dir: Path = Path("tests/fixtures/llm")

    # Agent → model routing ("provider:model"); the reviewer must land on a
    # different provider than the producing agent (see llm/router.py)
    model_supervisor: str = "openai:gpt-4o"
    model_specialist: str = "openai:gpt-4o-mini"
    model_reviewer: str = "anthropic:claude-sonnet-5"
    # "cross" moves the reviewer off the producer's provider; "same" keeps it there
    # (the evaluation compares the two; see llm/router.py)
    reviewer_provider: Literal["cross", "same"] = "cross"
    model_memory: str = "openai:gpt-4o-mini"  # extraction + consolidation summaries

    # Memory
    working_memory_ttl_s: int = 86_400  # safety net for crashed runs; normal path clears explicitly
    memory_retrieval_k: int = 3  # top-k per collection injected into planning
    memory_half_life_days: float = 14.0  # importance decay half-life
    memory_consolidation_similarity: float = 0.9  # cosine sim above which memories merge
    memory_expiry_days: float = 30.0  # minimum staleness before a memory may expire
    memory_min_importance: float = 0.2  # expire only below this importance
    mock_embedding_dim: int = 256  # token-hash embedding size under MOCK_LLM

    # Tools
    workspace_root: Path = Path("task_workspaces")
    max_tool_iterations: int = 5  # model turns per specialist attempt; a parallel batch of tool calls is one turn
    api_call_allowlist: list[str] = ["api.github.com"]
    # "disabled" leaves code_exec out of the registry: for deployments without a sandbox (tools/code_exec.py)
    code_exec_backend: Literal["docker", "subprocess", "disabled"] = "docker"
    code_exec_image: str = "orchestrator-sandbox"
    code_exec_timeout_s: int = 20

    # MCP: retrieval tools from the RAG stack's MCP server
    mcp_rag_url: str = ""  # Streamable HTTP URL, e.g. http://localhost:8001/mcp; empty disables the rag_* tools
    mcp_call_timeout_s: float = 30.0

    # Planning (planning/capabilities.py): the most subtasks a plan may have, and whether a
    # plan that fails the lint is sent back for the one retry ("enforce") or only recorded ("off")
    max_plan_subtasks: int = Field(4, ge=1)
    plan_lint: Literal["enforce", "off"] = "enforce"
    # After synthesis, each file the plan lists as a deliverable must have been written at
    # exactly its path; a missing one sends its producer back once, then fails the task.
    # False skips the check (graph/edges.py check_deliverables).
    verify_deliverables: bool = True
    # Review with evidence: a subtask that lists tools, none of which returned a result in an
    # attempt, is reworked with no reviewer call, and the reviewer is shown the request and
    # the attempt's tool receipts. False restores the earlier review (agents/reviewer.py).
    review_evidence: bool = True

    # Thresholds
    plan_confidence_threshold: float = 0.7
    review_score_threshold: int = 3
    max_specialist_retries: int = 2
    # Hard cap on one subtask's attempts: at this many, an approval no longer buys
    # another one and the task fails; a human can still take over or reject
    max_subtask_attempts: int = Field(5, ge=1)
    # Hard caps on one task's model calls and spend, counted across runs and workers
    # (observability/tracing.py): reaching either fails the task before its next call. 0 disables.
    max_task_llm_calls: int = Field(60, ge=0)
    max_task_cost_usd: float = Field(0.50, ge=0)

    # Observability: spans also go to Azure Monitor (Application Insights) when this is set
    applicationinsights_connection_string: str = ""
    app_version: str = "dev"  # service.version on every span; published images set their tag

    # Human-in-the-loop
    approval_webhook_url: str = ""
    # Per-trigger approval-level overrides, e.g. {"sensitive_operation": "notify"}
    approval_level_overrides: dict[str, str] = {}

    @property
    def initial_attempt_allowance(self) -> int:
        """Attempts a subtask may make before its first escalation: the first try plus
        the retries, never more than the cap."""
        return min(self.max_specialist_retries + 1, self.max_subtask_attempts)


@lru_cache
def get_settings() -> Settings:
    return Settings()
