import os

# Tests never call real providers; set before any orchestrator import.
os.environ.setdefault("MOCK_LLM", "1")
# Tests run code_exec via the isolated subprocess backend (no Docker needed).
os.environ.setdefault("CODE_EXEC_BACKEND", "subprocess")
# Tests run graphs in-process; a developer .env set to RUN_MODE=celery would
# otherwise enqueue every test task to whatever worker is listening.
os.environ["RUN_MODE"] = "inline"
# Tests opt in to the RAG tools over MCP explicitly; a developer .env pointing at a
# live MCP server would otherwise add them to every graph run.
os.environ["MCP_RAG_URL"] = ""

import pytest
from fastapi.testclient import TestClient

from orchestrator.main import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    # Async tests run on the anyio pytest plugin; pin the backend so each test
    # runs once on asyncio (trio is not installed).
    return "asyncio"
