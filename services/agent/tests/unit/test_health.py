def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["run_mode"] in ("inline", "celery")


def test_config_shows_the_routing_and_no_secrets(client, monkeypatch):
    from orchestrator.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "model_reviewer", "anthropic:claude-sonnet-5")
    monkeypatch.setattr(settings, "reviewer_provider", "cross")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-secret-0001")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-secret-0002")
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-secret-0003")
    monkeypatch.setattr(settings, "mcp_rag_url", "http://localhost:8001/mcp")

    response = client.get("/config")

    assert response.status_code == 200
    assert not any(s in response.text for s in ("secret-0001", "secret-0002", "secret-0003", "postgresql"))
    body = response.json()
    assert body["models"]["reviewer"] == "anthropic:claude-sonnet-5"  # cross, for OpenAI specialists
    assert (body["reviewer_provider"], body["mcp_rag"], body["mock_llm"]) == ("cross", True, True)

    monkeypatch.setattr(settings, "reviewer_provider", "same")
    assert client.get("/config").json()["models"]["reviewer"] == "openai:gpt-4o"


def test_config_shows_the_backends(client, monkeypatch):
    from orchestrator.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "working_memory_backend", "redis")
    monkeypatch.setattr(settings, "code_exec_backend", "docker")
    body = client.get("/config").json()
    assert (body["working_memory"], body["code_exec"]) == ("redis", "docker")

    monkeypatch.setattr(settings, "working_memory_backend", "postgres")
    monkeypatch.setattr(settings, "code_exec_backend", "disabled")
    body = client.get("/config").json()
    assert (body["working_memory"], body["code_exec"]) == ("postgres", "disabled")


def test_config_says_whether_traces_go_to_azure_without_showing_the_string(client, monkeypatch):
    from orchestrator.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "applicationinsights_connection_string", "")
    assert client.get("/config").json()["telemetry"] is False

    monkeypatch.setattr(settings, "applicationinsights_connection_string", "InstrumentationKey=secret-0004")
    response = client.get("/config")
    assert response.json()["telemetry"] is True
    assert "secret-0004" not in response.text


def test_the_lifespan_installs_tracing_and_shuts_it_down_on_exit(monkeypatch):
    from fastapi.testclient import TestClient

    from orchestrator import main

    class FakeProvider:
        shut_down = False

        def shutdown(self):
            self.shut_down = True

    provider = FakeProvider()
    monkeypatch.setattr(main, "setup_tracing", lambda: provider)
    with TestClient(main.create_app()) as client:  # the context manager runs the lifespan
        assert client.get("/health").status_code == 200
        assert not provider.shut_down
    assert provider.shut_down  # the batched exporter flushes on shutdown
