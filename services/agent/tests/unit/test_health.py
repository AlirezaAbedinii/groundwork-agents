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
