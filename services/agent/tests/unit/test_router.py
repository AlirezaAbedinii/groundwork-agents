"""Reviewer routing: off the producer's provider by default, onto it on request."""

import pytest
from pydantic import ValidationError

from orchestrator.config import Settings, get_settings
from orchestrator.llm.router import ModelChoice, route


@pytest.fixture()
def settings(monkeypatch):
    """The cached settings with the default routing, whatever a local .env says."""
    current = get_settings()
    monkeypatch.setattr(current, "model_reviewer", "anthropic:claude-sonnet-5")
    monkeypatch.setattr(current, "reviewer_provider", "cross")
    return current


def test_cross_moves_a_colliding_reviewer_to_the_other_provider(settings, monkeypatch):
    monkeypatch.setattr(settings, "model_reviewer", "openai:gpt-4o-mini")

    assert route("reviewer", "openai") == ModelChoice("anthropic", "claude-sonnet-5")
    assert route("reviewer", "anthropic") == ModelChoice("openai", "gpt-4o-mini")


def test_same_keeps_the_reviewer_on_the_producers_provider(settings, monkeypatch):
    monkeypatch.setattr(settings, "reviewer_provider", "same")

    assert route("reviewer", "openai") == ModelChoice("openai", "gpt-4o")
    # A reviewer already configured on the producer's provider keeps its model.
    assert route("reviewer", "anthropic") == ModelChoice("anthropic", "claude-sonnet-5")


def test_without_a_producer_the_configured_reviewer_runs(settings, monkeypatch):
    for policy in ("cross", "same"):
        monkeypatch.setattr(settings, "reviewer_provider", policy)
        assert route("reviewer") == ModelChoice("anthropic", "claude-sonnet-5")


def test_an_unknown_reviewer_policy_fails_validation(monkeypatch):
    monkeypatch.setenv("REVIEWER_PROVIDER", "nearest")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
