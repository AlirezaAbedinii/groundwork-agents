import pytest

from orchestrator.config import Settings


def test_defaults_are_dev_friendly():
    settings = Settings(_env_file=None)
    assert settings.run_mode == "inline"
    assert settings.plan_confidence_threshold == 0.7
    assert settings.review_score_threshold == 3
    assert settings.max_specialist_retries == 2
    assert settings.max_subtask_attempts == 5
    assert settings.initial_attempt_allowance == 3


def test_the_first_allowance_never_passes_the_attempt_cap(monkeypatch):
    monkeypatch.setenv("MAX_SPECIALIST_RETRIES", "6")
    monkeypatch.setenv("MAX_SUBTASK_ATTEMPTS", "4")
    assert Settings(_env_file=None).initial_attempt_allowance == 4


def test_the_attempt_cap_is_at_least_one(monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setenv("MAX_SUBTASK_ATTEMPTS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "celery")
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("PLAN_CONFIDENCE_THRESHOLD", "0.9")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@example:5432/db")

    settings = Settings(_env_file=None)
    assert settings.run_mode == "celery"
    assert settings.mock_llm is True
    assert settings.plan_confidence_threshold == 0.9
    assert settings.database_url.endswith("@example:5432/db")
