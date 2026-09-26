"""Structured output: its error type, and validation for clients that stand in for a provider."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when LLM output cannot be parsed into the expected schema."""


def validate_output(text: str, model_cls: type[T]) -> T:
    """Validate *text* (a JSON document) against *model_cls*.

    The mock and replay clients use it where a provider would parse its own
    structured output, so a bad fixture fails the same way a bad response does.
    """
    try:
        return model_cls.model_validate_json(text)
    except ValidationError as exc:
        raise StructuredOutputError(str(exc)) from exc
