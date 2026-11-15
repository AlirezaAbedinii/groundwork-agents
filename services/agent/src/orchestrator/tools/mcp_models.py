"""Types for tools that live on an MCP server.

Discovery turns each server tool into an ordinary ToolSpec, so the registry
applies the same permissions, rate limits, validation, logging and tracing to
MCP tools as to local ones. What the agent allows for a server tool comes from
a McpToolPolicy on this side, never from the server's own annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from orchestrator.tools.serialize import DEFAULT_RESULT_BUDGET_CHARS


@dataclass(frozen=True)
class McpToolPolicy:
    """Which specialists may call one server tool, how often per task, whether it needs approval,
    and how many characters of its result the model sees."""

    owners: frozenset[str]
    rate_limit: int
    sensitive: bool = False
    result_budget_chars: int = DEFAULT_RESULT_BUDGET_CHARS
    changes_state: bool = False  # ToolSpec.changes_state; a server tool that writes must set it


class McpToolOutput(BaseModel):
    """An MCP tool's result: the structured content if the server sent it, else its text."""

    content: dict | list | str


class McpUnavailableError(RuntimeError):
    """Discovery could not reach or talk to the MCP server."""


# Where the tools live: a Streamable HTTP URL in production, an in-process MCPServer in tests.
Target = str | Any
