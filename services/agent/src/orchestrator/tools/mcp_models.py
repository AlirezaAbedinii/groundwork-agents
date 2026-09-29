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


@dataclass(frozen=True)
class McpToolPolicy:
    """Which specialists may call one server tool, how often per task, and whether it needs approval."""

    owners: frozenset[str]
    rate_limit: int
    sensitive: bool = False


class McpToolOutput(BaseModel):
    """An MCP tool's result: the structured content if the server sent it, else its text."""

    content: dict | list | str


class McpUnavailableError(RuntimeError):
    """Discovery could not reach or talk to the MCP server."""


# Where the tools live: a Streamable HTTP URL in production, an in-process MCPServer in tests.
Target = str | Any
