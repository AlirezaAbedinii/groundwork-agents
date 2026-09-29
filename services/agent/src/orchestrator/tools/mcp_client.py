"""MCP client adapter: tools on an MCP server become ordinary registry ToolSpecs.

``discover_tools`` lists a server's tools and turns each one its policy allows
into a ToolSpec; the registry then applies permissions, rate limits, argument
validation, logging, tracing and the approval check to it exactly as to a local
tool, so none of that is repeated here. Tools without a policy entry are
skipped: deny by default, whatever the server's annotations claim.

``call_tool`` is the ToolSpec handler's body. It is sync because the registry
runs handlers in worker threads with no event loop, so each call drives its own
short loop: one client session, one call, closed, and nothing that outlives the
loop it was made on (a session reused across loops or threads would break).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping

from mcp import Client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel

from orchestrator.tools.base import ToolContext, ToolExecutionError, ToolSpec
from orchestrator.tools.json_schema_model import model_from_json_schema
from orchestrator.tools.mcp_models import McpToolOutput, McpToolPolicy, McpUnavailableError, Target

logger = logging.getLogger(__name__)


async def discover_tools(
    target: Target, policy: Mapping[str, McpToolPolicy], *, prefix: str, timeout_s: float = 30.0
) -> list[ToolSpec]:
    """ToolSpecs named ``prefix + tool name`` for the server tools ``policy`` allows."""
    try:
        offered = await _list_tools(target, timeout_s)
    except Exception as exc:  # connect, protocol, timeout: nothing to register
        raise McpUnavailableError(f"MCP server unreachable: {_root_cause(exc)}") from exc

    specs = []
    for tool in offered:
        allowed = policy.get(tool.name)
        if allowed is None:
            logger.info("MCP tool %r has no policy entry, so it isn't registered", tool.name)
            continue
        specs.append(
            ToolSpec(
                name=prefix + tool.name,
                description=tool.description or tool.name,
                # A schema outside the supported subset raises ValueError here, loudly.
                input_schema=model_from_json_schema(f"{prefix}{tool.name}_input", tool.input_schema),
                output_schema=McpToolOutput,
                owners=allowed.owners,
                rate_limit=allowed.rate_limit,
                sensitive=allowed.sensitive,
                handler=_handler(target, tool.name, timeout_s),
            )
        )
    for missing in sorted(policy.keys() - {tool.name for tool in offered}):
        logger.warning("MCP policy allows %r, but the server doesn't offer it", missing)
    return specs


def call_tool(target: Target, name: str, arguments: dict, *, timeout_s: float = 30.0) -> McpToolOutput:
    """Call one server tool from a thread with no running loop; every failure is a ToolExecutionError."""
    try:
        result = asyncio.run(_call(target, name, arguments, timeout_s))
    except Exception as exc:  # the transport's errors arrive wrapped in anyio ExceptionGroups
        raise ToolExecutionError(f"MCP server unreachable: {_root_cause(exc)}") from exc
    text = "\n".join(block.text for block in result.content if getattr(block, "text", None) is not None)
    if result.is_error:
        raise ToolExecutionError(text or f"MCP tool {name!r} failed")
    return McpToolOutput(content=result.structured_content if result.structured_content is not None else text)


def _handler(target: Target, name: str, timeout_s: float) -> Callable[[BaseModel, ToolContext], McpToolOutput]:
    def handler(args: BaseModel, ctx: ToolContext) -> McpToolOutput:
        # The server's own name, and only the arguments that are set: None means "use your default".
        return call_tool(target, name, args.model_dump(mode="json", exclude_none=True), timeout_s=timeout_s)

    return handler


async def _list_tools(target: Target, timeout_s: float) -> list[Tool]:
    tools: list[Tool] = []
    async with Client(target, read_timeout_seconds=timeout_s) as client:
        cursor = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            if not (cursor := page.next_cursor):
                return tools


async def _call(target: Target, name: str, arguments: dict, timeout_s: float) -> CallToolResult:
    # Only the call happens inside the session; results are mapped after it closes,
    # so a ToolError raised there can't come back wrapped in the session's task group.
    async with Client(target, read_timeout_seconds=timeout_s) as client:
        return await client.call_tool(name, arguments)


def _root_cause(exc: BaseException) -> str:
    """The first leaf of nested exception groups, as "Type: message"."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"
