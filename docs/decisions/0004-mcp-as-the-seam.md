# 0004. MCP as the seam between the agents and retrieval

Date: 2026-09-29. Status: accepted.

## Context

The repository held two services that didn't call each other. The retrieval
service answered questions over HTTP, and the agents' research specialist could
only search the web. The goal was retrieval that any agent can use through a
standard interface, with the agent's existing controls (permissions, rate
limits, argument validation, logging, tracing and the approval gate) applied to
it exactly as to its local tools. The current MCP Python SDK is 2.x: it renamed
the v1 server API most examples still use (`FastMCP` is now `MCPServer`) and
negotiates the 2026-07-28 protocol revision. The LangChain MCP adapters still
require the 1.x SDK, and they produce LangChain tools, which the agents' tool
loop doesn't use.

## Decision

A small MCP server in `mcp/` sits in front of the retrieval API and exposes
three read-only tools. `search` calls a new retrieval-only endpoint, so it needs
no generation key; `ask` returns a grounded answer; `list_sources` lists the
indexed documents. The server runs as its own process, as a container in the
retrieval stack, and speaks Streamable HTTP, with stdio for desktop clients. Its
results are sized for the agent loop, which shows the model at most 2000
characters of a tool result.

On the agent side, an adapter lists the server's tools and turns each one into
an ordinary registry `ToolSpec`, with an input model built from the tool's JSON
Schema. A policy table in the agent decides which server tools exist for it
(deny by default), which specialists own them, their per-task rate limits and
whether a call needs approval. They are registered as `rag_search`, `rag_ask`
and `rag_list_sources`. The registry therefore governs them exactly as it
governs local tools, and neither the registry nor the tool loop changed.

Each call opens a short-lived MCP session on its own event loop, inside the
registry's worker thread, and closes it; nothing outlives the loop that created
it (see [ADR 0003](0003-one-event-loop-per-worker.md)). Discovery happens on the
first task run that can reach the server, not at import. A run that can't reach
it goes ahead with the local tools, and the next run tries again. With
`MCP_RAG_URL` empty, the default, none of this happens.

## Consequences

- The agent, not the server, enforces permissions and rate limits. Another MCP
  client of the same server gets no limits; the tools are read-only, and the
  server binds to localhost unless told otherwise.
- Server tool descriptions reach the model's prompt verbatim, so a malicious or
  compromised server could plant instructions there (tool poisoning). The
  policy allow-list limits which tools exist; screening descriptions and
  retrieved text is left to a later guardrails pass.
- There is no transport authentication yet: the server trusts its network until
  the Azure deployment puts it behind internal-only ingress.
- `ask` spends the retrieval service's LLM budget, which the agent's cost
  tracking doesn't count. Each result carries its `cost_usd`, so the evaluation
  work can add it up from the tool-invocation log.
- A fresh session per call costs about 30 ms on a local network. In the live
  check, a whole `rag_search` took about 0.7 s, most of it hybrid retrieval with
  a CPU reranker.
- `search` returns 3 hits with 300-character snippets by default. Measured on
  the sample corpus as the loop sees it, 5 hits of that size come to 2541
  characters in the worst case, and 3 hits to 1588. The alternative of 5 hits
  of 150 characters (1757) trades evidence per hit for breadth; the retrieval
  evaluation should settle which is better.
- The MCP SDK takes most of a second to import, so the agent imports it only
  when `MCP_RAG_URL` is set.

## Alternatives

- **`langchain-mcp-adapters`.** Requires the 1.x SDK, and produces LangChain
  tools that the tool loop doesn't take.
- **Mounting the MCP server inside the retrieval API's process.** Saves one
  local HTTP hop, but ties the server to the retrieval service's heavy
  dependencies and to its single-process vector store.
- **Async tool handlers and an async registry call.** A change to the registry
  and the loop, for no gain at this scale.
- **Trusting the server's tool annotations** (read-only, destructive) for
  permissions or approvals. The SDK's own documentation says annotations from
  other servers are hints, never guarantees.
