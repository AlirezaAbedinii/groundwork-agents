# mcp

The MCP layer lands here in Phase 2: a server that exposes the RAG service as
MCP tools (`search`, `ask`, `list_sources`) and the client adapter through which
the agent orchestrator calls them, so the tool registry applies the same
permissions and rate limits to MCP tools as to local ones. Empty until then.
