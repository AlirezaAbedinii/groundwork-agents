# mcp

The retrieval service as MCP tools. `server.py` is a thin MCP server in front of
the RAG HTTP API (`services/rag`): each tool call becomes one request to that
API, and the result comes back as structured content sized for an agent's
context. The agents in `services/agent` discover these tools over MCP and run
them through their tool registry (see `MCP_RAG_URL` there).

## Tools

All three are read-only.

| Tool | RAG endpoint | Returns |
|---|---|---|
| `search(query, top_k=3, mode=None)` | `POST /v1/search` | The best-matching passages, up to 8: source file, section, score, the text, and whether the text is the whole passage. Retrieval only, so it needs no generation key. |
| `ask(question, mode=None, top_k=None)` | `POST /v1/ask` | A grounded answer with `[n]` citations, a confidence score and the request's cost. Needs the RAG service's generation key. |
| `list_sources(prefix=None)` | `GET /v1/documents` | The indexed documents (those whose path starts with `prefix`) with their chunk counts. |

`mode` is `hybrid` (keywords and meaning, reranked) or `dense` (meaning only);
left out, the RAG service's default applies. The agent shows its model up to
4000 characters of each tool's JSON, and `search` and `list_sources` size their
results to 3600 (`tool_models.py`), so they arrive whole:

- `search` fills hits in rank order with whole passages while the result fits.
  Each hit after that gets an equal share of what is left, filled with the
  window of consecutive lines that holds the most distinct query words (of three
  or more characters), or the passage's start when no such window fits; a share
  a hit doesn't use passes to the next. `complete` is true for a whole passage.
  At the default `top_k` of 3, each of the evaluation's 150 searches of the
  public docs corpus came back as three whole passages; with more hits, the
  later ones are shorter excerpts.
- `list_sources` sends the whole list when it fits. A longer one comes as a
  count of documents and chunks per directory, then the first documents and
  `truncated: {"shown", "of"}`; a directory is a `prefix` that lists its
  documents.
- `ask` leaves out the passages it answered from.

The evaluation measures `search`'s results with this same code: see "What MCP
search returns to an agent" in [`eval/reports/rag.md`](../eval/reports/rag.md).

When the RAG API fails, the tool returns an error the model can read and act on,
not a crash: `retrieval service returned 503: <detail>`, or
`retrieval service unreachable at <url>`.

## Run it

With the RAG stack in Docker (from `services/rag`):

```bash
docker compose up -d --build --wait api mcp   # MCP at http://localhost:8001/mcp
docker compose run --rm seed                  # index the sample corpus
```

Search works without any API key when the RAG service embeds locally; see the
"Keyless search" block in `services/rag/.env.example`. `ask` still needs the
generation key.

On the host, against a RAG API at `RAG_API_URL` (default `http://localhost:8000`):

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
.venv/bin/python server.py                      # Streamable HTTP on 127.0.0.1:8001/mcp
.venv/bin/python server.py --transport stdio    # for desktop MCP clients
```

`--host` and `--port` change where it listens. Bound to a loopback address, the
server accepts only localhost `Host` headers. It has no authentication, so keep
it on localhost or an internal network.

In Azure ([`infra/`](../infra/README.md)) it is a Container App with internal ingress
only: it has no public address, and the agent reaches it at `http://groundwork-mcp/mcp`.
It calls the retrieval API by its internal name, `RAG_API_URL=http://groundwork-rag`,
because the retrieval API's IP allow-list filters calls to its public address even from
inside the environment ([ADR 0010](../docs/decisions/0010-azure-topology-and-access.md)).

A desktop MCP client starts it over stdio with a config like this:

```json
{
  "mcpServers": {
    "groundwork-rag": {
      "command": "/path/to/groundwork-agents/mcp/.venv/bin/python",
      "args": ["/path/to/groundwork-agents/mcp/server.py", "--transport", "stdio"],
      "env": {"RAG_API_URL": "http://localhost:8000"}
    }
  }
}
```

## Check it

```bash
.venv/bin/python smoke.py http://localhost:8001/mcp
```

`smoke.py` prints the negotiated protocol, the tools, and what `list_sources`,
`search` and `ask` return; tool errors are printed rather than raised. Against
the seeded sample corpus with local embeddings and no key:

```
protocol 2026-07-28
tools ['ask', 'list_sources', 'search']
list_sources: 7 sources, 37 chunks
search 'FERRY-429' (hybrid):
  1. 0.999  05-rate-limits.md  What happens when you exceed the limit
  2. 0.998  04-error-codes.md  Request errors
  3. 0.962  README.md  Files
ask: tool error: Error executing tool ask: retrieval service returned 503: Missing required configuration: OPENAI_API_KEY. ...
```

The tests need no running service: they put a fake RAG API behind the real
server (`make test-mcp` from the repo root, or `.venv/bin/python -m pytest` here).
