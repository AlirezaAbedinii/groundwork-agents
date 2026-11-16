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

## Addendum, 2026-11-15: each result as valid JSON within its tool's budget

The loop showed the model the first 2000 characters of a result's JSON. A longer
result was cut wherever the 2000th character fell, often inside a hit or a
string, so the model got text that did not parse and no sign of what was left
out. That fixed slice is replaced, and with it the 2000-character limit
described above.

- **Valid JSON that says what it left out.** `tools/serialize.py` renders every
  result. One within its budget is exactly the JSON it was before. A longer one
  loses whole items from the end of its longest list, at the top level or under
  `content` (where the MCP tools' results are), and gains
  `"truncated": {"shown": n, "of": m}` right after that list. When there is no
  list, or not even one whole item fits, the longest strings are shortened
  first, each ending in `…[+N chars]`. Every result parses and fits its budget.
- **A budget per tool.** `ToolSpec.result_budget_chars`, 2000 by default and at
  least 200. The agent's MCP policy sets it for server tools: `rag_search` and
  `rag_ask` get 4000, because search hits and an answer's citations are the
  evidence the research specialist works from; `rag_list_sources` keeps 2000.
  A search with `top_k` 10 arrives as whole hits, as many as fit, with their
  count.
- **The server sends what it sent before.** Its results at the default
  arguments are still sized to fit 2000 characters (3 hits with 300-character
  snippets), so they arrive whole under either budget.
- **Cost.** Only a search or an answer longer than 2000 characters costs more:
  up to 2000 more characters, roughly 500 prompt tokens, in each later turn of
  that attempt.

Alternatives: a larger slice still cuts inside the JSON; a limit inside each
tool repeats the same logic in every tool and can't reach another MCP server's
tools; summarizing long results with a model adds a paid call, latency and a
result that differs from run to run.

## Addendum, 2026-11-16: whole passages within one budget, a summarised source list, ask first

`search` cut every hit to its first 300 characters and took `top_k` up to 20, so
asking for more hits never showed more of any one hit. On the public docs corpus's
test split, the top 3 hits hold 0.691 of the questions' evidence as whole chunks
and 0.272 as snippets in dense mode (hybrid 0.647 and 0.208): a hit's evidence
mostly sits past its first 300 characters. `list_sources` sent every document,
about 17,000 characters for that corpus, of which the loop showed the first 2000.
Nothing told the research specialist when to ask rather than search.

- **One budget per search result.** `top_k` is at most 8 (default 3). The result's
  JSON is at most 3600 characters, inside the 4000 the agent gives the tool once it
  wraps the result in `{"content": …}`. Hits take their whole text in rank order
  while the result fits. From the first that doesn't, each takes an equal share of
  what is left, and a share it doesn't use passes to the hits after it. A share is
  filled with the window of consecutive lines that holds the most distinct query
  words (lower-cased, longer than two characters; the earliest of equal windows),
  or with the passage's start when no window with a query word fits. Every hit
  says whether its text is `complete`; `text` replaces `snippet`. Hits keep their
  metadata, the echoed query is cut at 200 characters, and only metadata too long
  to fit at all drops hits from the end, so no result is over the budget.
- **`list_sources(prefix)`.** The server filters by path prefix (the RAG API is
  unchanged). A list over the budget becomes a count of documents and chunks per
  directory, as many as fit in path order, then the first documents, with
  `truncated: {"shown", "of"}`; a directory is a prefix that lists its documents.
  The agent's budget for this tool rises from 2000 to 4000, like the other two.
- **Ask first.** The research role says to call `rag_ask` first for a question
  the documentation should answer; to use `rag_search` to find where something is
  documented, quote exact wording, gather material for a longer document or
  double-check a refusal; and to report a refusal that no search overturns as not
  documented, never filling the gap from the model's own knowledge. The `ask`
  description says the same.
- **Measured with the server's own code.** The sizing lives in
  `mcp/tool_models.py` as functions of plain dicts. The retrieval evaluation loads
  that file and records, for every recorded search, what each hit covers as
  `search` returns it with `top_k` 3, 5 and 8.

Consequences:

- Evidence recall on the test split, as `search` returns it (and as whole chunks):

  | `top_k` | dense | hybrid |
  |---|---|---|
  | 3 | 0.691 (0.691) | 0.647 (0.647) |
  | 5 | 0.730 (0.740) | 0.672 (0.725) |
  | 8 | 0.652 (0.794) | 0.681 (0.784) |

- At the default `top_k` nothing is lost: all 150 recorded searches came back as
  three whole passages. At 8, the hits' metadata takes about half the budget, so
  fewer of the top hits fit whole and the rest get excerpts of about 50
  characters; dense recall at 8 is below dense recall at 3. The `top_k`
  description tells the model that more hits leave less of each.
- Which words count, and which of equal windows wins, made no difference on the
  dev split (substring against whole-word matching, earliest window against the
  most matches in total); the rule takes whole words and the earliest window.
- A default search's result grows from about 1600 to up to 3600 characters,
  roughly 500 more prompt tokens in each later turn of that attempt.
- `snippet` is gone, so a client that read it breaks. The agent passes results to
  the model as JSON and reads no field of them; a contract test pins the new shape
  through the real server, adapter and registry.
- The source summary groups by full directory path: the public corpus has 31
  directories, about 2000 characters of counts. A corpus with many more shows the
  first that fit, and `prefix` narrows the list.

Alternatives: a fourth tool to fetch a whole chunk is outside the roadmap's tool
set, and whole passages at the default `top_k` remove most of the need for it.
Equal excerpts for every hit give up the whole top passages, which hold most of the
evidence. A budget that grows with `top_k` would also need a larger budget in the
agent for every value. Dropping the hits that only get a short excerpt would return
fewer hits than asked, though their source and section still say where to look.
