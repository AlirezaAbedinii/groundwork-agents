# 0002. Native tool calling, structured outputs, and an async core

Date: 2026-09-26. Status: accepted.

## Context

The agent service's specialists asked the model to write one JSON action per
turn as plain text (`{"action": "tool", "tool": ..., "arguments": ...}`),
extracted it with a hand-written parser, retried once when that failed, and
ran one tool per turn. A failing tool raised an exception that failed the
whole attempt. Plans, review verdicts and memory extraction used the same
prompt-and-parse path: a JSON template in the prompt and a parser on the reply.
Everything ran synchronously, so each task held a worker thread for its whole
run, most of it waiting on providers.

Both providers now offer native tool calling (typed arguments, several calls
per turn, results tied to call ids) and structured output constrained to a
JSON schema.

## Decision

**Specialists use native tool calling.** Each tool is declared to the provider
from its `ToolSpec` input schema. The model may ask for several calls in one
turn, and every call is answered with one tool result before the model's next
turn. Sensitive calls go to the human-approval gate before any call of the
turn runs; the approved calls run concurrently. A tool error is a result the
model sees (`status="error"`), not an exception: an attempt fails only when the
turn budget runs out or when something other than a tool error escapes.
`MAX_TOOL_ITERATIONS` counts model turns, and a batch of calls is one turn.

**Plans, review verdicts and memory extraction use native structured output**
(JSON schema, strict mode on OpenAI). The schema replaces the JSON template in
the prompt, and the field meanings the template carried move into the schema's
field descriptions, which both providers receive. Pydantic still checks what a
schema cannot: the plan's dependency rules, and score ranges that Anthropic
does not enforce while decoding. Those failures keep their single retry with
the error in the prompt.

**Async stops where the waiting stops.** The LLM client and the four graph
nodes that call it are async, and the graph runs under `ainvoke` with the async
Postgres checkpointer. Bookkeeping nodes, the tool registry and the tool
handlers stay synchronous; tools run in worker threads. The checkpointer's
connection is opened per run, because the saver is bound to the event loop that
created it and runs happen on short-lived loops (one per Celery task, one per
test-client request).

## Consequences

- A model can recover from a bad call (wrong arguments, a rate limit, a denied
  request) inside the same attempt. The cost: a model that keeps calling a tool
  that does not exist now spends its whole turn budget before the attempt fails.
- A tool-calling turn has little or no text, so replay needs the calls
  themselves: `llm_calls.tool_calls` stores them, and a strict replay serves
  them back.
- Approval keys carry the attempt and the call's position in its turn, so two
  sensitive calls to the same tool in one turn get two approvals.
- Two concurrent calls to the same tool can both pass the per-task rate-limit
  check, because the registry counts past executions before it runs a call.
  At today's limits (5 to 20 calls per task) this is accepted, not fixed.
- Each run opens one database connection and compiles the graph, a few tens of
  milliseconds against runs that take seconds.
- Repository writes and memory lookups inside the async nodes still block the
  event loop briefly. They are the first thing to move to threads if the inline
  server shows stalls.
- OpenAI's strict mode accepts a subset of JSON Schema. Whether it takes the
  `minimum`, `maximum` and `default` keywords the plan schema carries is not
  verified yet; the live smoke test (`pytest -m live`) exercises it once a
  funded key is available.

## Alternatives

- **Plans as a forced tool call** ("submit_plan") instead of a JSON-schema
  response format. It works on older models, but it runs plan parsing through
  the tool-call machinery for no gain; JSON-schema output is the direct feature
  on both providers.
- **Keep the graph synchronous and bridge each LLM call with `asyncio.run`.**
  No graph changes, but a new event loop per call rules out connection reuse
  and running a tool batch concurrently on the loop.
- **Async all the way down**, including tool handlers, the registry and the
  database layer. The most uniform option, but a rewrite of every tool and the
  repository layer for no measurable gain today: tools already run concurrently
  in threads.
- **Cache the async checkpointer per event loop.** It leaves one open database
  connection behind for every short-lived loop that has finished.
