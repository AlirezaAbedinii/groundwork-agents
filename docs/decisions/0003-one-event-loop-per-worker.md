# 0003. One event loop per Celery worker thread

Date: 2026-09-26. Status: accepted.

## Context

[ADR 0002](0002-native-tool-calling.md) made the LLM client async. The Celery
wrappers ran each task with `asyncio.run()`, so every task got a fresh event
loop. The first run of the live smoke tests against OpenAI showed what that
does to a process that makes LLM calls on more than one loop: the first call
worked, and a later call on a new loop failed inside the HTTP connection pool
with "Event loop is closed". LangChain caches one async HTTP client per provider
for the whole process, and its connections belong to the loop that opened them.
A worker process runs many tasks, so every task after its first would have
failed the same way. The API server was not affected, since it runs on one
loop, and neither were the test suites, whose fixture player makes no HTTP
calls.

The same run settled the question ADR 0002 left open: OpenAI's strict mode
accepts the plan schema as generated, `minimum`, `maximum` and `default`
included, so the schema needs no rewriting.

## Decision

Celery tasks and beat jobs run their coroutines on one long-lived event loop
per worker thread (`workers/loop.py`) instead of `asyncio.run()` per task. The
API already runs on a single loop, and the live smoke tests now run on one loop
too, one after another, the way a worker runs its tasks.

## Consequences

- A worker reuses its HTTP connections to the providers across tasks.
- Anything a task leaves scheduled on the loop would run during that thread's
  next task. The graph finishes its own work before `ainvoke` returns, so
  nothing is expected to.
- The checkpointer still opens one database connection per run (ADR 0002): the
  test client makes a new loop for every request.
- New entry points that run coroutines (a script, a notebook) must also keep to
  one loop per thread, or build their own clients per loop.

## Alternatives

- **Fresh chat models and HTTP clients per task.** The OpenAI chat model takes
  an explicit async HTTP client, but the Anthropic one has no public way to do
  the same, so this would depend on library internals.
- **Clear LangChain's cached clients between tasks.** Also internals, and it
  gives up connection reuse.
- **Keep `asyncio.run()` per task.** Broken as described above.
