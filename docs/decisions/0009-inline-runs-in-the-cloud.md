# 0009. Inline runs in the cloud, with working memory in Postgres and no code execution

Date: 2026-11-08. Status: accepted.

## Context

Locally the agent API hands each run to a Celery worker through Redis; Redis also holds
a task's working memory (its plan, subtask outputs and errors, which the approval chat
reads back), and a beat process runs memory maintenance. The `code_exec` tool starts a
sibling container through the host's Docker socket. On Azure Container Apps there is no
Docker socket, a managed Redis is a paid service, and every app scales to zero when
idle. The agent already had an inline mode, the settings default: `POST /tasks` and an
approval's resolve answer 202 and run the graph as a FastAPI background task in the
API's own process. Checkpoints live in Postgres, so a paused task survives a restart; a
run in progress doesn't. Measured locally over 26 evaluation runs, a task takes a median
of 80 s and at most 433 s, approval waits included.

## Decision

In Azure the agent runs inline (`RUN_MODE=inline`) in one replica, with a scale-in
cooldown of 600 s. Working memory moves to Postgres behind a setting,
`WORKING_MEMORY_BACKEND=postgres`: a `working_memory` table with the Redis class's
methods and expiry semantics, while `redis` stays the default so local behavior is
unchanged. Code execution is off (`CODE_EXEC_BACKEND=disabled`): the tool is left out of
the registry, and `/config` reports it. Celery, Redis, the worker and the beat jobs stay
in the local compose stack.

## Consequences

- The cloud agent needs no Redis and no worker app. An inline run with Redis pointing at
  a closed port completes, which the end-to-end suite checks.
- A run in progress lives in one process. A replica that scales in mid-run loses the
  run; the 600 s cooldown outlasts every measured run segment, at the price of about five
  extra minutes of activity per session. A task paused for approval survives scale to
  zero, and its approval chat still finds its working memory.
- One replica only: inline runs and the MCP tools discovered at startup live in that
  process, so the agent can't scale out as configured. In the cloud check a real task
  completed in 146 s on that one replica.
- The two code tasks in the agent suite can't pass in the cloud; the analysis and code
  specialists keep `db_query` and the file tools.
- Memory maintenance doesn't run in Azure. Every `make down` deletes the database, and
  with it the memories, so nothing accumulates.

## Alternatives

- **Celery and Redis in Azure.** A Redis container app needs storage that survives scale
  to zero, or a paid managed cache, plus a worker app; all of it to run one task at a time.
- **The in-process working memory.** The approval chat would lose its snapshot whenever
  the replica scaled to zero while a task waited for a human.
- **Dropping the Redis backend.** It would change local behavior for no gain.
- **The `subprocess` backend with a scrubbed environment.** Model-written code would
  still run in the API's container, with its network and as a user who can read the
  process's environment, which holds both provider keys and the database password.
- **Container Apps dynamic sessions.** Sandboxes built for exactly this, but billed per
  session and outside this phase's scope; a candidate for later.
