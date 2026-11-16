# 0016. Tool contracts the model can act on

Date: 2026-11-16. Status: accepted.

## Context

A specialist knows a tool only by its name, its description and its input schema. Four
of those contracts left out what the model needed in order to use the tool well:

- `db_query` said it queried "the demo database schema" and named no table, so the
  model guessed table names. An unqualified guess could also resolve to the agent's own
  tables in `public` (`plans`, `tool_invocations`), which share the `orchestrator`
  database with `demo`.
- `code_exec` reported the sandbox's streams and exit code, and the registry logged the
  call as a success whenever the sandbox ran. Code that only defined a function came
  back as two empty strings, with nothing to say why.
- Each tool has a per-task call limit that the model could not see until a call was
  refused.
- `web_search` stayed registered without a Tavily key, answered by DuckDuckGo's
  instant-answer API. That API returns results only for queries that match an
  encyclopedia entry, so the tool cost turns and calls and seldom returned anything.

## Decision

- **`db_query`.** When the registry is built, the description gets the `demo` schema's
  tables and columns from `information_schema.columns`, for example `Tables:
  demo.vector_db_stats(name text, github_stars integer, license text)`. If the database
  can't be read, or the schema is empty, the description stays the static sentence.
  Every query runs with `SET LOCAL search_path TO demo`, after `SET TRANSACTION READ
  ONLY`, so an unqualified name resolves to `demo` for that transaction only.
- **`code_exec`.** The result gains `ok` (`exit_code == 0`) and a `note` when both
  streams are empty: "The code ran and printed nothing. Print the values you need." A
  non-zero exit is still a successful call in the invocation log. The tool did its job
  and the model reads `ok`.
- **Limits.** `tool_definitions_for(specialist, task_id)` ends each description with
  "Limit: N calls per task, M left.", counted the way the registry counts calls toward
  its rate limit. The specialist loop reads the definitions again on every turn.
- **`web_search`.** Registered only when `TAVILY_API_KEY` is set, as `code_exec` is
  registered only when there is a sandbox. Under `MOCK_LLM` it stays, with canned
  results, so the deterministic tests and fixtures are unchanged.

## Consequences

- The `search_path` setting is hygiene, not a security boundary. `db_query` connects
  with the same login as the rest of the agent, and that login owns the agent's tables
  (ADR 0010: one admin login for both apps). A qualified `SELECT * FROM public.plans`
  still runs, and an integration test pins that so nobody mistakes the setting for a
  restriction. What stops writes is the read-only transaction and the single-statement
  `SELECT`/`WITH` check. Keeping the model away from the agent's own tables would need
  a database role that can read only `demo`, which is a separate change.
- The description is read once per registry, so per process. A table added to `demo`
  after the process starts appears after a restart. If the database is unreachable when
  the registry is built, the process keeps the static description until it restarts.
  The API builds its registry on the first run, when the database is already needed for
  the run itself.
- The limit text changes from turn to turn. The definitions are not part of the mock
  client's fixture key, so no fixture changed.
- Without a Tavily key the research specialist has the `rag_*` tools and `api_call`,
  and nothing that searches the web. Its role sentence now says it uses whichever of
  these tools it is given.
- Each turn costs one count query per offered tool against the invocation log.

## Alternatives

- **A read-only role for `db_query`.** This is the real boundary. It needs a second
  login, its grants and its secret in both deployments, which is more than this change
  calls for. It is left for later, and nothing here depends on it.
- **Reject SQL that names a schema other than `demo`.** Parsing SQL to find every
  relation it touches is easy to get wrong, and a wrong parse would look like a
  boundary.
- **Keep `web_search` and the DuckDuckGo fallback.** A tool that seldom answers takes
  calls and turns from tools that do.
- **Count a non-zero exit as a failed `code_exec` call.** A failed call answers with
  an error and no streams. The model needs the traceback in `stderr` to fix its code.
