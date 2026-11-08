# 0012. Traces to Application Insights beside the agent's own store

Date: 2026-11-08. Status: accepted.

## Context

The agent records every decision as an OpenTelemetry span and exports it synchronously
to its own Postgres database, which its trace explorer, cost views and replay read. The
retrieval API had no OpenTelemetry at all: it kept stage timings and one SQLite row per
`/v1/ask`, which a container loses on every restart. In Azure both services need traces
that outlive their replicas and can be queried in one place, within the free 5 GB of
logs a month. The Azure Monitor exporter for OpenTelemetry has only ever shipped as
betas.

## Decision

When `APPLICATIONINSIGHTS_CONNECTION_STRING` is set, both services export their spans to
one workspace-based Application Insights resource; without it nothing changes, locally
or in CI.

- The agent adds a batched Azure Monitor processor beside its synchronous Postgres one,
  on the same provider, and flushes it on shutdown, when a replica scales in.
- The retrieval API gets a provider of its own for FastAPI's request spans, with the
  pipeline's results as attributes (`rag.mode`, `rag.refused`, `rag.cost_usd`, the stage
  timings). `/v1/stats` stays as it was.
- `service.name` is `groundwork-agent` or `groundwork-rag`, and `service.version` is the
  image tag, baked into the image at build time so a deploy from CI can't leave it stale.
- No sampling: a task is a few hundred spans. The workspace is capped at 0.15 GB a day
  and Application Insights at 0.1 GB, under the free amount for a 31-day month.
- The exporter is pinned exactly (`azure-monitor-opentelemetry-exporter==1.0.0b58`), and
  `/health` is left out of the request spans, since the platform's probes would be noise.

## Consequences

- Costs are visible in Azure exactly as the agent computes them: on the cloud check, the
  task's 21 LLM spans carried `cost_usd` attributes summing to $0.048324, the figure the
  agent's `/traces/<id>/costs` reported.
- The agent's explorer, replay and cost views work in the cloud unchanged, against the
  cloud database. Both stores belong to the stack: the database and the workspace
  (30-day retention) are deleted at `make down`, so a session's traces have to be read,
  or exported, before it ends.
- A run's spans and the retrieval calls it causes are separate traces: no trace context
  crosses the MCP hop, so agent and retrieval spans share no operation id.
- Errors that a step catches and retries appear only as span attributes and in the
  subtask's record, not in the container's log: when a revoked API key made every review
  fail with 401, the log showed only HTTP access lines. Look in `/traces/<id>` or in
  Application Insights.
- The exporter also starts statsbeat, its own usage metrics sent to Microsoft rather
  than to this resource, and a retry thread spooling under `/tmp`. Both are left at their
  defaults; `APPLICATIONINSIGHTS_STATSBEAT_DISABLED_ALL=true` turns statsbeat off.

## Alternatives

- **The `azure-monitor-opentelemetry` distro.** It installs its own global provider,
  which would displace the one the agent's Postgres exporter hangs on.
- **OTLP through a collector.** Another container to run and pay for.
- **Replacing the agent's Postgres exporter with Application Insights.** The explorer,
  cost views and replay query spans with SQL next to the tasks they belong to, and they
  must keep working locally with no Azure account.
- **Sampling.** At a few hundred spans per task, there is nothing to save, and a sampled
  trace can't account for a task's cost.
