# Architecture decision records

One short file per decision that shaped this repository, numbered in the order
the decisions were made. An ADR is not edited after the fact; a later ADR
supersedes it.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-monorepo-layout.md) | One repository for both services, imported with git subtree | accepted |
| [0002](0002-native-tool-calling.md) | Native tool calling, structured outputs, and an async core | accepted |
| [0003](0003-one-event-loop-per-worker.md) | One event loop per Celery worker thread | accepted |
| [0004](0004-mcp-as-the-seam.md) | MCP as the seam between the agents and retrieval | accepted |
| [0005](0005-pgvector-over-chroma.md) | Postgres with pgvector instead of Chroma | accepted |
| [0006](0006-evaluation-harness-and-judge.md) | One evaluation suite over HTTP, evidence-quote labels and a judge from another provider | accepted |
| [0007](0007-refusal-threshold-calibration.md) | Refusal thresholds calibrated per retrieval mode on held-out questions | accepted |
| [0008](0008-agent-evaluation.md) | Agent tasks scored on their deliverables and their invocation log | accepted |
| [0009](0009-inline-runs-in-the-cloud.md) | Inline runs in the cloud, with working memory in Postgres and no code execution | accepted |
| [0010](0010-azure-topology-and-access.md) | Three Container Apps, one Postgres server, and IP allow-lists for access | accepted |
| [0011](0011-terraform-state-identity-and-secrets.md) | Two Terraform configurations, secrets only in Key Vault, and OIDC for CI | accepted |
| [0012](0012-traces-to-application-insights.md) | Traces to Application Insights beside the agent's own store | accepted |
| [0013](0013-a-hard-time-limit-on-the-stack.md) | A hard time limit on the stack instead of a spending cap | accepted |
| [0014](0014-an-agent-reliability-phase-before-presentation.md) | An agent reliability phase before presentation | accepted |

## Format

`NNNN-short-title.md`, four sections:

- **Context**: the situation and the constraints that forced a choice.
- **Decision**: what was decided, in one or two sentences.
- **Consequences**: what becomes easier, what becomes harder, what to watch.
- **Alternatives**: what else was considered and why it lost.
