# 0001. One repository for both services, imported with git subtree

Date: 2026-09-21. Status: accepted.

## Context

The retrieval service and the agent orchestrator started as separate
repositories (`rag-hybrid-search` and `agent-orchestration-system`), each with
its own history, CI and README. The next phases connect them: the agents call
the retrieval service over MCP, one evaluation suite measures the whole system,
and one Terraform module deploys both. Two repositories would mean two READMEs
describing half a system each, two CI pipelines that cannot test the
integration, and a deployment that belongs to neither.

Both histories were worth keeping. They record how each service was built and
measured, and rewriting them would change every commit hash that the old
repositories and their CI runs refer to.

## Decision

One repository, `groundwork-agents`, with the services under `services/rag` and
`services/agent`, and top-level `mcp/`, `eval/` and `infra/` for the parts that
span both.

Each old repository was imported with `git subtree add --prefix=services/<name>`
without squashing. The import is a merge commit whose second parent is the old
repository's tip, so every original commit keeps its hash, author and date, and
`git blame` inside `services/` still points at the original commits.

Each service keeps its own `pyproject.toml`, ruff configuration, Makefile,
Dockerfiles and Compose stack. The root holds only what spans both: the CI
workflow (one job per service, run from the service directory), a Makefile that
delegates to the services, the Dependabot configuration and the README.

The original repositories stay online as they are; development continues here.

## Consequences

- Path-scoped history needs a flag: `git log -- services/rag/<file>` stops at
  the import commit by default; `git log --full-history -- <path>` crosses into
  the original commits. `--follow` does not cross the boundary.
- Agent commands run from `services/agent`, because its fixtures, `alembic.ini`
  and `.env` are resolved relative to the working directory. The root Makefile
  and the CI workflow do this.
- The lint rules differ between the services (the retrieval service enforces a
  wider ruff rule set at 100 columns, the agent service a narrower set at 120).
  A shared configuration waits until there is Python code at the root.
- CI runs both services on Python 3.12. The retrieval service's Dockerfile still
  builds from `python:3.11-slim`; it moves to 3.12 when the images are rebuilt
  for deployment in Phase 4, so that CI tests the interpreter the images ship.
- The original repositories keep their hashes, so links into them stay valid,
  and every one of their commits is also reachable here under `services/`.

## Alternatives

- **Rewrite the histories with `git filter-repo`** so every original commit
  already places its files under the prefix. Path-scoped `git log` would then
  work without flags, but every hash changes and the imported history no longer
  matches the original repositories.
- **Squashed import** (`git subtree add --squash`): one commit per service and
  no history. Rejected for the same reason: the history is part of the record.
- **Three repositories**, the two services plus a platform repository for MCP,
  evaluation and infrastructure. The services stay independent, but the
  platform repository has to pin and check out both to test anything, and the
  split README problem remains.
