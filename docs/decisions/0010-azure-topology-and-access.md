# 0010. Three Container Apps, one Postgres server, and IP allow-lists for access

Date: 2026-11-08. Status: accepted.

## Context

The system has three processes: the retrieval API, the MCP server in front of it, and
the agent API. The budget is the free amounts: Container Apps' monthly grant (180,000
vCPU-seconds and 360,000 GiB-seconds), and Postgres Flexible Server's B1ms offer of 750
hours a month in an account's first year. The clients (the evaluation harness, three
Streamlit UIs, the MCP server's calls to the retrieval API and the agent's MCP client)
send no credentials; ADR 0004 left the MCP server trusting its network until it sat
behind internal-only ingress. The published retrieval numbers come from one collection,
`toolchain_docs`, embedded once.

## Decision

- **Apps.** One Consumption environment with three Container Apps, each scaling between
  0 and 1 replicas: `groundwork-rag` (1 vCPU / 2 GiB), `groundwork-mcp` (0.25 / 0.5) and
  `groundwork-agent` (0.5 / 1, with an init container that runs the migrations and the
  demo seed). With all three active that's 1.75 vCPU and 3.5 GiB, so the grant covers
  about 28 hours a month.
- **Database.** One B1ms server, PostgreSQL 16, 32 GB, with two databases (`rag` and
  `orchestrator`) and the `vector` extension allow-listed before any app starts. Public
  access with two firewall rules, Azure services and the operator's IP; TLS required;
  one admin login for both apps.
- **Data.** `make up` copies `toolchain_docs` from the local database, vectors included,
  and checks a checksum over every column; nothing is re-embedded.
- **Access.** The retrieval and agent apps are public with an IP allow-list holding only
  the operator's address; the MCP server has internal ingress only. A probe from inside
  the MCP app on the first deploy decided how it reaches the retrieval API: the public
  address answered 403, because the allow-list filters it even from inside the
  environment, and the internal name `http://groundwork-rag` answered 200. So the MCP
  server calls the internal name, as the agent calls `http://groundwork-mcp`.

## Consequences

- The allow-list protects only the public addresses. Any app in the environment can call
  the retrieval API by its internal name without authentication; today those are these
  three apps. A changed home IP means running `make up` again (about a minute).
- `db_query` reads through the agent's own connection under a read-only transaction, so
  it can read every table in the agent's database, not just `demo`. Behind the
  allow-list that is the operator's own data; per-app roles are out of scope.
- **The cloud serves the evaluated vectors, but not always the same hits.** HNSW search
  is approximate, and a rebuilt index makes different misses. Copying the collection
  locally showed it before the deploy: rows and vectors bit-identical, 142 of 150 hit
  lists identical. Against Azure (2026-11-08, 75 questions in both modes): 71 of 75 dense
  and 69 of 75 hybrid top-10 lists identical; the refusal gate decided the same on all
  75 questions in dense mode and on 73 in hybrid. For every question that differed, an
  exact search returned the same top 20 on both databases, with the same similarities;
  each difference was one side's index missing a nearer chunk (Azure's index missed 13
  of q046's exact top 20, the local one missed q037's nearest), except one tie between
  two chunks of equal score, ordered differently because OpenAI's query embeddings vary
  slightly between calls. Aggregates moved little (dense MRR@10 0.678 → 0.689, hybrid
  R@10 0.761 → 0.749). The published numbers include the local index's misses; exact
  retrieval (a higher `hnsw.ef_search`, or no index at this corpus size) would remove the
  difference and is a candidate for later.
- Cold starts: a scaled-to-zero app answers its first request in about 35 s, and the
  first hybrid search loads the reranker (11 s). The agent's MCP client waits 30 s, so a
  cold retrieval API can fail the first tool call; `make warm` wakes both public apps.

## Alternatives

- **MCP as a sidecar.** Beside the retrieval API, the agent would need a second ingress
  port on that app; beside the agent, its call to the retrieval API would still cross
  that app's ingress.
- **One app with all three containers.** No service boundary, and the retrieval API would
  be out of the evaluation harness's reach.
- **A virtual network with private endpoints.** About $7 a month for the endpoint plus a
  private DNS zone, for a stack that is up a few hours at a time.
- **An API key on the retrieval API**, the planned fallback had the probe failed. The
  internal name made it unnecessary; it remains the way to protect the API from other
  apps in the environment.
- **Entra sign-in in front of the apps.** An app registration and a token flow in every
  client: about a day's work for an MVP.
- **Re-embedding inside Azure on every `make up`.** About $0.01 and 12 minutes each time,
  and vectors that aren't bit-identical to the evaluated ones. `pg_dump` was rejected
  too: it copies every collection's registry rows and is harder to scope.
