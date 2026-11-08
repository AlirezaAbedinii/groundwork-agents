# infra

Terraform and scripts that run Groundwork on Azure: the retrieval API, the MCP server
and the agent API as Container Apps, one Postgres Flexible Server with pgvector, Key
Vault for the keys, and Application Insights for traces. The stack is up only while
it's in use: `make up` creates it in about 10 minutes, `make down` deletes it, and a
scheduled GitHub workflow deletes any stack left up past a time limit. Images come from
ghcr.io, and image updates are deployed from GitHub Actions over OIDC.

## Architecture

```
your machine (allow-listed IPv4)          Azure, canadacentral
  eval harness, 3 Streamlit UIs, curl      rg-groundwork  (make up / make down)
        │ HTTPS                              cae-groundwork (Consumption; logs → log-groundwork)
        ├──────────────▶ groundwork-agent   external, allow-list · 0–1 replicas · init: migrations + demo seed
        │                     │ http://groundwork-mcp/mcp (internal)
        │                     ▼
        │               groundwork-mcp      internal only · 0–1
        │                     │ http://groundwork-rag (internal name)
        └──────────────▶ groundwork-rag     external, allow-list · 0–1 · reranker baked into the image
                              │ TLS                    │ OpenTelemetry (batched)
        psql-groundwork-<sfx> (B1ms, PG16: rag, orchestrator; vector)   appi-groundwork → log-groundwork
        Each app reads its own secrets from the shared Key Vault with its own managed identity.

rg-groundwork-shared (bootstrap; never destroyed): state storage (Entra-only), kv-groundwork-<sfx>,
  id-groundwork-github (OIDC: the `azure` environment deploys, `main` runs the guard), budget alert
GitHub: push to main → images.yml → ghcr.io/alirezaabedinii/groundwork-{rag,mcp,agent} (public)
        deploy.yml (manual, environment azure) → PATCH each app's containers to a tag
        stack-guard.yml (hourly) → deletes rg-groundwork once it's older than STACK_MAX_HOURS
```

| App | Size | Ingress | Settings that differ from local |
|---|---|---|---|
| `groundwork-rag` | 1 vCPU / 2 GiB | external, port 8000, your IP only | `DEFAULT_MODE=hybrid`, `HF_HUB_OFFLINE=1`, `COLLECTION=toolchain_docs` |
| `groundwork-mcp` | 0.25 vCPU / 0.5 GiB | internal only | `RAG_API_URL=http://groundwork-rag` |
| `groundwork-agent` | 0.5 vCPU / 1 GiB | external, port 8080, your IP only | `RUN_MODE=inline`, `WORKING_MEMORY_BACKEND=postgres`, `CODE_EXEC_BACKEND=disabled`, scale-in cooldown 600 s |

Why it's shaped this way: [ADR 0009](../docs/decisions/0009-inline-runs-in-the-cloud.md)
(inline runs, no Redis, no code execution), [ADR 0010](../docs/decisions/0010-azure-topology-and-access.md)
(apps, database, access), [ADR 0011](../docs/decisions/0011-terraform-state-identity-and-secrets.md)
(state, identities, secrets), [ADR 0012](../docs/decisions/0012-traces-to-application-insights.md)
(telemetry) and [ADR 0013](../docs/decisions/0013-a-hard-time-limit-on-the-stack.md) (the guard).

## What it costs

| State | Monthly cost |
|---|---|
| Down | about $0.01: the shared group's state storage and Key Vault operations |
| Up, within the free amounts | $0. Container Apps' free grant covers about 28 hours a month with all three apps active; Postgres B1ms is free for 750 hours a month in the account's first 12 months; logs are capped below the free 5 GB |
| Up, past the free amounts (canadacentral list prices) | Container Apps about $0.26/h with all three active and nothing while scaled to zero; Postgres about $0.024/h; requests $0.40 per million. Always on, Postgres alone would be about $17 a month, which is why `make down` is the rule |

Measured on a two-hour session (2026-11-08) with the retrieval check and one real agent
task: 36, 33 and 60 replica-minutes for rag, mcp and agent, about 2.5 % of the monthly
free grant; Postgres on its free meter; $0.06 of model calls. Model calls bill to the
OpenAI and Anthropic accounts, not to Azure: set spend limits in their consoles.

## Prerequisites

- An Azure subscription where you are Owner (Terraform creates role assignments).
- The Azure CLI, Terraform 1.16, `jq` and `gh` (logged in to this repository).
- For seeding, the RAG compose stack's database with the `toolchain_docs` collection
  on `localhost:5433` (`SEED_SOURCE_URL` changes it; see
  [the RAG README](../services/rag/README.md)).

## One-time setup

1. Sign in: `az login` (on a machine without a browser, see Troubleshooting).
2. Create the shared resources:

   ```bash
   make bootstrap ALERT_EMAIL=you@example.com
   ```

   It applies [`bootstrap/`](bootstrap) on local state, then moves that state into the
   storage account it just created, and writes the gitignored `terraform/backend.hcl`
   and `terraform/bootstrap.auto.tfvars`. It creates `rg-groundwork-shared`: the state
   account (Entra-only, no shared keys, versioned), the Key Vault, the GitHub deploy
   identity with its two federated credentials, and a $5 monthly budget that emails at
   20, 60 and 100 % and on the forecast.
3. Put the provider keys in Key Vault, without echoing them:

   ```bash
   KV=$(terraform -chdir=infra/bootstrap output -raw key_vault_name)
   read -rs KEY && printf %s "$KEY" | az keyvault secret set --vault-name "$KV" -n openai-api-key --file /dev/stdin --encoding utf-8 -o none; unset KEY
   read -rs KEY && printf %s "$KEY" | az keyvault secret set --vault-name "$KV" -n anthropic-api-key --file /dev/stdin --encoding utf-8 -o none; unset KEY
   ```

   The same commands replace a key later; apps read the new version when a new revision
   starts (a deploy, or the next `make up`).
4. On GitHub: create the environment `azure` with yourself as required reviewer and
   deployments from `main` only, then store the three ids as repository secrets (secrets
   rather than variables only to keep them out of the public logs; none is a credential):

   ```bash
   terraform -chdir=infra/bootstrap output -raw github_deploy_client_id | gh secret set AZURE_CLIENT_ID
   terraform -chdir=infra/bootstrap output -raw tenant_id | gh secret set AZURE_TENANT_ID
   terraform -chdir=infra/bootstrap output -raw subscription_id | gh secret set AZURE_SUBSCRIPTION_ID
   ```

   Optionally set the repository variable `STACK_MAX_HOURS` (8 if unset).
5. Make the three ghcr packages public (package settings → Change visibility), so
   Azure pulls them without registry credentials.

## Running a session

```bash
make plan        # what make up would change, with your current IP; free
make up          # apply (asks), copy the collection, wake the apps, print the URLs
make urls        # the URLs again
make ui-cloud    # the three Streamlit UIs on this machine against the cloud APIs
make down        # destroy (asks); fails unless rg-groundwork is gone
```

`make up` refuses to start unless `stack-guard.yml` is active on GitHub. It allow-lists
the IPv4 address `api.ipify.org` reports (`OPERATOR_IP=a.b.c.d` overrides it), and
`TF_ARGS` passes variables through, for example `TF_ARGS='-var mock_llm=true'` for a
keyless stack. `make seed-cloud` copies `toolchain_docs`, vectors included, with a
checksum over every column, so the cloud serves the exact vectors the published numbers
came from; it's idempotent. `make warm` waits for both public apps and sends one hybrid
search, which loads the reranker and builds the BM25 index.

Measured: `make up` 599 s from nothing, seed and warm-up included (Postgres takes 4
minutes of it, the environment 2). A scaled-to-zero app answers its first request in about 35 s; the first hybrid
search takes 11 s, later ones about 2 s. `make down` takes about 25 minutes, almost all
of it the environment's deletion. The variables, with their defaults and validation,
are in [`terraform/variables.tf`](terraform/variables.tf); `rag_min_replicas = 1` keeps
the RAG warm for a live demo (it draws on the free grant while idle, so set it back).

## Deploying an image update

`images.yml` publishes `sha-<7>` and `main` tags for every push to `main` that touches a
service. To move the running apps to a published tag:

```bash
gh workflow run deploy.yml -f image_tag=sha-abc1234   # then approve it in the azure environment
```

The job signs in over OIDC (no stored credential), stops with "the stack is down: run
make up first" if it is, checks all three images exist, then PATCHes each app's
containers, the agent's init container included, so migrations run from the new image
too. It waits until each new revision runs the tag, is Running (or RunningAtMaxScale)
and Healthy, and prints the URLs: about 45 s for all three. An app already on the tag is
left as it is, since a PATCH that changes nothing makes no new revision. Terraform
ignores image changes, so the next `make up` keeps the deployed tag.

## The stack guard

[`stack-guard.yml`](../.github/workflows/stack-guard.yml) runs hourly from `main` and
deletes `rg-groundwork` once its oldest resource is older than `STACK_MAX_HOURS`. Azure
has no spending cap outside the Free Trial (budgets only send email, about a day late),
so a time limit is the guard that stops spending. Its identity's role lives on the group
and goes with it, so while the stack is down the check is refused and reports "down".

- **A failed run is the alert.** GitHub emails it, whether the guard deleted the stack
  ("past its N h limit, so rg-groundwork is being deleted") or couldn't check it.
- **After it fires**, run `make down`: Terraform's state still lists the stack, and the
  database secrets and the identities' grants in the shared Key Vault outlive the group.
- **To end a stack now:** `gh workflow run stack-guard.yml -f max_hours=0`. Measured: the
  group was gone 25 minutes after the delete call, and `make down` cleaned up the rest in
  101 s.
- **Limits.** GitHub starts schedules late or skips them: the first scheduled run came
  about 10 hours after the workflow reached `main`, later ones 4.5 to 7 hours apart. So
  the real bound is the limit plus GitHub's delay, about 15 hours at worst so far.
  Schedules only run from the default branch, and GitHub turns them off after 60 days
  without repository activity (`make up` then refuses).

## Troubleshooting

- **Sign-in on a machine without a browser.** A new tenant's security defaults block
  device-code sign-in after its first day. Run `AZURE_CORE_LOGIN_EXPERIENCE_V2=off
  BROWSER=echo az login -o none`, open a tunnel to the port in the printed URL from
  your laptop (`ssh -N -L PORT:127.0.0.1:PORT you@host`), then open the URL there.
- **A request answers 403 "RBAC: access denied".** Your IP changed; run `make up`
  again (about a minute: only the allow-lists and the database's firewall rule change).
- **A resource provider isn't registered.** Terraform registers the ones it needs
  (`make plan` does it the first time; free). Azure lists one as `microsoft.insights`,
  and the provider matches names exactly, so it's declared in lower case.
- **The first apply can't read a secret.** Key Vault role assignments take a few
  minutes to propagate; run `make up` again.
- **The first call after an idle spell times out.** The agent's MCP client and the
  eval harness wait 30 s, and a cold RAG behind the MCP server can take longer. Run
  `make warm` first, or set `rag_min_replicas = 1`.
- **Free amounts.** The Free Trial's credit lasts 30 days; after it, the subscription
  must be upgraded to keep running, and pay-as-you-go has no spending limit, so the
  budget email and the guard are what's left. The Postgres offer lasts 12 months from
  the account's creation. A free subscription can't always create B1ms in every region:
  `TF_ARGS='-var location=eastus2'`.
- **App Insights.** Azure adds an unmanaged "Application Insights Smart Detection"
  action group, which goes with the resource group. To query traces without the CLI
  extension: `az rest --resource https://api.applicationinsights.io` against
  `/v1/apps/<AppId>/query`, or the workspace's `AppRequests` and `AppDependencies`
  tables through `https://api.loganalytics.io`.
- **After a guard deletion,** run `make down` before the next `make up`; otherwise
  Terraform still expects the old server.

## Layout

| Path | Contents |
|---|---|
| [`bootstrap/`](bootstrap) | The shared, persistent resources; applied once by `make bootstrap` |
| [`terraform/`](terraform) | The root module behind `make up` and `make down`, with the `observability`, `database` and `apps` modules |
| [`scripts/`](scripts) | `deploy-images.sh` and `stack-guard.sh` (with `arm.sh`, the token they share), their tests (`make test-infra`), `warm.sh`, `ui-cloud.sh` |
