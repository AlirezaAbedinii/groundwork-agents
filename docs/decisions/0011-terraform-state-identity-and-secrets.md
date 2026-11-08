# 0011. Two Terraform configurations, secrets only in Key Vault, and OIDC for CI

Date: 2026-11-08. Status: accepted.

## Context

The stack is created and destroyed many times, while a few things must outlive it: the
Terraform state, the provider API keys and the identity GitHub Actions signs in as.
Terraform state holds every attribute of what it manages, so anything that passes
through a variable or a data source lands there. The repository is public, and so are
its Actions logs. azurerm 5.x offers write-only arguments, which Terraform sends to the
provider but never stores.

## Decision

- **Two configurations.** `infra/bootstrap` is applied once and never destroyed: the
  state storage account (Entra-only access, shared keys off, versioning on), the Key
  Vault, the GitHub deploy identity and a budget alert. Its first apply runs on local
  state, then moves the state into the account it created. `infra/terraform` is the
  stack behind `make up` and `make down`; its state lives in the same account under its
  own key, locked by blob lease. The backend block names only the key; the account comes
  from a gitignored `backend.hcl`.
- **Secrets.** The OpenAI and Anthropic keys are set once with the Azure CLI into the
  shared Key Vault and never pass through Terraform, a tfvars file or GitHub. The apps
  reference them by versionless URI, built as a string rather than read through a data
  source. Each app has its own managed identity with Key Vault Secrets User on exactly
  its own secrets: the retrieval API gets its database URL and the OpenAI key, the agent
  its database URL and both keys, the MCP server nothing. Terraform generates the
  database password as an ephemeral value, writes it once into a Key Vault secret with a
  write-only argument, and builds the server and both database URLs from that secret; so
  no password or URL enters state.
- **CI.** CI never runs Terraform and never sees a provider key. The deploy workflow
  signs in with OIDC as `id-groundwork-github`, which holds Contributor on
  `rg-groundwork` only; the root module grants it at each `make up`, and the grant goes
  with the group. One federated credential is scoped to the `azure` environment (manual
  approval, deployments from `main` only), another to the `main` branch for the hourly
  guard (ADR 0013). The client, tenant and subscription ids are repository secrets, only
  so that GitHub masks them in the public logs; none is a credential.

## Consequences

- State holds names, ids and configuration, never a secret value. Reading it takes an
  Entra role on the storage account; a bad apply can be undone from the account's blob
  versions.
- `make down` removes everything it created, the database secrets in the shared vault
  included, and leaves the shared group, the budget and the ghcr images.
- GitHub gives repositories created after 2026-07-15 OIDC subjects that carry the
  owner's and the repository's numeric ids (`repo:owner@id/name@id:…`), so a later
  repository with the same name can't sign in. The first sign-in failed until the
  credentials used that form; the prefix is a bootstrap variable, read from
  `gh api repos/{owner}/{repo}/actions/oidc/customization/sub`.
- Contributor on the group lets the deploy identity change an app's image or command,
  so deploying is as powerful as reading that app's secrets. The guards are the
  environment's required reviewer and the environment-scoped subject. The `main`
  credential skips the reviewer: any workflow on `main` can sign in as this identity.
- The deploy step runs `az rest` with a token it fetches itself, written to a file only
  the runner's user can read, never put on a command line: plain `az rest` looks the
  subscription up in the CLI's own login first and fails while the identity can see none,
  that is, whenever the stack is down.

## Alternatives

- **One configuration.** `make down` would destroy the state storage holding its own
  state, and the keys with it.
- **A Key Vault per stack.** The keys would have to come in through tfvars, and so into
  state; soft-deleted vault names also block re-creating one under the same name.
- **Keys as app secrets from tfvars, or as GitHub secrets.** Both put the keys where
  this design keeps them out: state, or CI.
- **The password as an ordinary resource attribute.** It would sit in state, readable by
  anyone with the state role. Feeding one ephemeral password straight into the server
  and both URLs was rejected too: after a half-failed apply, the rerun draws a new
  password for whatever wasn't created yet.
- **Terraform apply from CI.** The identity would need rights to create role
  assignments and access to the provider keys, and a push could create paid
  infrastructure while the stack is meant to be down.
- **A client secret for the deploy identity.** A stored credential that expires and can
  leak; OIDC tokens last minutes and are bound to the repository and environment.
