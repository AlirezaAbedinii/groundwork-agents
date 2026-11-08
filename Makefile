# Groundwork root entrypoints. `make lint test` here runs the same commands as
# the CI workflow, for both services, the MCP server and the evaluation suite.
#
# Each project uses its own virtualenv. The agent Makefile picks up
# services/agent/.venv by itself; the rag Makefile expects `python` on PATH, so
# point it at services/rag/.venv when that exists; mcp/ has no Makefile and uses
# mcp/.venv when present, and eval/ likewise (CI has no .venv and uses the
# runner's interpreter).
RAG_PYTHON := $(if $(wildcard services/rag/.venv/bin/),PYTHON=.venv/bin/python,)
AGENT_PYTHON := $(if $(wildcard services/agent/.venv/bin/),.venv/bin/python,python)
MCP_PYTHON := $(if $(wildcard mcp/.venv/bin/),.venv/bin/python,python)
EVAL_PYTHON := $(if $(wildcard eval/.venv/bin/),.venv/bin/python,python)
AGENT_URL ?= http://localhost:8080
ALERT_EMAIL ?=

.DEFAULT_GOAL := help
.PHONY: help lint test lint-rag lint-agent lint-mcp lint-eval test-rag test-agent test-mcp \
	test-eval e2e-agent eval-score eval-agent-smoke bootstrap tf-init plan up seed-cloud warm urls \
	ui-cloud down tf-check test-infra guard-check

help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lint: lint-rag lint-agent lint-mcp lint-eval ## Lint the services, the MCP server and eval with ruff.

test: test-rag test-agent test-mcp test-eval ## Test the services, the MCP server and eval (as CI).

lint-rag: ## Lint the RAG service.
	$(MAKE) -C services/rag $(RAG_PYTHON) lint

lint-agent: ## Lint the agent service.
	cd services/agent && $(AGENT_PYTHON) -m ruff check .

lint-mcp: ## Lint the MCP server.
	cd mcp && $(MCP_PYTHON) -m ruff check .

lint-eval: ## Lint the evaluation suite.
	cd eval && $(EVAL_PYTHON) -m ruff check .

test-rag: ## RAG unit tests (no paid API calls).
	$(MAKE) -C services/rag $(RAG_PYTHON) test

test-agent: ## Agent unit + integration tests (integration skips without postgres).
	$(MAKE) -C services/agent test

test-mcp: ## MCP server tests (the RAG API is faked; no services needed).
	cd mcp && $(MCP_PYTHON) -m pytest -q

test-eval: ## Evaluation suite tests (services faked; no keys, no paid calls).
	cd eval && $(EVAL_PYTHON) -m pytest -q -m "not live"

e2e-agent: ## Agent end-to-end tests (needs the compose services up).
	$(MAKE) -C services/agent e2e

eval-score: ## Re-score the published evaluation from its committed records ($0, no services).
	@cd eval && if [ -f reports/rag-records.jsonl ]; then \
		$(EVAL_PYTHON) -m rag_eval score --records reports/rag-records.jsonl; \
	else echo "no published RAG records yet (eval/reports/rag-records.jsonl)"; fi
	@cd eval && if [ -f reports/agent-records.jsonl ]; then \
		$(EVAL_PYTHON) -m agent_eval score --records reports/agent-records.jsonl; \
	else echo "no published agent records yet (eval/reports/agent-records.jsonl)"; fi

# Needs the agent API at AGENT_URL in inline mode with MOCK_LLM=1, a migrated and seeded
# database, and MCP_RAG_URL pointing at the RAG stack's MCP server (see eval/README.md).
eval-agent-smoke: ## Run the 2 smoke tasks against a local MOCK_LLM agent ($0) into eval/runs/agent-smoke.
	cd eval && rm -rf runs/agent-smoke && \
		$(EVAL_PYTHON) -m agent_eval run --tasks tasks/smoke_tasks.jsonl --label smoke \
			--run-id agent-smoke --agent $(AGENT_URL) --judge fake --max-cost-usd 0 && \
		$(EVAL_PYTHON) -m agent_eval score --run-id agent-smoke

# --- Azure (infra/; see infra/README.md) ---------------------------------------

# The Terraform code names no subscription; it takes the Azure CLI's (az login).
TF = ARM_SUBSCRIPTION_ID=$${ARM_SUBSCRIPTION_ID:-$$(az account show --query id -o tsv)} \
	terraform -chdir=infra/terraform
# Your public IPv4, for the ingress allow-lists and the Postgres firewall. Read when
# a recipe needs it; set it by hand if that fails: make up OPERATOR_IP=203.0.113.7
OPERATOR_IP ?= $(shell curl -fsS --max-time 10 https://api.ipify.org)
# Starts a recipe line that needs the IP, in $$ip; stops early if it couldn't be read.
WITH_IP = ip='$(OPERATOR_IP)'; test -n "$$ip" || \
	{ echo "couldn't read your public IPv4; pass OPERATOR_IP=a.b.c.d" >&2; exit 1; };
OPERATOR_VAR = -var "operator_cidrs=[\"$$ip/32\"]"
# Extra variables for plan, up and down, e.g. TF_ARGS='-var mock_llm=true'.
TF_ARGS ?=
CLOUD_URLS = RAG_URL=$$($(TF) output -raw rag_url) AGENT_URL=$$($(TF) output -raw agent_url)
# seed-cloud copies this collection from the local RAG database (compose, :5433).
SEED_COLLECTION ?= toolchain_docs
SEED_SOURCE_URL ?= postgresql://rag:rag@localhost:5433/rag
# Extra arguments for the copy, e.g. SEED_ARGS=--dry-run.
SEED_ARGS ?=
SEED_PYTHON := $(if $(wildcard services/rag/.venv/bin/),services/rag/.venv/bin/python,python)
# Local ports for the cloud UIs, clear of a local stack's 8501, 8511 and 8512.
RAG_UI_PORT ?= 8521
REVIEW_UI_PORT ?= 8522
TRACE_UI_PORT ?= 8523

bootstrap: ## Once: shared Azure resources (state, Key Vault, deploy identity, budget); asks first. ALERT_EMAIL=… on the first run.
	ALERT_EMAIL='$(ALERT_EMAIL)' infra/bootstrap/bootstrap.sh

tf-init:
	@test -f infra/terraform/backend.hcl || \
		{ echo "infra/terraform/backend.hcl is missing: run make bootstrap first" >&2; exit 1; }
	@$(TF) init -input=false -backend-config=backend.hcl > /dev/null

plan: tf-init ## Plan the stack against your current IP; free, creates nothing (the first run registers resource providers). TF_ARGS adds variables.
	@$(WITH_IP) $(TF) plan $(OPERATOR_VAR) $(TF_ARGS)

# stack-guard.yml is what ends a forgotten stack, so no stack starts without it.
guard-check:
	@state=$$(gh api 'repos/{owner}/{repo}/actions/workflows/stack-guard.yml' --jq .state 2>/dev/null); \
	test "$$state" = active || { echo "stack-guard.yml isn't active on GitHub (state: $${state:-not found}), \
	so nothing would end a forgotten stack; put it on main or enable it first" >&2; exit 1; }

up: guard-check tf-init ## Create or update the stack (asks first), seed it, warm it, print its URLs. TF_ARGS adds variables.
	@$(WITH_IP) $(TF) apply $(OPERATOR_VAR) $(TF_ARGS)
	@$(MAKE) --no-print-directory seed-cloud warm
	@$(TF) output
	@echo "stack-guard deletes rg-groundwork once it's older than STACK_MAX_HOURS (8 unless set on GitHub); make down when you're done."

# The target's URL goes from Key Vault into the copy's environment only: never an
# argument (the process list) and never printed.
seed-cloud: ## Copy toolchain_docs from the local RAG db to Azure, vectors and all ($0, idempotent).
	@set -e; kv=$$(terraform -chdir=infra/bootstrap output -raw key_vault_name); \
	url=$$(az keyvault secret show --vault-name "$$kv" -n rag-database-url --query value -o tsv); \
	SOURCE_DATABASE_URL='$(SEED_SOURCE_URL)' TARGET_DATABASE_URL="$$url" \
		$(SEED_PYTHON) services/rag/scripts/copy_collection.py --collection $(SEED_COLLECTION) $(SEED_ARGS)

warm: tf-init ## Wake the RAG and the agent; one hybrid search builds the BM25 index (a fraction of a cent).
	@$(CLOUD_URLS) infra/scripts/warm.sh

urls: tf-init ## The stack's URLs and names (Terraform's outputs).
	@$(TF) output

ui-cloud: tf-init ## The three Streamlit UIs here against the cloud APIs: RAG :8521, review :8522, traces :8523.
	@$(CLOUD_URLS) RAG_UI_PORT=$(RAG_UI_PORT) REVIEW_UI_PORT=$(REVIEW_UI_PORT) \
		TRACE_UI_PORT=$(TRACE_UI_PORT) infra/scripts/ui-cloud.sh

down: tf-init ## Destroy the stack (asks first); fails unless rg-groundwork is gone. TF_ARGS adds variables.
	@$(WITH_IP) $(TF) destroy $(OPERATOR_VAR) $(TF_ARGS)
	@test "$$(az group exists -n rg-groundwork)" = false || \
		{ echo "rg-groundwork still exists; check it in the portal, or run make down again" >&2; exit 1; }
	@echo "rg-groundwork is gone. Kept: rg-groundwork-shared (state, Key Vault, deploy identity, budget) and the ghcr images."

tf-check: ## Terraform fmt and validate for both configurations, without credentials (as CI runs it).
	terraform fmt -check -recursive infra
	@for dir in infra/bootstrap infra/terraform; do \
		terraform -chdir=$$dir init -backend=false -lockfile=readonly -input=false > /dev/null && \
		terraform -chdir=$$dir validate || exit 1; \
	done

test-infra: ## The stack-guard and deploy scripts against stand-ins for az and docker (as CI runs it). Needs jq.
	infra/scripts/test-scripts.sh
