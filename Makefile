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

.DEFAULT_GOAL := help
.PHONY: help lint test lint-rag lint-agent lint-mcp lint-eval test-rag test-agent test-mcp \
	test-eval e2e-agent eval-score eval-agent-smoke

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
