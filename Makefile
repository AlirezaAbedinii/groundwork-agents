# Groundwork root entrypoints. Every target delegates to a service Makefile, so
# `make lint test` here runs the same commands as the CI workflow.
#
# Each service uses its own virtualenv. The agent Makefile picks up
# services/agent/.venv by itself; the rag Makefile expects `python` on PATH, so
# point it at services/rag/.venv when that exists (CI has no .venv and uses the
# runner's interpreter).
RAG_PYTHON := $(if $(wildcard services/rag/.venv/bin/),PYTHON=.venv/bin/python,)
AGENT_PYTHON := $(if $(wildcard services/agent/.venv/bin/),.venv/bin/python,python)

.DEFAULT_GOAL := help
.PHONY: help lint test lint-rag lint-agent test-rag test-agent e2e-agent

help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lint: lint-rag lint-agent ## Lint both services with ruff.

test: test-rag test-agent ## Test both services (what CI runs).

lint-rag: ## Lint the RAG service.
	$(MAKE) -C services/rag $(RAG_PYTHON) lint

lint-agent: ## Lint the agent service.
	cd services/agent && $(AGENT_PYTHON) -m ruff check .

test-rag: ## RAG unit tests + mocked eval smoke (no paid API calls).
	$(MAKE) -C services/rag $(RAG_PYTHON) test eval-smoke

test-agent: ## Agent unit + integration tests (integration skips without postgres).
	$(MAKE) -C services/agent test

e2e-agent: ## Agent end-to-end tests (needs the compose services up).
	$(MAKE) -C services/agent e2e
