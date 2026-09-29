# Groundwork

Grounded multi-agent research assistant: LangGraph, MCP, hybrid RAG, evals, Azure/Terraform

[![CI](https://github.com/AlirezaAbedinii/groundwork-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/AlirezaAbedinii/groundwork-agents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Groundwork is a research assistant that answers questions from a collection of
documents and carries out multi-step tasks on top of those answers. It is built
from two services:

- **Retrieval service** (`services/rag`): indexes technical documents and
  answers questions with citations. Retrieval combines keyword and vector
  search with a reranker, and the service refuses to answer when the documents
  do not support an answer.
- **Agent service** (`services/agent`): breaks a request into tasks, runs
  specialist agents that call tools through the providers' native tool calling,
  pauses for human approval when a step calls for it, and records a trace of
  every step with its cost.

Both services run locally, each with its own tests and measured results (see
their READMEs). The agents reach the retrieval service over MCP: its search and
question answering are tools the research specialist can call, under the same
permissions, rate limits and tracing as its other tools. Still in progress: one
evaluation suite that measures the whole system, and Terraform that deploys it
to Azure.

## Layout

| Path | Contents |
|---|---|
| [`services/rag`](services/rag/README.md) | Retrieval service: ingestion, hybrid search, reranking, API, UI, evaluation harness |
| [`services/agent`](services/agent/README.md) | Agent service: supervisor and specialist agents, tools, approvals, tracing, API, UIs |
| [`mcp/`](mcp/README.md) | MCP server: the retrieval service's search, ask and list_sources as tools |
| [`eval/`](eval/README.md) | Evaluation across both services (planned) |
| [`infra/`](infra/README.md) | Terraform for Azure (planned) |
| [`docs/decisions/`](docs/decisions/README.md) | Architecture decision records |

## Quickstart

Each service has its own Docker Compose stack. Clone once, then start the one
you want.

```bash
git clone https://github.com/AlirezaAbedinii/groundwork-agents.git
cd groundwork-agents
```

Retrieval service (needs an OpenAI API key in `.env`):

```bash
cd services/rag
cp .env.example .env
docker compose up -d --build && docker compose run --rm seed
```

Agent service (runs without API keys when `MOCK_LLM=1` is set in `.env`):

```bash
cd services/agent
cp .env.example .env
docker compose up -d --build && make demo
```

To let the agents search the documents, start the retrieval stack (it serves
MCP on port 8001) and set `MCP_RAG_URL` in `services/agent/.env`; its
`.env.example` shows the values.

The service READMEs cover endpoints, configuration and local development.
`make test` at the repository root runs all three test suites: both services
and the MCP server.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Two services with tests and CI, brought into one repository | done |
| 1 | Native tool calling and structured outputs in the agent loop | done |
| 2 | Retrieval service exposed as an MCP server; the agents consume it as a client | done |
| 3 | Real corpus in pgvector; retrieval, answer and agent evaluation with published metrics | planned |
| 4 | Azure deployment with Terraform: Container Apps, Postgres, Key Vault, Application Insights | planned |
| 5 | Results, screenshots and a recorded demo | planned |

## License

MIT. See [LICENSE](LICENSE).
