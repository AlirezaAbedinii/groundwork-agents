# Groundwork

> Grounded multi-agent research assistant — LangGraph · MCP · hybrid RAG · evals · Azure/Terraform

Groundwork is a LangGraph multi-agent orchestrator that answers research
questions by retrieving from a hybrid-search RAG service over MCP, with
human-in-the-loop approvals, OpenTelemetry tracing and a measured evaluation
suite, deployed to Azure with a single `terraform apply`. This repository is
where it comes together: the RAG service and the agent orchestrator live under
`services/` with their full histories, each with its own README, tests and
measured results; the MCP layer, the cross-service evaluation suite and the
infrastructure that connect them land in `mcp/`, `eval/` and `infra/`.

## License

MIT — see [LICENSE](LICENSE).
