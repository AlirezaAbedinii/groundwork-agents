output "resource_group" {
  description = "The stack's resource group; make down deletes it."
  value       = azurerm_resource_group.main.name
}

output "rag_url" {
  description = "The RAG API, reachable from operator_cidrs."
  value       = module.apps.rag_url
}

output "agent_url" {
  description = "The agent API, reachable from operator_cidrs."
  value       = module.apps.agent_url
}

output "postgres_fqdn" {
  description = "The Postgres server; the apps' URLs for it are Key Vault secrets."
  value       = module.database.postgres_fqdn
}

output "rag_app_name" {
  description = "The RAG's Container App."
  value       = module.apps.rag_app_name
}

output "mcp_app_name" {
  description = "The MCP server's Container App (internal ingress only)."
  value       = module.apps.mcp_app_name
}

output "agent_app_name" {
  description = "The agent's Container App."
  value       = module.apps.agent_app_name
}

output "app_insights_name" {
  description = "Where both services' traces go."
  value       = module.observability.app_insights_name
}
