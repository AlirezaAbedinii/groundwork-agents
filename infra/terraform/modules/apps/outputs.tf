output "rag_url" {
  description = "https:// URL of the RAG's ingress."
  value       = "https://${azurerm_container_app.rag.ingress[0].fqdn}"
}

output "agent_url" {
  description = "https:// URL of the agent's ingress."
  value       = "https://${azurerm_container_app.agent.ingress[0].fqdn}"
}

output "rag_app_name" {
  description = "The RAG's Container App."
  value       = azurerm_container_app.rag.name
}

output "mcp_app_name" {
  description = "The MCP server's Container App, also its host name inside the environment."
  value       = azurerm_container_app.mcp.name
}

output "agent_app_name" {
  description = "The agent's Container App."
  value       = azurerm_container_app.agent.name
}
