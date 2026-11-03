output "rag_url" {
  description = "https:// URL of the RAG's ingress."
  value       = null
}

output "agent_url" {
  description = "https:// URL of the agent's ingress."
  value       = null
}

output "rag_app_name" {
  description = "The RAG's Container App."
  value       = null
}

output "mcp_app_name" {
  description = "The MCP server's Container App, also its host name inside the environment."
  value       = null
}

output "agent_app_name" {
  description = "The agent's Container App."
  value       = null
}
