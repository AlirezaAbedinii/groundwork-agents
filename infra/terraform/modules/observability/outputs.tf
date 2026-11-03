output "log_analytics_workspace_id" {
  description = "The workspace the Container Apps environment sends its logs to."
  value       = azurerm_log_analytics_workspace.main.id
}

output "app_insights_name" {
  description = "The Application Insights resource."
  value       = azurerm_application_insights.main.name
}

output "app_insights_connection_string" {
  description = "Where the services export traces; the apps hold it as a plain app secret."
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}
