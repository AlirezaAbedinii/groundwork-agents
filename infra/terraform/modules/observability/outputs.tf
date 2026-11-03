output "log_analytics_workspace_id" {
  description = "The workspace the Container Apps environment sends its logs to."
  value       = null
}

output "app_insights_name" {
  description = "The Application Insights resource."
  value       = null
}

output "app_insights_connection_string" {
  description = "Where the services export traces; the apps hold it as a plain app secret."
  value       = null
  sensitive   = true
}
