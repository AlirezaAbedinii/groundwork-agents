output "postgres_fqdn" {
  description = "The server's host name."
  value       = null
}

output "key_vault_secrets" {
  description = "The database URLs' secrets, rag-database-url and agent-database-url: each one's versionless URI and resource id. Available only once the server allows the vector extension, so an app that reads one starts after that."
  value       = null
}
