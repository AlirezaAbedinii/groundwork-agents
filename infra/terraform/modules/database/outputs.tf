output "postgres_fqdn" {
  description = "The server's host name."
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "key_vault_secrets" {
  description = "The database URLs' secrets, rag-database-url and agent-database-url: each one's versionless URI and resource id. Available only once the databases exist, Azure's addresses may connect and the vector extension is allowed, so an app that reads one starts after that."
  value = {
    for secret in azurerm_key_vault_secret.database_url : secret.name => {
      uri         = secret.versionless_id
      resource_id = secret.resource_versionless_id
    }
  }

  depends_on = [
    azurerm_postgresql_flexible_server_configuration.extensions,
    azurerm_postgresql_flexible_server_database.main,
    azurerm_postgresql_flexible_server_firewall_rule.azure_services,
  ]
}
