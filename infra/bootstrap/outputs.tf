output "resource_group_name" {
  description = "The shared resource group."
  value       = azurerm_resource_group.shared.name
}

output "storage_account_name" {
  description = "Holds the state of both configurations."
  value       = azurerm_storage_account.state.name
}

output "state_container_name" {
  description = "Container for the state blobs."
  value       = azurerm_storage_container.tfstate.name
}

output "key_vault_name" {
  description = "The shared Key Vault."
  value       = azurerm_key_vault.shared.name
}

output "key_vault_id" {
  description = "The shared Key Vault's resource id."
  value       = azurerm_key_vault.shared.id
}

output "github_deploy_client_id" {
  description = "The deploy identity's client id: GitHub's AZURE_CLIENT_ID variable."
  value       = azurerm_user_assigned_identity.github.client_id
}

output "github_deploy_principal_id" {
  description = "The deploy identity's object id, which the root module grants Contributor on rg-groundwork."
  value       = azurerm_user_assigned_identity.github.principal_id
}

output "github_federated_subject" {
  description = "The only workload that may sign in as the deploy identity."
  value       = azurerm_federated_identity_credential.github.subject
}

# Not secrets, but kept out of plan and apply logs; `terraform output -raw` reads them.
output "tenant_id" {
  description = "GitHub's AZURE_TENANT_ID variable."
  value       = data.azurerm_client_config.current.tenant_id
  sensitive   = true
}

output "subscription_id" {
  description = "GitHub's AZURE_SUBSCRIPTION_ID variable."
  value       = data.azurerm_client_config.current.subscription_id
  sensitive   = true
}
