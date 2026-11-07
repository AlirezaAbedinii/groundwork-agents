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
  description = "The deploy identity's client id: GitHub's AZURE_CLIENT_ID secret."
  value       = azurerm_user_assigned_identity.github.client_id
}

output "github_deploy_principal_id" {
  description = "The deploy identity's object id, which the root module grants Contributor on rg-groundwork."
  value       = azurerm_user_assigned_identity.github.principal_id
}

output "github_federated_subject" {
  description = "The deploy workflow's job: signs in as the deploy identity from the reviewed environment."
  value       = azurerm_federated_identity_credential.github.subject
}

output "github_guard_subject" {
  description = "The stack guard's job: signs in as the deploy identity from the default branch."
  value       = azurerm_federated_identity_credential.github_guard.subject
}

# Not secrets, but kept out of plan and apply logs; `terraform output -raw` reads them.
output "tenant_id" {
  description = "GitHub's AZURE_TENANT_ID secret."
  value       = data.azurerm_client_config.current.tenant_id
  sensitive   = true
}

output "subscription_id" {
  description = "GitHub's AZURE_SUBSCRIPTION_ID secret."
  value       = data.azurerm_client_config.current.subscription_id
  sensitive   = true
}
