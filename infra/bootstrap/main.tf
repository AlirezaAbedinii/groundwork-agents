# The shared, persistent resources: applied once (make bootstrap), never destroyed
# by `make down`. The state storage for both configurations, the Key Vault that
# holds the provider keys, the identity GitHub Actions deploys with, and a budget.

provider "azurerm" {
  # 5.x registers no resource providers by default; these are the ones used here.
  resource_provider_registrations = "none"
  resource_providers_to_register = [
    "Microsoft.Consumption",
    "Microsoft.KeyVault",
    "Microsoft.ManagedIdentity",
    "Microsoft.Storage",
  ]
  # Shared keys are off on the state account, so any data-plane call uses Entra ID.
  storage_use_azuread = true

  features {
    storage {
      # The account uses neither queue properties nor a static website, so the
      # provider skips its data-plane checks, which would need the blob role this
      # same apply grants.
      data_plane_available = false
    }
  }
}

data "azurerm_client_config" "current" {}

data "azurerm_subscription" "current" {}

locals {
  tags = merge({ project = "groundwork", lifecycle = "persistent" }, var.tags)

  budget_alerts = [
    { threshold = 20, type = "Actual" },
    { threshold = 60, type = "Actual" },
    { threshold = 100, type = "Actual" },
    { threshold = 100, type = "Forecasted" },
  ]
}

# Storage account and Key Vault names are global across Azure, so they get a suffix.
resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

resource "azurerm_resource_group" "shared" {
  name     = "rg-groundwork-shared"
  location = var.location
  tags     = local.tags
}

# --- Terraform state ----------------------------------------------------------

resource "azurerm_storage_account" "state" {
  name                     = "stgwtfstate${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.shared.name
  location                 = azurerm_resource_group.shared.location
  account_kind             = "StorageV2"
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  # Entra ID only: no account keys and no SAS, so reading state takes a role.
  shared_access_key_enabled       = false
  default_to_oauth_authentication = true
  allow_nested_items_to_be_public = false

  blob_properties {
    # Every state write is a new version: the undo for a bad apply.
    versioning_enabled = true

    delete_retention_policy {
      days = 7
    }

    container_delete_retention_policy {
      days = 7
    }
  }

  tags = local.tags
}

resource "azurerm_storage_container" "tfstate" {
  name                  = "tfstate"
  storage_account_id    = azurerm_storage_account.state.id
  container_access_type = "private"
}

# --- Secrets ------------------------------------------------------------------

# Holds the provider keys, which you set with the Azure CLI, so they never pass
# through Terraform; the root module adds each session's database URLs.
resource "azurerm_key_vault" "shared" {
  name                       = "kv-groundwork-${random_string.suffix.result}"
  resource_group_name        = azurerm_resource_group.shared.name
  location                   = azurerm_resource_group.shared.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  rbac_authorization_enabled = true
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = local.tags
}

# --- The operator's access (whoever runs `az login`) --------------------------

resource "azurerm_role_assignment" "operator_state" {
  scope                = azurerm_storage_account.state.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "operator_secrets" {
  scope                = azurerm_key_vault.shared.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

# --- The deploy identity ------------------------------------------------------

# GitHub Actions signs in as this identity over OIDC: no client secret anywhere.
# Its only role, Contributor on rg-groundwork, comes from the root module at each
# `make up`, so it can touch nothing while the stack is down.
resource "azurerm_user_assigned_identity" "github" {
  name                = "id-groundwork-github"
  resource_group_name = azurerm_resource_group.shared.name
  location            = azurerm_resource_group.shared.location
  tags                = local.tags
}

resource "azurerm_federated_identity_credential" "github" {
  name                      = "github-${var.github_environment}"
  user_assigned_identity_id = azurerm_user_assigned_identity.github.id
  issuer                    = "https://token.actions.githubusercontent.com"
  audience                  = ["api://AzureADTokenExchange"]
  # Only jobs that run in this repository's environment get a token, whatever branch.
  subject = "repo:${var.github_repository}:environment:${var.github_environment}"
}

# For stack-guard.yml, which deletes a stack that has outlived its time limit. It must
# run unattended, so it can't use the reviewed environment above; this credential
# trusts jobs on the default branch that name no environment instead.
resource "azurerm_federated_identity_credential" "github_guard" {
  name                      = "github-guard"
  user_assigned_identity_id = azurerm_user_assigned_identity.github.id
  issuer                    = "https://token.actions.githubusercontent.com"
  audience                  = ["api://AzureADTokenExchange"]
  subject                   = "repo:${var.github_repository}:ref:refs/heads/main"
}

# --- Budget -------------------------------------------------------------------

resource "azurerm_consumption_budget_subscription" "monthly" {
  count = var.budget_enabled ? 1 : 0

  name            = "budget-groundwork-monthly"
  subscription_id = data.azurerm_subscription.current.id
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    # The first of the current month: budgets start on a month boundary.
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", plantimestamp())
  }

  dynamic "notification" {
    for_each = local.budget_alerts
    content {
      enabled        = true
      threshold      = notification.value.threshold
      threshold_type = notification.value.type
      operator       = "GreaterThanOrEqualTo"
      contact_emails = [var.alert_email]
    }
  }

  lifecycle {
    # The start date is fixed once the budget exists; plantimestamp() moves every plan.
    ignore_changes = [time_period]
  }
}
