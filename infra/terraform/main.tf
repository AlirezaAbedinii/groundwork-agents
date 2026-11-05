# The stack itself: created by `make up` and destroyed by `make down`. All of it
# lives in rg-groundwork except what sits in the shared Key Vault: the database
# password and URLs, and the apps' role assignments on secrets. Postgres, the
# Container Apps and their observability are modules that see each other only
# through outputs passed here.

provider "azurerm" {
  # 5.x registers no resource providers by default; these are the ones used here.
  resource_provider_registrations = "none"
  resource_providers_to_register = [
    "Microsoft.App",
    "Microsoft.DBforPostgreSQL",
    "Microsoft.KeyVault",
    "Microsoft.ManagedIdentity",
    "Microsoft.OperationalInsights",
    # Lowercase on purpose: Azure lists it this way and the provider matches names
    # exactly, so "Microsoft.Insights" would be skipped with only a log warning.
    "microsoft.insights",
  ]

  features {
    application_insights {
      # Otherwise App Insights adds a "Failure Anomalies" alert rule to the group
      # by itself, which Terraform doesn't know about.
      disable_generated_rule = true
    }

    resource_group {
      # The group lasts one session: make down deletes it even if Azure has added
      # something to it.
      prevent_deletion_if_contains_resources = false
    }

    log_analytics_workspace {
      # Skip the workspace's soft-delete, so the next make up can reuse the name.
      permanently_delete_on_destroy = true
    }

    key_vault {
      # The database URLs are recreated at every make up under the same names:
      # make down purges them, and a leftover soft-deleted one is recovered.
      purge_soft_deleted_secrets_on_destroy = true
      recover_soft_deleted_secrets          = true
    }
  }
}

locals {
  tags = merge({ project = "groundwork", lifecycle = "ephemeral" }, var.tags)

  # The provider keys were set once with the Azure CLI. Only their addresses are
  # built here, as strings: a key_vault_secret data source would copy the values
  # into state. Each pairs the URI an app references with the resource id that
  # app's role assignment is scoped to.
  provider_key_secrets = {
    for name in ["openai-api-key", "anthropic-api-key"] : name => {
      uri         = "${data.azurerm_key_vault.shared.vault_uri}secrets/${name}"
      resource_id = "${data.azurerm_key_vault.shared.id}/secrets/${name}"
    }
  }
}

resource "azurerm_resource_group" "main" {
  name     = "rg-groundwork"
  location = var.location
  tags     = local.tags
}

# For the Postgres server's name, which is global across Azure. Each make up draws
# a new one, so a server deleted by the last make down never blocks the name.
resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

# The vault itself, never its secrets.
data "azurerm_key_vault" "shared" {
  name                = var.key_vault_name
  resource_group_name = var.shared_resource_group
}

# --- Modules ------------------------------------------------------------------

module "observability" {
  source = "./modules/observability"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  log_daily_cap_gb    = var.log_daily_cap_gb
  tags                = local.tags
}

module "database" {
  source = "./modules/database"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_suffix         = random_string.suffix.result
  operator_cidrs      = var.operator_cidrs
  key_vault_id        = data.azurerm_key_vault.shared.id
  tags                = local.tags
}

module "apps" {
  source = "./modules/apps"

  resource_group_name            = azurerm_resource_group.main.name
  location                       = azurerm_resource_group.main.location
  log_analytics_workspace_id     = module.observability.log_analytics_workspace_id
  app_insights_connection_string = module.observability.app_insights_connection_string
  key_vault_secrets              = merge(local.provider_key_secrets, module.database.key_vault_secrets)
  image_registry                 = var.image_registry
  image_tag                      = var.image_tag
  operator_cidrs                 = var.operator_cidrs
  internal_cidrs                 = var.internal_cidrs
  mock_llm                       = var.mock_llm
  rag_min_replicas               = var.rag_min_replicas
  agent_min_replicas             = var.agent_min_replicas
  agent_cooldown_seconds         = var.agent_cooldown_seconds
  tags                           = local.tags
}

# --- The deploy identity's access ---------------------------------------------

# GitHub Actions deploys image updates as the identity infra/bootstrap created.
# Its only role is granted here, so it lasts exactly as long as the stack.
resource "azurerm_role_assignment" "github_deploy" {
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Contributor"
  principal_id         = var.github_deploy_principal_id
  principal_type       = "ServicePrincipal"
}
