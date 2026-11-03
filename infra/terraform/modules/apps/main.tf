# The Container Apps and the environment they share. The RAG and the agent read
# their secrets from the shared Key Vault, each with an identity of its own that
# may read exactly the secrets its app uses; the MCP server reads none. Terraform
# sets each image when it creates an app; deploys from CI change it afterwards.

locals {
  # The Key Vault secrets each app reads, by the environment variable that carries
  # each one. The apps' secret references and their identities' role assignments
  # all derive from this.
  key_vault_env = {
    rag = {
      DATABASE_URL   = "rag-database-url"
      OPENAI_API_KEY = "openai-api-key"
    }
    agent = {
      DATABASE_URL      = "agent-database-url"
      OPENAI_API_KEY    = "openai-api-key"
      ANTHROPIC_API_KEY = "anthropic-api-key"
    }
  }

  # One grant per (app, secret), such as "rag/openai-api-key".
  secret_grants = merge([
    for app, env in local.key_vault_env : {
      for secret in values(env) : "${app}/${secret}" => { app = app, secret = secret }
    }
  ]...)

  # Who may reach the RAG: you, and any ranges that calls from inside the
  # environment turn out to come from.
  rag_allowed_cidrs = merge(
    { for cidr in var.operator_cidrs : cidr => "operator" },
    { for cidr in var.internal_cidrs : cidr => "internal" },
  )

  agent_env = {
    APP_VERSION            = var.image_tag
    CODE_EXEC_BACKEND      = "disabled" # no sandbox here (D4)
    MCP_RAG_URL            = "http://${azurerm_container_app.mcp.name}/mcp"
    MOCK_LLM               = tostring(var.mock_llm)
    REVIEWER_PROVIDER      = "cross"
    RUN_MODE               = "inline" # no Celery or Redis in Azure
    WORKING_MEMORY_BACKEND = "postgres"
  }

  # The calibrated thresholds stay at config.py's defaults.
  rag_env = {
    APP_ENV            = "azure"
    APP_VERSION        = var.image_tag
    COLLECTION         = "toolchain_docs"
    DEFAULT_MODE       = "hybrid"
    EMBEDDING_MODEL    = "text-embedding-3-small"
    EMBEDDING_PROVIDER = "openai"
    HF_HUB_OFFLINE     = "1"
    LLM_PROVIDER       = "openai"
  }

  # Plain app secret, not a Key Vault one: an ingestion key, not a credential.
  app_insights_secret = "appinsights-connection-string"
}

resource "azurerm_container_app_environment" "main" {
  name                = "cae-groundwork"
  resource_group_name = var.resource_group_name
  location            = var.location

  # 5.x only streams logs unless told where to keep them.
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = var.log_analytics_workspace_id

  # Just the Consumption profile: billed per use, with the free grant. Declared
  # because workload-profile environments are Azure's default now (the older
  # Consumption-only type is legacy), and profiles can't be added later.
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }

  tags = var.tags
}

# --- Identities and their access to secrets -----------------------------------

resource "azurerm_user_assigned_identity" "app" {
  for_each = local.key_vault_env

  name                = "id-groundwork-${each.key}"
  resource_group_name = var.resource_group_name
  location            = var.location
  tags                = var.tags
}

# Scoped to each secret, not the vault: the RAG's identity can't read the
# Anthropic key, and neither identity can read the database password.
resource "azurerm_role_assignment" "secret_reader" {
  for_each = local.secret_grants

  scope                = var.key_vault_secrets[each.value.secret].resource_id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.app[each.value.app].principal_id
  principal_type       = "ServicePrincipal"
}

# --- RAG ----------------------------------------------------------------------

resource "azurerm_container_app" "rag" {
  name                         = "groundwork-rag"
  resource_group_name          = var.resource_group_name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app["rag"].id]
  }

  # The platform reads each one with the app's identity; a versionless URI follows
  # the secret's latest version.
  dynamic "secret" {
    for_each = toset(values(local.key_vault_env.rag))
    content {
      name                = secret.value
      key_vault_secret_id = var.key_vault_secrets[secret.value].uri
      identity            = azurerm_user_assigned_identity.app["rag"].id
    }
  }

  secret {
    name  = local.app_insights_secret
    value = var.app_insights_connection_string
  }

  template {
    min_replicas = var.rag_min_replicas
    max_replicas = 1

    container {
      name   = "rag"
      image  = "${var.image_registry}/groundwork-rag:${var.image_tag}"
      cpu    = 1
      memory = "2Gi"

      dynamic "env" {
        for_each = local.rag_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.key_vault_env.rag
        content {
          name        = env.key
          secret_name = env.value
        }
      }

      env {
        name        = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        secret_name = local.app_insights_secret
      }

      # Startup is imports only (the models load on first use): 2 minutes is ample.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 5
        timeout                 = 3
        failure_count_threshold = 24
      }

      # Lenient: a reranking burst can keep the one CPU busy, and a slow /health
      # shouldn't restart the replica under a request.
      liveness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 30
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }

  ingress {
    external_enabled           = true
    target_port                = 8000
    allow_insecure_connections = false

    # Only these ranges get in; the platform denies everything else.
    dynamic "ip_security_restriction" {
      for_each = local.rag_allowed_cidrs
      content {
        name             = "${ip_security_restriction.value}-${replace(replace(ip_security_restriction.key, ".", "-"), "/", "-")}"
        description      = ip_security_restriction.value == "operator" ? "Your machine" : "Calls from inside the environment"
        action           = "Allow"
        ip_address_range = ip_security_restriction.key
      }
    }

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  tags = var.tags

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }

  # The platform reads the Key Vault secrets as it creates the app, so the grants
  # must exist first.
  depends_on = [azurerm_role_assignment.secret_reader]
}

# --- MCP server ---------------------------------------------------------------

resource "azurerm_container_app" "mcp" {
  name                         = "groundwork-mcp"
  resource_group_name          = var.resource_group_name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "mcp"
      image  = "${var.image_registry}/groundwork-mcp:${var.image_tag}"
      cpu    = 0.25
      memory = "0.5Gi"

      # Through the RAG's public ingress, so its allow-list applies; the first
      # deploy checks this against http://groundwork-rag inside the environment.
      env {
        name  = "RAG_API_URL"
        value = "https://${azurerm_container_app.rag.ingress[0].fqdn}"
      }

      startup_probe {
        transport               = "TCP"
        port                    = 8001
        interval_seconds        = 5
        timeout                 = 3
        failure_count_threshold = 12
      }

      liveness_probe {
        transport               = "TCP"
        port                    = 8001
        interval_seconds        = 30
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }

  # No public address: only apps in the environment reach it (ADR 0004), over
  # plain HTTP at http://groundwork-mcp.
  ingress {
    external_enabled           = false
    target_port                = 8001
    allow_insecure_connections = true

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  tags = var.tags

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}

# --- Agent --------------------------------------------------------------------

resource "azurerm_container_app" "agent" {
  name                         = "groundwork-agent"
  resource_group_name          = var.resource_group_name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app["agent"].id]
  }

  dynamic "secret" {
    for_each = toset(values(local.key_vault_env.agent))
    content {
      name                = secret.value
      key_vault_secret_id = var.key_vault_secrets[secret.value].uri
      identity            = azurerm_user_assigned_identity.app["agent"].id
    }
  }

  secret {
    name  = local.app_insights_secret
    value = var.app_insights_connection_string
  }

  template {
    # One replica at most: runs execute inside the API process (RUN_MODE=inline),
    # and so do the MCP tools it discovers.
    min_replicas = var.agent_min_replicas
    max_replicas = 1
    # A run keeps going after its 202 response, with no request in flight; the
    # replica must not scale in under it.
    cooldown_period_in_seconds = var.agent_cooldown_seconds

    # Migrations and the demo seed, at every replica start; both are idempotent.
    # Sized so the app's total is a valid Consumption pair whether or not init
    # containers count toward it (0.5 + 0.25 vCPU, 1 + 0.5 Gi).
    init_container {
      name    = "migrate"
      image   = "${var.image_registry}/groundwork-agent:${var.image_tag}"
      command = ["sh", "-c", "alembic upgrade head && python -m orchestrator.db.seed_demo_data"]
      cpu     = 0.25
      memory  = "0.5Gi"

      env {
        name        = "DATABASE_URL"
        secret_name = local.key_vault_env.agent.DATABASE_URL
      }
    }

    container {
      name   = "agent"
      image  = "${var.image_registry}/groundwork-agent:${var.image_tag}"
      cpu    = 0.5
      memory = "1Gi"

      dynamic "env" {
        for_each = local.agent_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.key_vault_env.agent
        content {
          name        = env.key
          secret_name = env.value
        }
      }

      env {
        name        = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        secret_name = local.app_insights_secret
      }

      startup_probe {
        transport               = "HTTP"
        port                    = 8080
        path                    = "/health"
        interval_seconds        = 5
        timeout                 = 3
        failure_count_threshold = 24
      }

      liveness_probe {
        transport               = "HTTP"
        port                    = 8080
        path                    = "/health"
        interval_seconds        = 30
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }

  # Your ranges only; nothing else calls the agent.
  ingress {
    external_enabled           = true
    target_port                = 8080
    allow_insecure_connections = false

    dynamic "ip_security_restriction" {
      for_each = var.operator_cidrs
      content {
        name             = "operator-${replace(replace(ip_security_restriction.value, ".", "-"), "/", "-")}"
        description      = "Your machine"
        action           = "Allow"
        ip_address_range = ip_security_restriction.value
      }
    }

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  tags = var.tags

  lifecycle {
    ignore_changes = [
      template[0].container[0].image,
      template[0].init_container[0].image,
    ]
  }

  depends_on = [azurerm_role_assignment.secret_reader]
}
