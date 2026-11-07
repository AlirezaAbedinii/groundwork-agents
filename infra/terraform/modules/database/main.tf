# Postgres for both services: one Flexible Server with a database each, the vector
# extension allowed, and each app's URL in the shared Key Vault. No password or URL
# enters Terraform's state: they pass only through ephemeral values into
# write-only arguments (`*_wo`), which Terraform sends to Azure and never records.

locals {
  admin_login = "pgadmin"

  # Each app's database, and the URL scheme its client expects.
  databases = {
    rag   = { name = "rag", scheme = "postgresql" }                  # psycopg
    agent = { name = "orchestrator", scheme = "postgresql+psycopg" } # SQLAlchemy
  }
}

# --- The admin password -------------------------------------------------------

# Drawn anew in every plan and apply, and never stored. Alphanumeric, so the URLs
# need no percent-encoding.
ephemeral "random_password" "admin" {
  length  = 32
  special = false
}

# The password's one durable copy, which the server and both URLs are built from.
# Feeding the ephemeral value to all three directly would leave them out of step
# after an apply that failed half-way: the rerun would draw a new password for
# whatever it hadn't created yet.
resource "azurerm_key_vault_secret" "admin_password" {
  name             = "postgres-admin-password"
  key_vault_id     = var.key_vault_id
  value_wo         = ephemeral.random_password.admin.result
  value_wo_version = 1
  tags             = var.tags
}

# Reads that copy back. While the secret's version is unknown (the apply that
# creates it), Terraform opens this only once the secret exists.
ephemeral "azurerm_key_vault_secret" "admin_password" {
  name         = azurerm_key_vault_secret.admin_password.name
  key_vault_id = var.key_vault_id
  version      = azurerm_key_vault_secret.admin_password.version
}

# --- Server -------------------------------------------------------------------

resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-groundwork-${var.name_suffix}"
  resource_group_name = var.resource_group_name
  location            = var.location
  version             = "16"

  # The free offer covers exactly this: B1ms with 32 GB. Growing past 32 GB would
  # be billed, so storage doesn't grow by itself.
  sku_name          = "B_Standard_B1ms"
  storage_mb        = 32768
  auto_grow_enabled = false

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  # Container Apps' outbound addresses aren't fixed, so the server is public behind
  # its firewall rules; TLS is required (the server's default).
  public_network_access_enabled = true

  administrator_login               = local.admin_login
  administrator_password_wo         = ephemeral.azurerm_key_vault_secret.admin_password.value
  administrator_password_wo_version = 1

  tags = var.tags

  lifecycle {
    # Azure picks a zone when none is given, and 5.x reads it back without treating
    # it as computed: every later plan would try to move the server, which fails.
    ignore_changes = [zone]
  }
}

# The RAG creates the vector extension at startup, and so does the agent's
# migration 0007; Azure refuses CREATE EXTENSION for anything not listed here.
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "VECTOR"
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  for_each = local.databases

  name      = each.value.name
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# --- Firewall -----------------------------------------------------------------

# 0.0.0.0 is Azure's "allow Azure services" rule: any Azure address, not only this
# subscription's, so the password is what keeps others out.
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# For the seeding copy from your machine.
resource "azurerm_postgresql_flexible_server_firewall_rule" "operator" {
  for_each = var.operator_cidrs

  name             = "operator-${replace(replace(each.value, ".", "-"), "/", "-")}"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = cidrhost(each.value, 0)
  end_ip_address   = cidrhost(each.value, -1)
}

# --- The apps' URLs -----------------------------------------------------------

resource "azurerm_key_vault_secret" "database_url" {
  for_each = local.databases

  name             = "${each.key}-database-url"
  key_vault_id     = var.key_vault_id
  value_wo         = "${each.value.scheme}://${local.admin_login}:${ephemeral.azurerm_key_vault_secret.admin_password.value}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${each.value.name}?sslmode=require"
  value_wo_version = 1
  tags             = var.tags

  lifecycle {
    # The URL names the server; a new server needs new secrets, and value_wo alone
    # is never resent while value_wo_version stays the same.
    replace_triggered_by = [azurerm_postgresql_flexible_server.main.id]
  }
}
