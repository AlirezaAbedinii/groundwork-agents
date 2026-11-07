# Where the stack's logs and traces go: one Log Analytics workspace, which the
# Container Apps environment writes its logs to, and a workspace-based Application
# Insights the services export their OpenTelemetry spans to. Both are capped, so a
# runaway log can't run past the free 5 GB a month.

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-groundwork"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = "PerGB2018"
  # The shortest retention the SKU allows. App Insights' data lives in this
  # workspace's tables, so it's kept 30 days too (its own retention_in_days, 90 by
  # default, is the classic setting).
  retention_in_days = 30
  # The default, 0.15 GB a day, is 4.65 GB in a 31-day month: under the free 5 GB.
  daily_quota_gb = var.log_daily_cap_gb
  tags           = var.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-groundwork"
  resource_group_name = var.resource_group_name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  # The spans land in the workspace and count against its cap too; this one keeps
  # traces from crowding out the environment's logs.
  daily_data_cap_in_gb = 0.1
  # No sampling: a task is a few hundred spans, and every one is worth keeping.
  sampling_percentage = 100
  tags                = var.tags
}
