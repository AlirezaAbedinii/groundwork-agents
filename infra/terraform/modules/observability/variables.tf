variable "resource_group_name" {
  type        = string
  description = "The stack's resource group."
}

variable "location" {
  type        = string
  description = "Azure region."
}

variable "log_daily_cap_gb" {
  type        = number
  description = "Daily ingestion cap on the Log Analytics workspace, in GB."
}

variable "tags" {
  type        = map(string)
  description = "Tags on every resource."
}
