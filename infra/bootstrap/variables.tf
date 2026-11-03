variable "location" {
  type        = string
  description = "Azure region for the shared resources."
  default     = "canadacentral"
}

variable "alert_email" {
  type        = string
  description = "Where the budget's alerts go."

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email))
    error_message = "alert_email must be an email address."
  }
}

variable "budget_enabled" {
  type        = bool
  description = "Create the monthly budget. Off only for an offer type that doesn't support budgets."
  default     = true
}

variable "budget_amount" {
  type        = number
  description = "Monthly budget, in the billing account's currency; alerts at 20, 60 and 100 % of it, and at a 100 % forecast."
  default     = 5

  validation {
    condition     = var.budget_amount > 0
    error_message = "budget_amount must be positive."
  }
}

variable "github_repository" {
  type        = string
  description = "owner/name of the repository whose deploy workflow signs in as the deploy identity."
  default     = "AlirezaAbedinii/groundwork-agents"
}

variable "github_environment" {
  type        = string
  description = "GitHub environment the deploy job runs in; the federated credential trusts only that environment."
  default     = "azure"
}

variable "tags" {
  type        = map(string)
  description = "Extra tags on every resource."
  default     = {}
}
