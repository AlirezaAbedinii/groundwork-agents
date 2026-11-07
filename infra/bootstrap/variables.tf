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

# Repositories created after 2026-07-15 get the owner's and the repository's ids in
# their tokens' subject, so a later owner of the same name doesn't match. GitHub reports
# the prefix: gh api 'repos/{owner}/{repo}/actions/oidc/customization/sub' --jq .sub_claim_prefix
variable "github_subject_prefix" {
  type        = string
  description = "How GitHub's OIDC tokens name the repository whose workflows sign in as the deploy identity."
  default     = "repo:AlirezaAbedinii@69193677/groundwork-agents@1379017005"

  validation {
    condition     = can(regex("^repo:[^/:@]+@[0-9]+/[^/:@]+@[0-9]+$", var.github_subject_prefix))
    error_message = "github_subject_prefix must look like repo:owner@id/name@id."
  }
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
