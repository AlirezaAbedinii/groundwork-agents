# No variable is sensitive: no secret is an input. The provider keys stay in Key
# Vault, and the database password is generated inside the database module.

variable "location" {
  type        = string
  description = "Azure region for the stack; eastus2 if a subscription can't create the B1ms server in canadacentral."
  default     = "canadacentral"
}

# --- Access -------------------------------------------------------------------

variable "operator_cidrs" {
  type        = set(string)
  description = "IPv4 ranges allowed through the RAG and agent ingress and the Postgres firewall; make up passes your current IP as a /32."

  validation {
    condition     = length(var.operator_cidrs) > 0
    error_message = "operator_cidrs needs at least one range, or nothing can reach the APIs."
  }

  validation {
    # cidrnetmask() takes IPv4 CIDRs only: a bare address or an IPv6 range fails it.
    condition     = alltrue([for cidr in var.operator_cidrs : can(cidrnetmask(cidr)) && !endswith(cidr, "/0")])
    error_message = "Each operator_cidrs entry must be an IPv4 CIDR such as 203.0.113.7/32, and never a /0 such as 0.0.0.0/0."
  }
}

variable "internal_cidrs" {
  type        = set(string)
  description = "Extra ranges in the RAG's allow-list only, for the MCP server's calls from inside the Container Apps environment if that list filters them."
  default     = []

  validation {
    condition     = alltrue([for cidr in var.internal_cidrs : can(cidrnetmask(cidr)) && !endswith(cidr, "/0")])
    error_message = "Each internal_cidrs entry must be an IPv4 CIDR, and never a /0."
  }
}

# --- Images -------------------------------------------------------------------

variable "image_registry" {
  type        = string
  description = "Registry and namespace of the groundwork-{rag,mcp,agent} images."
  default     = "ghcr.io/alirezaabedinii"

  validation {
    # Image references are lowercase, though the GitHub account name isn't.
    condition     = var.image_registry == lower(var.image_registry) && !endswith(var.image_registry, "/")
    error_message = "image_registry must be lowercase, without a trailing slash."
  }
}

variable "image_tag" {
  type        = string
  description = "Tag the apps start with when they're created. Deploys from CI change it afterwards, and Terraform leaves that alone (ignore_changes)."
  default     = "main"

  validation {
    condition     = can(regex("^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$", var.image_tag))
    error_message = "image_tag must be a valid image tag, such as main or sha-1a2b3c4."
  }
}

# --- Behavior and scale ---------------------------------------------------------

variable "mock_llm" {
  type        = bool
  description = "The agent replays recorded fixtures instead of calling the model providers: a deploy check that spends nothing."
  default     = false
}

variable "rag_min_replicas" {
  type        = number
  description = "1 keeps a RAG replica warm for a live demo; an idle replica still draws on the free grant."
  default     = 0

  validation {
    condition     = contains([0, 1], var.rag_min_replicas)
    error_message = "rag_min_replicas must be 0 or 1."
  }
}

variable "agent_min_replicas" {
  type        = number
  description = "1 keeps an agent replica warm for a live demo; an idle replica still draws on the free grant."
  default     = 0

  validation {
    condition     = contains([0, 1], var.agent_min_replicas)
    error_message = "agent_min_replicas must be 0 or 1."
  }
}

variable "agent_cooldown_seconds" {
  type        = number
  description = "Seconds without requests before the agent scales in. Runs continue after their 202 response, so this must outlast a run segment."
  default     = 600

  validation {
    # 300 s is the platform's default; going below it only makes losing a run likelier.
    condition     = var.agent_cooldown_seconds == floor(var.agent_cooldown_seconds) && var.agent_cooldown_seconds >= 300
    error_message = "agent_cooldown_seconds must be a whole number of at least 300."
  }
}

# --- From infra/bootstrap (make bootstrap writes them to bootstrap.auto.tfvars) ---

variable "shared_resource_group" {
  type        = string
  description = "The persistent resource group that holds the Key Vault and this module's state."
}

variable "key_vault_name" {
  type        = string
  description = "The shared Key Vault: it holds the provider keys, and this module writes the database URLs to it."
}

variable "github_deploy_principal_id" {
  type        = string
  description = "Object id of the identity GitHub Actions deploys as; it gets Contributor on the stack's resource group."

  validation {
    condition     = can(regex("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", var.github_deploy_principal_id))
    error_message = "github_deploy_principal_id must be an object id (a lowercase GUID)."
  }
}

# --- Cost guards and metadata -------------------------------------------------

variable "log_daily_cap_gb" {
  type        = number
  description = "Daily ingestion cap on the Log Analytics workspace, in GB."
  default     = 0.15

  validation {
    # The workspace reads -1 as "no cap".
    condition     = var.log_daily_cap_gb > 0
    error_message = "log_daily_cap_gb must be positive."
  }
}

variable "tags" {
  type        = map(string)
  description = "Extra tags on every resource."
  default     = {}
}
