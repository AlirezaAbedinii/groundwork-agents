variable "resource_group_name" {
  type        = string
  description = "The stack's resource group."
}

variable "location" {
  type        = string
  description = "Azure region."
}

variable "log_analytics_workspace_id" {
  type        = string
  description = "Where the Container Apps environment sends its logs."
}

variable "app_insights_connection_string" {
  type        = string
  description = "Where the RAG and the agent export traces."
  sensitive   = true
}

variable "key_vault_secrets" {
  type = map(object({
    uri         = string
    resource_id = string
  }))
  description = "The Key Vault secrets the apps read, by name (openai-api-key, anthropic-api-key, rag-database-url, agent-database-url): the versionless URI an app references, and the resource id its identity's role assignment is scoped to."
}

variable "image_registry" {
  type        = string
  description = "Registry and namespace of the groundwork-{rag,mcp,agent} images."
}

variable "image_tag" {
  type        = string
  description = "Tag each app starts with; later deploys change it outside Terraform."
}

variable "operator_cidrs" {
  type        = set(string)
  description = "The only ranges the RAG and agent ingress let in."
}

variable "internal_cidrs" {
  type        = set(string)
  description = "Extra ranges in the RAG's allow-list only."
}

variable "mock_llm" {
  type        = bool
  description = "The agent replays recorded fixtures instead of calling the model providers."
}

variable "rag_min_replicas" {
  type        = number
  description = "The RAG's minimum replica count, 0 or 1."
}

variable "agent_min_replicas" {
  type        = number
  description = "The agent's minimum replica count, 0 or 1."
}

variable "agent_cooldown_seconds" {
  type        = number
  description = "Seconds without requests before the agent scales in."
}

variable "tags" {
  type        = map(string)
  description = "Tags on every resource."
}
