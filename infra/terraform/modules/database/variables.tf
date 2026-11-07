variable "resource_group_name" {
  type        = string
  description = "The stack's resource group."
}

variable "location" {
  type        = string
  description = "Azure region."
}

variable "name_suffix" {
  type        = string
  description = "Suffix for the server's name, which is global across Azure."
}

variable "operator_cidrs" {
  type        = set(string)
  description = "IPv4 ranges with a firewall rule each, for the seeding copy from your machine."
}

variable "key_vault_id" {
  type        = string
  description = "The shared Key Vault, which receives the two database URLs as secrets."
}

variable "tags" {
  type        = map(string)
  description = "Tags on every resource."
}
