# State lives in the storage account this configuration creates. The first
# `make bootstrap` therefore applies on local state (an override file swaps this
# backend for "local"), then `terraform init -migrate-state` moves the state here.
# The resource group, account and container come from backend.hcl, which that
# first run writes; reading the state takes an Entra role, never an account key.
terraform {
  backend "azurerm" {
    key              = "bootstrap.tfstate"
    use_azuread_auth = true
  }
}
