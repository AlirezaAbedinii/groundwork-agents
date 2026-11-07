# State lives beside infra/bootstrap's, in the account that configuration created,
# under this module's own key; the blob's lease is the lock. The resource group,
# account and container come from backend.hcl (`init -backend-config=backend.hcl`),
# which make bootstrap writes: a backend block can't read variables, and this way
# the code names no account. Reading the state takes an Entra role, never a key.
terraform {
  backend "azurerm" {
    key              = "groundwork.tfstate"
    use_azuread_auth = true
  }
}
