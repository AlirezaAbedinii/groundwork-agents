#!/usr/bin/env bash
# `make bootstrap`: the shared Azure resources (this directory's configuration).
#
# First run: the state's storage doesn't exist yet, so apply on local state (an
# override file swaps the azurerm backend for "local"), write backend.hcl from the
# outputs, then move the state into the account the apply created and delete the
# local copy. Later runs: plan and apply against that remote state.
#
# Every run ends by writing the root module's backend.hcl and bootstrap.auto.tfvars
# (both gitignored) from the outputs. Needs `az login`; ALERT_EMAIL on the first run.
# terraform apply asks before it changes anything.
set -euo pipefail

cd "$(dirname "$0")"
ROOT_MODULE=../terraform

# The code names no subscription; the provider and the backend take the CLI's.
ARM_SUBSCRIPTION_ID="${ARM_SUBSCRIPTION_ID:-$(az account show --query id -o tsv)}"
export ARM_SUBSCRIPTION_ID

if [[ ! -f terraform.tfvars ]]; then
  : "${ALERT_EMAIL:?the first run needs it: make bootstrap ALERT_EMAIL=you@example.com}"
  printf 'alert_email = "%s"\n' "$ALERT_EMAIL" > terraform.tfvars
fi

write_backend_config() {
  cat > "$1" <<EOF
resource_group_name  = "$(terraform output -raw resource_group_name)"
storage_account_name = "$(terraform output -raw storage_account_name)"
container_name       = "$(terraform output -raw state_container_name)"
EOF
}

if [[ -f backend.hcl ]]; then
  terraform init -input=false -reconfigure -backend-config=backend.hcl
  terraform apply
else
  trap 'rm -f local_backend_override.tf' EXIT
  printf 'terraform {\n  backend "local" {}\n}\n' > local_backend_override.tf
  terraform init -input=false -reconfigure
  terraform apply
  write_backend_config backend.hcl
  rm -f local_backend_override.tf

  # The blob role the apply just granted can take minutes to reach the data plane.
  for attempt in $(seq 1 10); do
    if terraform init -input=false -migrate-state -force-copy -backend-config=backend.hcl; then
      break
    fi
    if [[ $attempt -eq 10 ]]; then
      echo "the state is still local (terraform.tfstate); rerun make bootstrap" >&2
      exit 1
    fi
    echo "the state account isn't writable yet (role propagation); retrying in 30 s" >&2
    sleep 30
  done

  # Delete the local copy only once the remote state holds the account itself.
  terraform state list | grep -qx 'azurerm_storage_account.state'
  rm -f terraform.tfstate terraform.tfstate.backup
fi

mkdir -p "$ROOT_MODULE"
write_backend_config "$ROOT_MODULE/backend.hcl"
cat > "$ROOT_MODULE/bootstrap.auto.tfvars" <<EOF
# Written by make bootstrap from infra/bootstrap's outputs.
shared_resource_group      = "$(terraform output -raw resource_group_name)"
key_vault_name             = "$(terraform output -raw key_vault_name)"
github_deploy_principal_id = "$(terraform output -raw github_deploy_principal_id)"
EOF

echo
echo "Shared resources ready. Key Vault: $(terraform output -raw key_vault_name)"
echo "Wrote $ROOT_MODULE/backend.hcl and $ROOT_MODULE/bootstrap.auto.tfvars."
