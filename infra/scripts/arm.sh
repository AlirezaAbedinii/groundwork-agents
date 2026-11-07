# shellcheck shell=bash
# Sourced by stack-guard.sh and deploy-images.sh: `arm <az rest arguments>` calls Azure
# Resource Manager with a token fetched here instead of by az rest.
#
# az rest looks the URL's subscription up in the CLI's login before it sends anything.
# While the stack is down the deploy identity has no role, so its login lists no
# subscription and az rest stops at "Subscription ... not found" without asking Azure.
# With its own header the request goes out and Azure answers: AuthorizationFailed when
# the stack is down, SubscriptionNotFound when AZURE_SUBSCRIPTION_ID is wrong.
#
# The token reaches a file only this user can read, deleted on exit, and never a
# command line. A second --headers would replace this one, so the content type for
# PATCH bodies is in the file too.
arm_headers=$(umask 077 && mktemp)
trap 'rm -f "$arm_headers"' EXIT

arm() {
  if [[ ! -s $arm_headers ]]; then
    local token
    token=$(az account get-access-token --resource https://management.azure.com/ \
      --query accessToken -o tsv) || return
    printf '{"Authorization": "Bearer %s", "Content-Type": "application/json"}' "$token" >"$arm_headers"
  fi
  az rest --skip-authorization-header --headers "@$arm_headers" "$@"
}
