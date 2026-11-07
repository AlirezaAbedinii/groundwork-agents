#!/usr/bin/env bash
# deploy.yml's job: point the three Container Apps at one image tag, then wait until
# each app's newest revision runs that image and is healthy.
#
# Each app's containers go back whole with only their images changed, through the
# same PATCH `az containerapp update` sends. That command's --image skips init
# containers, and the agent's migrations run in one.
#
# Usage: deploy-images.sh <image tag>. Needs AZURE_SUBSCRIPTION_ID and a signed-in az.
set -euo pipefail

tag=${1:?usage: deploy-images.sh <image tag>}
: "${AZURE_SUBSCRIPTION_ID:?}"
WAIT_SECONDS=${WAIT_SECONDS:-600}
POLL_SECONDS=${POLL_SECONDS:-15}

registry=ghcr.io/alirezaabedinii
apps=(rag mcp agent)
api="api-version=2026-01-01"
group_url="https://management.azure.com/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/rg-groundwork"

app_url() { echo "$group_url/providers/Microsoft.App/containerApps/groundwork-$1"; }
image_of() { echo "$registry/groundwork-$1:$tag"; }

# The deploy identity's only role is on the group and goes with it, so a stopped
# stack shows up here as a refusal rather than as a missing group.
errors=$(mktemp)
if ! az rest --method get --url "$group_url?api-version=2023-07-01" -o none 2>"$errors"; then
  if grep -qE 'AuthorizationFailed|ResourceGroupNotFound' "$errors"; then
    echo "::error::the stack is down: run make up first" >&2
  else
    cat "$errors" >&2
    echo "::error::couldn't read the stack's resource group" >&2
  fi
  exit 1
fi

# Every image must exist before any app changes.
for app in "${apps[@]}"; do
  if ! docker manifest inspect "$(image_of "$app")" >/dev/null; then
    echo "::error::$(image_of "$app") doesn't exist; images.yml publishes the sha-<7> tags" >&2
    exit 1
  fi
done

body=$(mktemp)
for app in "${apps[@]}"; do
  az rest --method get --url "$(app_url "$app")?$api" --query properties.template -o json |
    jq --arg image "$(image_of "$app")" '
      (.containers[].image) = $image
      | (.initContainers[]?.image) = $image
      | {properties: {template: ({containers, initContainers} | del(.. | nulls))}}' >"$body"
  az rest --method patch --url "$(app_url "$app")?$api" \
    --headers content-type=application/json --body "@$body" -o none
  echo "$app: pointed at $(image_of "$app")"
done

# All apps in each round: a replica that started early may scale back to zero, and
# with it the revision's Running state, while a slower app is still being waited on.
pending=("${apps[@]}")
deadline=$((SECONDS + WAIT_SECONDS))
while true; do
  still_pending=()
  for app in "${pending[@]}"; do
    revision=$(az rest --method get --url "$(app_url "$app")?$api" \
      --query properties.latestRevisionName -o tsv 2>/dev/null || true)
    state=$(az rest --method get --url "$(app_url "$app")/revisions/$revision?$api" \
      --query '[properties.template.containers[0].image, properties.provisioningState, properties.runningState, properties.healthState]' \
      -o tsv 2>/dev/null || true)
    read -r image provisioning running health <<<"$state" || true

    # Until the new revision is the latest, these fields describe the old one.
    if [[ ${image:-} == "$(image_of "$app")" ]]; then
      if [[ $provisioning == Failed || $running == Failed ]]; then
        echo "::error::groundwork-$app: revision $revision failed ($provisioning, $running)" >&2
        exit 1
      fi
      if [[ $running == Running && $health == Healthy ]]; then
        echo "$app: $revision is running and healthy"
        continue
      fi
    fi
    echo "$app: waiting (${running:-no state yet}, ${health:-no health yet})"
    still_pending+=("$app")
  done

  ((${#still_pending[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    echo "::error::no healthy revision of tag $tag after $WAIT_SECONDS s: ${still_pending[*]}" >&2
    exit 1
  fi
  pending=("${still_pending[@]}")
  sleep "$POLL_SECONDS"
done

for app in rag agent; do
  fqdn=$(az rest --method get --url "$(app_url "$app")?$api" --query properties.configuration.ingress.fqdn -o tsv)
  echo "$app: https://$fqdn"
done
