#!/usr/bin/env bash
# The stack's hard time limit, run hourly by .github/workflows/stack-guard.yml: it
# deletes rg-groundwork once the group's oldest resource is older than MAX_HOURS.
#
# Time is the thing that can be capped. A budget only sends email, about a day late,
# and a pay-as-you-go subscription has no spending limit to set. The apps have one
# replica each, so the stack's cost per hour is bounded and so is a forgotten stack's.
#
# The deploy identity's only role is on this group and goes with it, so while the
# stack is down the read below is refused, and that's the normal case. Any other
# error fails the run, and so does a deletion: GitHub emails a failed run. Terraform's
# state still lists what was deleted, so `make down` comes next.
#
# Needs AZURE_SUBSCRIPTION_ID, MAX_HOURS and a signed-in az.
set -euo pipefail

: "${AZURE_SUBSCRIPTION_ID:?}"
: "${MAX_HOURS:?}"

if [[ ! $MAX_HOURS =~ ^[0-9]+$ ]]; then
  echo "::error::MAX_HOURS must be a whole number of hours, not '$MAX_HOURS'" >&2
  exit 1
fi

group_url="https://management.azure.com/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/rg-groundwork"
errors=$(mktemp)

if ! oldest=$(az rest --method get --url "$group_url/resources?\$expand=createdTime&api-version=2023-07-01" \
  --query 'min(value[].createdTime)' -o tsv 2>"$errors"); then
  if grep -qE 'AuthorizationFailed|ResourceGroupNotFound' "$errors"; then
    echo "The stack is down: nothing to do."
    exit 0
  fi
  cat "$errors" >&2
  echo "::error::couldn't read the stack's age, so the time limit isn't being enforced" >&2
  exit 1
fi

if [[ -z $oldest || $oldest == None ]]; then
  echo "rg-groundwork is empty: nothing to do."
  exit 0
fi

age=$(($(date +%s) - $(date -u -d "$oldest" +%s)))
if ((age < MAX_HOURS * 3600)); then
  echo "The stack is $((age / 60)) minutes old; its limit is $MAX_HOURS h."
  exit 0
fi

az rest --method delete --url "$group_url?api-version=2023-07-01" -o none
echo "::error::The stack is $((age / 60)) minutes old, past its $MAX_HOURS h limit, so rg-groundwork is being deleted. Run make down next: Terraform's state still lists it." >&2
exit 1
