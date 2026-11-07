#!/usr/bin/env bash
# `make test-infra`: stack-guard.sh and deploy-images.sh against stand-ins for az and
# docker. Nothing here reaches Azure or a registry. Needs jq.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
export WORK
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/bin"

cat >"$WORK/bin/docker" <<'STUB'
#!/usr/bin/env bash
echo "docker $*" >>"$WORK/calls"
[[ -z ${STUB_MISSING_IMAGE:-} || $* != *"$STUB_MISSING_IMAGE"* ]]
STUB

cat >"$WORK/bin/az" <<'STUB'
#!/usr/bin/env bash
if [[ "${1:-} ${2:-}" == "account get-access-token" ]]; then
  echo "az account get-access-token" >>"$WORK/calls"
  echo stub-token
  exit 0
fi
method='' url='' query='' body='' headers='' header_flags=0 own_token=no
args=$*
while (($#)); do
  case $1 in
    --method) method=$2; shift ;;
    --url) url=$2; shift ;;
    --query) query=$2; shift ;;
    --body) body=${2#@}; shift ;;
    --headers) headers=$2; header_flags=$((header_flags + 1)); shift ;;
    --skip-authorization-header) own_token=yes ;;
  esac
  shift
done
echo "az $method $url" >>"$WORK/calls"

# Every call carries the script's own token, from a file: plain az rest fails while
# the stack is down, and a token on the command line would be in the process list.
if [[ $own_token != yes || $header_flags != 1 || $headers != @* ]] ||
  ! grep -q '"Authorization": "Bearer stub-token"' "${headers#@}" ||
  ! grep -q '"Content-Type": "application/json"' "${headers#@}"; then
  echo "az rest without the script's own token: $args" >&2
  exit 3
fi

if [[ -n ${STUB_ERROR:-} && $method == get ]]; then
  echo "ERROR: $STUB_ERROR" >&2
  exit 1
fi

app=${url##*containerApps/groundwork-}
app=${app%%[/?]*}
case "$method $url" in
  "get "*/rg-groundwork/resources\?*) echo "${STUB_OLDEST:-}" ;;
  "get "*/rg-groundwork\?*) ;;
  "delete "*) ;;
  "patch "*) cp "$body" "$WORK/body-$app.json" ;;
  "get "*/revisions/*)
    printf '%s\t%s\t%s\t%s\n' "ghcr.io/alirezaabedinii/groundwork-$app:${STUB_REVISION_TAG:-sha-new}" \
      "${STUB_PROVISIONING:-Provisioned}" "${STUB_RUNNING:-Running}" "${STUB_HEALTH:-Healthy}"
    ;;
  "get "*)
    case $query in
      properties.template) cat "$WORK/template-$app.json" ;;
      properties.latestRevisionName) echo "groundwork-$app--r1" ;;
      properties.configuration.ingress.fqdn) echo "groundwork-$app.example" ;;
    esac
    ;;
  *) echo "unexpected az call: $args" >&2; exit 2 ;;
esac
STUB
chmod +x "$WORK/bin/az" "$WORK/bin/docker"
mkdir "$WORK/tmp"
export PATH="$WORK/bin:$PATH" AZURE_SUBSCRIPTION_ID=sub-id POLL_SECONDS=0 TMPDIR="$WORK/tmp"

failures=0
pass() { echo "ok    $1"; }
fail() {
  echo "FAIL  $1"
  echo "        ${output//$'\n'/$'\n'        }"
  failures=$((failures + 1))
}
# Runs a script with the variables given before it, keeping its output and status.
run() {
  : >"$WORK/calls"
  output=$(env "$@" 2>&1)
  status=$?
}
called() { grep -q "$1" "$WORK/calls"; }
# check <description> <az verb> <expected exit> <pattern that must be in the output> <verb called: yes/no>
check() {
  local verb=$2 want=$3 pattern=$4 changed=$5 did=no
  called "^az $verb " && did=yes
  if [[ $status == "$want" && $output == *"$pattern"* && $did == "$changed" ]]; then
    pass "$1"
  else
    fail "$1 (exit $status, $verb: $did)"
  fi
}

guard=$here/stack-guard.sh
hours_ago() { date -u -d "$1 hours ago" +%Y-%m-%dT%H:%M:%S.0000000Z; }

run MAX_HOURS=8 STUB_ERROR='Forbidden({"error":{"code":"AuthorizationFailed"}})' "$guard"
check "guard: a refused read means the stack is down" delete 0 "stack is down" no
run MAX_HOURS=8 STUB_ERROR='Not Found({"error":{"code":"ResourceGroupNotFound"}})' "$guard"
check "guard: a missing group means the stack is down" delete 0 "stack is down" no
run MAX_HOURS=8 STUB_ERROR='Service Unavailable' "$guard"
check "guard: any other read error fails the run and deletes nothing" delete 1 "isn't being enforced" no
run MAX_HOURS=8 STUB_ERROR='Not Found({"error":{"code":"SubscriptionNotFound"}})' "$guard"
check "guard: a wrong subscription id fails the run, not reported as down" delete 1 "isn't being enforced" no
run MAX_HOURS=8 STUB_OLDEST=None "$guard"
check "guard: an empty group is left alone" delete 0 "empty" no
run MAX_HOURS=8 STUB_OLDEST="$(hours_ago 7)" "$guard"
check "guard: a stack under its limit is left alone" delete 0 "420 minutes old" no
run MAX_HOURS=8 STUB_OLDEST="$(hours_ago 9)" "$guard"
check "guard: a stack over its limit is deleted, and the run fails" delete 1 "being deleted" yes
run MAX_HOURS=0 STUB_OLDEST="$(hours_ago 0)" "$guard"
check "guard: a limit of 0 deletes the stack now" delete 1 "being deleted" yes
run MAX_HOURS=8 STUB_OLDEST=not-a-date "$guard"
check "guard: an unreadable age fails the run and deletes nothing" delete 1 "invalid date" no
run MAX_HOURS=eight STUB_OLDEST="$(hours_ago 9)" "$guard"
check "guard: a bad limit fails the run and deletes nothing" delete 1 "whole number" no

cat >"$WORK/template-agent.json" <<'JSON'
{
  "containers": [{"name": "agent", "image": "old", "env": [{"name": "A", "value": "1"}], "args": null}],
  "initContainers": [{"name": "migrate", "image": "old", "command": ["sh"]}],
  "revisionSuffix": "keep-out",
  "volumes": null
}
JSON
for app in rag mcp; do
  echo "{\"containers\": [{\"name\": \"$app\", \"image\": \"old\"}], \"initContainers\": null}" >"$WORK/template-$app.json"
done

deploy=$here/deploy-images.sh
image=ghcr.io/alirezaabedinii/groundwork-agent:sha-new

run STUB_ERROR='Forbidden({"error":{"code":"AuthorizationFailed"}})' "$deploy" sha-new
check "deploy: a stopped stack stops it, with the message" patch 1 "the stack is down: run make up first" no
run STUB_ERROR='Service Unavailable' "$deploy" sha-new
check "deploy: another error isn't reported as a stopped stack" patch 1 "couldn't read" no
run STUB_MISSING_IMAGE=groundwork-agent "$deploy" sha-new
check "deploy: a missing image changes no app" patch 1 "doesn't exist" no
run STUB_RUNNING=Failed "$deploy" sha-new
check "deploy: a failed revision stops it at once" patch 1 "failed" yes
run WAIT_SECONDS=0 STUB_REVISION_TAG=sha-old "$deploy" sha-new
check "deploy: only a revision of the new image counts" patch 1 "no healthy revision" yes
run WAIT_SECONDS=0 STUB_HEALTH=Unhealthy "$deploy" sha-new
check "deploy: an unhealthy revision doesn't count" patch 1 "no healthy revision" yes
run "$deploy" sha-new
check "deploy: healthy revisions finish it, with the URLs" patch 0 "https://groundwork-agent.example" yes
if [[ $(grep -c '^az account get-access-token' "$WORK/calls") == 1 ]]; then
  pass "deploy: one token serves the whole run"
else
  output=$(cat "$WORK/calls") fail "deploy: tokens fetched"
fi
if ! grep -rqs stub-token "$WORK/tmp"; then
  pass "deploy: the token's file is gone after the run"
else
  output=$(grep -rls stub-token "$WORK/tmp") fail "deploy: the token's file is left behind"
fi

body=$WORK/body-agent.json
if jq -e --arg image "$image" '.properties.template
    | .containers[0].image == $image and .initContainers[0].image == $image
      and .containers[0].env == [{"name": "A", "value": "1"}]
      and (.containers[0] | has("args") | not)
      and (has("revisionSuffix") or has("volumes") | not)' "$body" >/dev/null; then
  pass "deploy: the agent's patch changes both images and keeps its settings"
else
  output=$(cat "$body") fail "deploy: the agent's patch"
fi
if jq -e '.properties.template | .containers[0].image == "ghcr.io/alirezaabedinii/groundwork-rag:sha-new"
    and (has("initContainers") | not)' "$WORK/body-rag.json" >/dev/null; then
  pass "deploy: an app without an init container gets no such field"
else
  output=$(cat "$WORK/body-rag.json") fail "deploy: the RAG's patch"
fi

if ((failures)); then
  echo "$failures failed"
  exit 1
fi
echo "all passed"
