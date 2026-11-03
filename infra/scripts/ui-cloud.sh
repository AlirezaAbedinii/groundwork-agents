#!/usr/bin/env bash
# `make ui-cloud`: the three Streamlit UIs on this machine, against the cloud APIs:
# the RAG's UI, the agent's review queue and its trace explorer. Each runs from its
# project's venv, which needs that project's `ui` extra. Ctrl-C stops all three,
# and so does any one of them exiting.
#
# Needs RAG_URL and AGENT_URL (the Makefile reads them from Terraform's outputs);
# the ports default to 8521, 8522 and 8523, clear of a local stack's.
set -euo pipefail

: "${RAG_URL:?is empty: is the stack up (make up)?}"
: "${AGENT_URL:?is empty: is the stack up (make up)?}"
RAG_UI_PORT=${RAG_UI_PORT:-8521}
REVIEW_UI_PORT=${REVIEW_UI_PORT:-8522}
TRACE_UI_PORT=${TRACE_UI_PORT:-8523}

cd "$(dirname "$0")/../.."

for project in services/rag services/agent; do
  if ! "$project/.venv/bin/python" -c 'import streamlit' 2>/dev/null; then
    echo "streamlit isn't in $project/.venv; install the ui extra:" >&2
    echo "  (cd $project && .venv/bin/python -m pip install -e '.[ui]')" >&2
    exit 1
  fi
done

pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

run() { # project, env assignment, script, port
  (cd "$1" && env "$2" .venv/bin/python -m streamlit run "$3" \
    --server.port "$4" --server.headless true) &
  pids+=($!)
}

run services/rag "RAG_API_URL=$RAG_URL" ui/app.py "$RAG_UI_PORT"
run services/agent "ORCHESTRATOR_API_URL=$AGENT_URL" ui/review_app.py "$REVIEW_UI_PORT"
run services/agent "ORCHESTRATOR_API_URL=$AGENT_URL" ui/trace_explorer.py "$TRACE_UI_PORT"

echo "RAG UI      http://localhost:$RAG_UI_PORT    -> $RAG_URL"
echo "Review UI   http://localhost:$REVIEW_UI_PORT    -> $AGENT_URL"
echo "Traces UI   http://localhost:$TRACE_UI_PORT    -> $AGENT_URL"

# Returns when the first one stops; the EXIT trap stops the rest.
wait -n
