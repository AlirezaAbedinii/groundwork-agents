#!/usr/bin/env bash
# `make warm`: wake the RAG and the agent from zero replicas, then send the RAG one
# hybrid search, which loads the reranker and builds the BM25 index on its replica.
#
# A cold start pulls the image and starts the container, so /health is polled with
# growing waits, up to 10 minutes per app. The search embeds its query with OpenAI:
# a fraction of a cent. Needs RAG_URL and AGENT_URL (the Makefile reads them from
# Terraform's outputs).
set -euo pipefail

: "${RAG_URL:?is empty: is the stack up (make up)?}"
: "${AGENT_URL:?is empty: is the stack up (make up)?}"

wait_healthy() {
  local name=$1 url=$2 start=$SECONDS delay=5 status
  while true; do
    status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 "$url/health" || true)
    case $status in
      200)
        echo "$name: healthy after $((SECONDS - start)) s"
        return 0
        ;;
      403)
        echo "$name: 403 from the allow-list; if your IP changed, run make up again" >&2
        return 1
        ;;
    esac
    if ((SECONDS - start + delay > 600)); then
      echo "$name: no healthy answer from $url/health within 10 minutes (last: ${status:-none})" >&2
      return 1
    fi
    echo "$name: not up yet (${status:-no answer}); retrying in $delay s"
    sleep "$delay"
    delay=$((delay * 2 > 60 ? 60 : delay * 2))
  done
}

# Both at once: each app scales out on its own.
wait_healthy rag "$RAG_URL" &
rag=$!
wait_healthy agent "$AGENT_URL" &
agent=$!
wait "$rag"
wait "$agent"

echo "rag: one hybrid search (loads the reranker, builds the BM25 index)"
start=$SECONDS
curl -fsS --max-time 600 -X POST "$RAG_URL/v1/search" -H 'content-type: application/json' \
  -d '{"query": "UV_CACHE_DIR", "mode": "hybrid", "top_k": 3}' |
  python3 -c 'import json, sys; hits = json.load(sys.stdin)["hits"]; print("rag: top hit", hits[0]["source_file"] if hits else "(none)")'
echo "rag: search answered after $((SECONDS - start)) s"
