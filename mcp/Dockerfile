# MCP server image: the RAG service's retrieval as MCP tools over Streamable HTTP.
# RAG_API_URL points it at the RAG API (compose sets http://api:8000).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code edits don't rebuild this layer. uv is mounted for
# the install only and doesn't stay in the image.
COPY pyproject.toml ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.17,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache -r pyproject.toml

COPY server.py tool_models.py ./

RUN useradd --create-home --uid 10001 mcp
USER mcp

EXPOSE 8001

# 0.0.0.0 so other containers and the host's port mapping can reach it; the SDK
# applies its localhost-only Host check just when bound to a loopback address.
CMD ["python", "server.py", "--host", "0.0.0.0"]
