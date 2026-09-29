"""Smoke-check a running MCP server: python smoke.py [url]  (default http://localhost:8001/mcp)

Prints the negotiated protocol, the tools, what list_sources, search and ask
return. Tool errors are printed, not raised: without a generation key the RAG
service can't answer, so ask reporting its 503 is the expected keyless result.
"""

import sys

import anyio
from mcp import Client

QUESTION = "What does FERRY-429 mean, and how should clients retry?"


async def call(client: Client, name: str, arguments: dict) -> dict | None:
    """The tool's structured result, or None after printing its error."""
    result = await client.call_tool(name, arguments)
    if result.is_error:
        text = " ".join(block.text for block in result.content if hasattr(block, "text"))
        print(f"{name}: tool error: {text}")
        return None
    return result.structured_content


async def main(target) -> None:
    async with Client(target) as client:
        print(f"protocol {client.protocol_version}")
        print(f"tools {sorted(tool.name for tool in (await client.list_tools()).tools)}")

        if listed := await call(client, "list_sources", {}):
            count, chunks = len(listed["sources"]), listed["total_chunks"]
            print(f"list_sources: {count} sources, {chunks} chunks")

        if found := await call(client, "search", {"query": "FERRY-429", "top_k": 3}):
            print(f"search 'FERRY-429' ({found['mode']}):")
            for rank, hit in enumerate(found["hits"], start=1):
                heading = hit["section_heading"]
                print(f"  {rank}. {hit['score']:.3f}  {hit['source_file']}  {heading}")

        if answer := await call(client, "ask", {"question": QUESTION}):
            confidence, cost = answer["confidence"], answer["cost_usd"]
            print(f"ask (confidence {confidence:.2f}, ${cost:.6f}): {answer['answer']}")
            for cited in answer["citations"]:
                heading = cited["section_heading"]
                print(f"  [{cited['index']}] {cited['source_file']}  {heading}  "
                      f"supported={cited['supported']}")


if __name__ == "__main__":
    anyio.run(main, sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/mcp")
