"""Long-term memory on Postgres with pgvector.

Per-user isolation, ranking by cosine distance, access bumps, deletion, and
consolidation over the stored embeddings. MOCK_LLM's token-hash embeddings
(256 dimensions) make every similarity below computable by hand: for unit
vectors over distinct tokens, cosine = shared tokens / sqrt(len(a) * len(b)).
"""

import json

import pytest

from orchestrator.config import get_settings
from orchestrator.llm.mock import MockLLMClient
from orchestrator.memory.embeddings import embed_texts
from orchestrator.memory.longterm import LongTermMemory
from orchestrator.memory.management import consolidate


@pytest.fixture()
def longterm():
    return LongTermMemory()


def test_another_users_closer_memory_is_never_returned(longterm):
    query = "Postgres stores vectors with the pgvector extension"
    longterm.add("facts", query, user_id="bob")  # identical text: distance 0, but bob's
    alice_id = longterm.add("facts", "Postgres runs on port 5432", user_id="alice")

    hits = longterm.query("facts", query, user_id="alice", k=3)

    assert [hit.id for hit in hits] == [alice_id]
    assert hits[0].metadata["user_id"] == "alice"


def test_k_is_respected_and_distances_ascend(longterm):
    # Cosine to the query: 1.0, 3/sqrt(12) = 0.866, 2/sqrt(8) = 0.707, 1/2, 0.
    texts = [
        "vector database index tuning",
        "vector database index",
        "vector database",
        "vector",
        "quarterly finance report due friday",
    ]
    ids = [longterm.add("facts", text, user_id="alice") for text in texts]

    hits = longterm.query("facts", "vector database index tuning", user_id="alice", k=3)

    assert [hit.id for hit in hits] == ids[:3]
    assert [hit.distance for hit in hits] == pytest.approx([0.0, 1 - 0.866, 1 - 0.7071], abs=1e-3)


def test_bump_access_raises_importance(longterm):
    memory_id = longterm.add("facts", "Alice tracks pgvector releases", user_id="alice")
    before = longterm.get_all("alice")["facts"][0]

    longterm.bump_access("facts", [memory_id])

    after = longterm.get_all("alice")["facts"][0]
    assert (before["access_count"], after["access_count"]) == (0, 1)
    assert after["importance"] > before["importance"]
    assert after["last_accessed_at"] >= before["last_accessed_at"]


def test_delete_user_counts_per_kind(longterm):
    longterm.add("facts", "Alice fact one", user_id="alice")
    longterm.add("facts", "Alice fact two", user_id="alice")
    longterm.add("preferences", "Alice prefers short memos", user_id="alice")
    longterm.add("facts", "Bob fact", user_id="bob")

    assert longterm.delete_user("alice") == {"episodes": 0, "facts": 2, "preferences": 1}
    assert sum(len(items) for items in longterm.get_all("alice").values()) == 0
    assert len(longterm.get_all("bob")["facts"]) == 1


def test_mock_vectors_and_metadata_round_trip(longterm):
    text = "Memories keep their mock embeddings"
    memory_id = longterm.add("episodes", text, user_id="alice", task_id="t" * 32)

    items = longterm.all_items("episodes", user_id="alice", with_embeddings=True)

    assert (items["ids"], items["documents"]) == ([memory_id], [text])
    (embedding,) = items["embeddings"]
    assert len(embedding) == get_settings().mock_embedding_dim == 256
    assert embedding == pytest.approx(embed_texts([text])[0], abs=1e-6)  # stored as float4
    meta = items["metadatas"][0]
    assert (meta["kind"], meta["user_id"], meta["task_id"]) == ("episodes", "alice", "t" * 32)
    assert meta["access_count"] == 0 and meta["created_at"] == meta["last_accessed_at"]
    assert longterm.all_items("episodes", user_id="alice")["embeddings"] is None


@pytest.mark.anyio
async def test_consolidation_merges_stored_near_duplicates(longterm, tmp_path):
    summary = "CONSOLIDATED: pgvector adds vector similarity search to Postgres."
    (tmp_path / "memory.json").write_text(
        json.dumps({"agent": "memory", "response": {"text": summary}}), encoding="utf-8"
    )
    # 9 shared tokens of 9 and 10: cosine 9 / sqrt(90) = 0.949, above the 0.9 threshold.
    first = longterm.add("facts", "pgvector adds vector similarity search to the Postgres database", user_id="alice")
    second = longterm.add(
        "facts", "pgvector adds vector similarity search to the Postgres database server", user_id="alice"
    )

    report = await consolidate(longterm, MockLLMClient(tmp_path), user_id="alice", events=None)

    assert {deleted["id"] for deleted in report["deleted"]} == {first, second}
    assert [item["text"] for item in longterm.get_all("alice")["facts"]] == [summary]
