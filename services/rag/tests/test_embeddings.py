"""The OpenAI embedding client batches its requests and adds up their cost."""

from __future__ import annotations

from types import SimpleNamespace

from rag.indexing.embeddings import OpenAIEmbeddingClient


class FakeEmbeddings:
    """Records each request's inputs; embeds a text as [its length]."""

    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    def create(self, model: str, input: list[str]):  # noqa: A002 - the SDK's name
        self.requests.append(input)
        data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=10 * len(input)))


def client(batch_size: int) -> tuple[OpenAIEmbeddingClient, FakeEmbeddings]:
    embedder = OpenAIEmbeddingClient("text-embedding-3-small", "test", price_per_million=0.02)
    embedder.batch_size = batch_size
    fake = FakeEmbeddings()
    embedder._client = SimpleNamespace(embeddings=fake)
    return embedder, fake


def test_texts_go_out_in_batches_and_come_back_in_order() -> None:
    embedder, fake = client(batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    assert embedder.embed_texts(texts) == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert fake.requests == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    # 10 tokens per text, $0.02 per million
    assert embedder.total_tokens == 50
    assert abs(embedder.total_cost_usd - 50 * 0.02 / 1_000_000) < 1e-15


def test_the_default_batch_stays_under_the_api_limits() -> None:
    embedder, fake = client(batch_size=OpenAIEmbeddingClient.batch_size)
    embedder.embed_texts(["x"] * 600)
    assert [len(r) for r in fake.requests] == [256, 256, 88]
    assert max(len(r) for r in fake.requests) <= 2048


def test_nothing_to_embed_makes_no_request() -> None:
    embedder, fake = client(batch_size=2)
    assert embedder.embed_texts([]) == []
    assert fake.requests == []
