"""Seed script: index the sample corpus (data/raw/ferry_docs) into Postgres (pgvector).

The one command that makes a fresh checkout queryable:

    python scripts/seed.py            # local (needs OPENAI_API_KEY in .env)
    docker compose run --rm seed      # same, inside the compose stack

Idempotent: chunk ids hash their content, so a re-run embeds and stores only
the chunks the collection doesn't hold yet (an unchanged corpus costs nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    corpus = settings.corpus_dir
    if not corpus.is_dir():
        print(f"error: sample corpus not found at {corpus}", file=sys.stderr)
        return 1

    from rag.indexing import index_path  # local import: needs indexing extras

    print(f"Seeding from {corpus} ...")
    summary = index_path(corpus, settings=settings)
    print(
        f"Seeded {summary.chunks_indexed} new chunks from {summary.files} file(s) into "
        f"collection '{settings.collection}' ({summary.chunks_already_stored} already stored; "
        f"skipped {summary.chunks_skipped_duplicates} dupes; "
        f"total now {summary.total_chunks_in_store})."
    )
    print(
        f"embedding_cost_usd={summary.embedding_cost_usd:.6f} timings_ms={summary.timings_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
