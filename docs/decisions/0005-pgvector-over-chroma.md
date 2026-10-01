# 0005. Postgres with pgvector instead of Chroma

Date: 2026-10-01. Status: accepted.

## Context

The repository kept vectors in two stores and a third index on disk. The
retrieval service embedded Chroma in its API process and wrote its BM25 index
to a pickle beside it; the agents kept their long-term memory in a Chroma
server container. The agents already ran Postgres for tasks, traces and
checkpoints, and the Azure deployment planned next has room for exactly one
managed Postgres server on its free tier. The evaluation work also needed more
from the retrieval store than Chroma recorded: which embedding model built a
collection, and a way to read one source's chunks in order.

## Decision

Both services keep their vectors in Postgres with the pgvector extension, on
the `pgvector/pgvector:0.8.1-pg16-trixie` image, and Chroma is gone. Each
compose stack runs its own Postgres; in Azure both databases will share one
server.

The retrieval service stores each collection in its own table,
`rag_chunks__<name>`, with an HNSW index (cosine) and an index on source file
and ordinal, and records it in a `rag_collections` registry: embedding model,
dimension and chunking. The API refuses (503) to serve a collection built with
a different embedding model than the one it's configured with, and a seed
refuses to add vectors from another model, dimension or chunking. It talks to
Postgres through psycopg 3 and a small connection pool, without an ORM.

BM25 stays `rank_bm25` in the process: the hybrid retriever builds it from the
collection's stored chunks, so the pickle is gone and Postgres is the only
store, and re-seeding embeds only the chunks the collection doesn't hold yet.
The agents' memories live in one `memories` table (Alembic migration 0007)
whose vector column has no fixed dimension and no ANN index: a query filters
one user's memories of one kind, then ranks them by exact cosine distance.

## Consequences

- One kind of store to deploy, back up and inspect with SQL. Azure's managed
  Postgres needs `vector` allow-listed (`azure.extensions`) before the first
  migration, a Phase 4 task.
- A running retrieval API sees chunks that another process adds right away in
  dense mode, but in hybrid mode only after it rebuilds its retriever (in
  practice, a restart), as with the pickle.
- HNSW is approximate, and its recall isn't measured here. At a few thousand
  chunks an exact scan would also be fast enough.
- Filtering a user's memories before ranking them can't lose results the way
  an approximate search followed by a filter can; it scans that user's rows,
  which stay few. Mixing embedding dimensions (256-dim mock vectors and
  1536-dim OpenAI vectors) for one user and kind fails at query time, so a
  deployment sticks to one embedder.
- Collection names are capped at 44 characters, so the index names derived
  from them fit Postgres's 63-byte identifiers.
- The image pins Debian trixie because the agents' existing data was
  initialized under its glibc (2.41); the default bookworm variant (2.36) made
  Postgres warn that text indexes might sort differently.

## Alternatives

- **Keep Chroma.** Two vector stores and a pickle, none of which fits in the
  free tier's one Postgres.
- **Postgres full-text search for the sparse side.** `ts_rank` isn't BM25, and
  its parser splits the compound tokens (`FERRY-429`,
  `ferry.worker.concurrency`) that the BM25 tokenizer keeps whole, which are
  the exact-token matches hybrid retrieval exists for.
- **IVFFlat instead of HNSW.** Its lists are trained on the rows present when
  the index is built, so it needs rebuilding as a collection grows.
- **One table for every collection.** An indexed vector column has a single
  dimension, and collections built with different embedding models don't.
