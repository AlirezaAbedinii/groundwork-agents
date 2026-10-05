# Golden set: schema and rules

`golden_set.jsonl` is the ground truth the retrieval, refusal and answer evaluations
score against: one JSON object per line, over the `toolchain_docs` corpus (the
English docs of uv, Pydantic and FastAPI at pinned releases). The schema in code is
`GoldenQuestion` in [`../schemas.py`](../schemas.py); `python -m dataset.validate`
checks every rule below that a program can check.

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | `q001`… for targeted rows, `s001`… for synthetic ones. Never reused. |
| `question` | string | The question as a user would ask it. |
| `reference_answer` | string | The answer, 1–3 sentences, using only facts in the evidence quotes. |
| `category` | string | `lookup`, `multi_hop`, `no_answer` or `ambiguous` (below). |
| `evidence` | list of items | The facts the answer needs; each item is `{"quotes": [...]}`. `[]` for `no_answer`. |
| `origin` | string | `targeted` or `synthetic` (below). |
| `split` | string | `dev` or `test`; filled in by `--assign-splits`. |
| `verified` | bool | `true` on every row in the set. |
| `notes` | string | What the row tests and, for `no_answer`, how the absence was checked. |

A quote is `{"source": "uv/concepts/cache.md", "text": "..."}`, where `source` is
the file's path relative to the corpus root (`services/rag/data/raw/toolchain_docs/`).
Unknown keys are errors, so a misspelled field can't slip through as a default.

## Evidence and relevance

The unit of relevance is the evidence item, not the chunk. An item is one fact the
answer needs, and every item is needed. It holds one or more quotes, any one of
which suffices: when the same fact is stated on two pages, both quotes go in one
item as alternatives rather than in two items.

A missing alternative makes a correct hit count as irrelevant. So when a row is
written, the top 10 hits of both retrieval modes are read, and every chunk that
states one of the row's facts gets its own quote added to that item (pooling, as in
TREC). What counts is the chunk a quote marks, not the quote alone, and a chunk that
only shows an example of the fact without stating it doesn't qualify. A page that no
search surfaced can still be missed, so the labels lean toward the retrievers used
for pooling.

At scoring time a retrieved chunk is relevant if it contains a quote of some item
(`textnorm.covered_items`: whitespace-insensitive, case-sensitive). An item covered
by several chunks counts once, at its first rank, so overlapping chunk windows don't
inflate recall. The labels never name chunks, so they survive re-chunking; the
validator's `--rag` check confirms they still fit the chunks being served.

Quote rules:

- **Verbatim from the markdown file**, not the rendered page: backticks, link syntax
  and emphasis stay as they are in the file. Line wraps and indentation don't matter.
- **12–120 characters.** The corpus is chunked in fixed windows of 800 characters
  that overlap by 120, so a quote this short always lies wholly inside some chunk.
- **One section's text.** Never a heading line and never across one: the loader
  splits each page at its `#` headings and keeps heading lines out of chunk text.
- **Specific.** Every chunk containing the quote counts as relevant, so quote the
  sentence that states the fact, not a phrase that occurs all over the docs
  (`uv pip install`). The validator warns when a quote is in more than 2 chunks.

## Categories

| Category | Question | Expected behavior | Evidence |
|---|---|---|---|
| `lookup` | One fact. | Answer it, with citations. | ≥ 1 item |
| `multi_hop` | Facts from different pages, or from different sections of one page. | Combine them in one answer. | ≥ 2 items |
| `no_answer` | Plausible for these tools, but the pinned docs don't say. | Refuse: say the docs don't cover it, and invent nothing. | `[]` |
| `ambiguous` | At least two readings that the docs answer differently ("dependencies" in uv vs FastAPI). | Name the readings instead of silently picking one. | ≥ 2 items, at least one per reading |

Rows with evidence feed the retrieval metrics (Recall@k, MRR@10, nDCG@10). A
question should be refused if and only if it is `no_answer`; that defines refusal
precision and recall. Answers are judged for correctness against
`reference_answer` and for faithfulness to the retrieved text. A `no_answer`
reference answer says what the docs do and don't cover; an `ambiguous` one names
the readings.

## Origin, split and verification

- **`targeted`** rows are written from the docs to a coverage spec with an LLM's
  help: exact tokens and paraphrases, questions that need two pages or two sections
  of one page, refusals and ambiguity. A person verifies each one against its quotes
  and their pages before it enters the set.
- **`synthetic`** rows are drafted by an LLM from a single chunk. A person checks
  each draft against that chunk and its page, rewrites what needs it, and accepts or
  rejects it. Drafts wait in `candidates_unverified.jsonl`, which isn't committed.
- **`split`**: about 40 % `dev` and 60 % `test` within each category. `dev` is used
  only to choose the refusal thresholds; numbers that depend on a threshold are
  reported on `test`. `--assign-splits` fills in missing splits in an order fixed by
  a hash of the ids, so the dev rows aren't simply the first ones written.
- **`verified: true`** on every row: unreviewed rows don't belong in this file.

Target composition (75 rows; the validator reports the gaps as warnings):

| | lookup | multi_hop | no_answer | ambiguous | total |
|---|---|---|---|---|---|
| targeted | 12 | 14 | 18 | 6 | 50 |
| synthetic | 25 | | | | 25 |
| total | 37 | 14 | 18 | 6 | 75 |

## Rules

1. **Every row is verified by hand.** A person checks the question, the reference
   answer and every quote against the source pages and corrects them before the row
   enters the set: the question reads like a user's, the quotes answer it, and the
   reference answer is correct and complete. The two origins are kept apart so that
   every metric is also reported per origin: questions drafted from one chunk tend
   to borrow its words, which flatters retrieval.
2. **A reference answer uses only facts in its quotes.** A `no_answer` row has none:
   its answer says what the pinned docs don't cover and may name the nearest feature
   they do cover, with that page recorded in `notes`.
3. **`no_answer` rows stay absent from the pinned docs.** Check each one by grepping
   the corpus for its key terms and reading the top search hits in both retrieval
   modes, and write down what was checked in `notes`. When the corpus pin changes,
   check them again.
4. **Re-validate after any change** to the corpus pin or the chunking, with
   `--corpus` and `--rag`.
5. **One id forever.** Don't renumber or reuse ids; a removed row's id stays retired.
6. **A row never moves between `dev` and `test`** once it has a split.

## Checking

```bash
cd eval
.venv/bin/python -m dataset.validate golden/golden_set.jsonl \
  --corpus ../services/rag/data/raw/toolchain_docs --rag http://localhost:8000 --assign-splits
```

Without options it checks the schema, unique ids, `verified` and the composition,
offline. `--corpus` finds every quote in its file and section, and prints the
closest line of a near miss. `--rag` finds every quote in the chunks the RAG API
serves (`/v1/chunks`), which also catches one lost with a duplicate chunk, and
flags quotes that aren't specific. The corpus comes from
`services/rag/scripts/fetch_corpus.py toolchain_docs`. Errors exit 1; warnings don't.

## Template

Leave out `split`; `--assign-splits` adds it.

```json
{"id": "qNNN", "question": "", "reference_answer": "", "category": "lookup", "evidence": [{"quotes": [{"source": "", "text": ""}]}], "origin": "targeted", "verified": true, "notes": ""}
```

An item with two alternative quotes: `{"quotes": [{"source": "a/x.md", "text": "..."}, {"source": "b/y.md", "text": "..."}]}`.

## The corpus

The quotes are short excerpts from the documentation of
[uv](https://github.com/astral-sh/uv) 0.12.21 (MIT OR Apache-2.0),
[Pydantic](https://github.com/pydantic/pydantic) v2.13.5 (MIT) and
[FastAPI](https://github.com/fastapi/fastapi) 0.142.2 (MIT), at the commits pinned
in [`services/rag/corpora/toolchain_docs.json`](../../services/rag/corpora/toolchain_docs.json).
