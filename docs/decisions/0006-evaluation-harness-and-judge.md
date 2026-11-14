# 0006. One evaluation suite over HTTP, evidence-quote labels and a judge from another provider

Date: 2026-10-06. Status: accepted.

## Context

The retrieval service had its own evaluation harness, and its numbers had stopped
telling anything apart. Its golden questions were labelled with file names over a
seven-file demo corpus, so "a labelled file in the top k" and faithfulness both sat
at 1.000. Its judge came from whichever provider generated the answers, so a model
graded its own family's output. A correct refusal by the model (the gate let the
question through and the model declined) was scored as a wrong answer, because only
the gate's refusals were flagged. And dense mode sent ten contexts to the generator
where hybrid mode sent five, so part of the cost and quality gap between the modes
was the context size, not the retrieval. The agents had no evaluation at all.

## Decision

One evaluation suite, `eval/` at the repository root, evaluates both services over
their HTTP APIs and nothing else. For the retrieval service:

- **Labels are evidence quotes.** Each golden question lists the facts its answer
  needs; each fact is one or more verbatim quotes from the corpus, any of which
  covers it. A retrieved chunk is relevant to a fact if it contains one of its
  quotes, and a fact found by several chunks counts once, at its first rank, so
  recall and nDCG measure facts found, not chunks.
- **The judge comes from another provider than the generator**: Anthropic's
  claude-haiku-4-5 grades gpt-4o-mini's answers. Its verdicts are structured
  (reasoning before the rating or the supported flag), its prompts carry a version,
  and verdicts are cached by provider, model, version and prompt text. The harness
  refuses a judge from the generator's provider unless told otherwise, and the
  report says so when it is.
- **Both modes get the same `top_k` (5) when answering**, so the modes differ only in
  retrieval. Retrieval metrics use `/v1/search` at 10.
- **Collect once, score many.** `collect` is the only step that spends: it records
  raw responses and verdicts under a mandatory spending cap. `score` is free and
  deterministic; it writes the report and a compact records file from which every
  published number can be recomputed without the run, the golden set or a key.
- **A person grades 30 answers blind**, and the report publishes Cohen's κ between
  the judge and them. This wasn't in the original plan for the phase; it's added
  because whether the judge agrees with a person is the first question anyone asks
  about an LLM judge, and answering it costs no API spend.

## Consequences

- Every published number can be recomputed for $0 from the committed records, and
  fixing a metric never means paying for the runs again.
- Labels survive re-chunking: they never mention chunk ids, and relevance is
  computed from the hit text at scoring time.
- The judge's agreement with a person is measured, not assumed, though 30 items make
  κ a coarse check. Known bias: the judge rates a complete answer that adds correct
  detail beyond the reference 4 rather than 5, despite its prompt, so mean ratings
  understate thorough answers; the reports lead with the pass rate (4 or 5), which
  this doesn't change.
- Going through HTTP means the suite measures what actually runs, including a
  deployed service at another URL, at the price of needing the services up to
  collect (never to score).
- A refusal by the model now counts as a refusal, and refusal precision and recall
  are reported end to end as well as for the gate alone.
- The suite depends on the services' response shapes (`/v1/search`, `/v1/ask`,
  `/v1/config`); changing them means changing the harness too.

## Alternatives

- **RAGAS or DeepEval.** Another dependency with its own metric definitions and
  judge prompts; here the metrics and the prompts are the thing to own and explain.
- **Evaluating in-process.** It would couple the harness to both services' internals
  and dependency sets, and wouldn't measure a deployed service.
- **Chunk-id labels.** Any change to chunking invalidates all of them, and with
  overlapping windows "which of the two chunks holds the fact" is arbitrary.
- **A judge from the generator's provider.** Cheaper to set up (one key), but a
  model grading its own family's answers is the bias the change exists to remove.

## Addendum, 2026-11-14: the golden set grows to 197 rows

With 75 rows, each question type has a dozen or so rows and wide intervals, so 125
targeted rows were written to strata fixed before any
question was written, checked one by one against their quotes and pages, and the
122 that passed were added (`eval/golden/SCHEMA.md`, "Expansion").

- **Strata.** A stratum is category × phrasing. `phrasing` is new and set only on
  lookup and multi-hop rows: `exact_token` when the question names the flag,
  variable, function or setting the evidence is about, `paraphrase` when it avoids
  the passage's key term. The plan was 60 lookup, 30 multi-hop, 10 ambiguous and 25
  no-answer rows, half of each answerable category in each phrasing, spread over uv,
  Pydantic and FastAPI by their share of the corpus's chunks. The check rejected 3
  and moved 3 between categories, so the added rows are lookup 30 exact-token and 31
  paraphrase, multi-hop 15 and 14, 9 ambiguous and 23 no-answer. The gaps are left
  open rather than filled with unchecked rows; the rejects and their reasons are in
  `targeted_rejects.jsonl`.
- **Written without running retrieval.** The questions, quotes and answers come from
  the corpus files alone: no retrieval service, index, retrieval record or report was
  consulted while choosing or writing them, so no question is in the set because one
  mode finds it. This replaces the pooling step for these rows: every passage a
  search of the whole corpus finds stating a fact is an alternative quote for it.
  Labels written this way don't lean toward either retriever, but a chunk that states
  a fact in words the search missed counts as irrelevant; the `--rag` check against
  the served chunks is still to run.
- **Split rule.** The 75 rows keep their ids, text and splits, byte for byte, and a
  test pins them. `python -m dataset.promote` moves only rows marked verified and
  splits them inside each stratum of the promoted rows: ordered by the SHA-256 of a
  fixed seed and the id, the first round(0.4 × n) go to dev and the rest to test
  (49 dev, 73 test). The split depends only on the ids, strata and seed, so it is
  reproducible, independent of the order the rows were written in, and a row never
  moves once split. Splitting per stratum rather than per category keeps
  exact-token and paraphrase questions in the same proportion on both sides of the
  threshold choice.

The published RAG report is unchanged: it is scored from its committed records, which
cover the first 75 rows. The new rows enter a report only with a new paid run.
