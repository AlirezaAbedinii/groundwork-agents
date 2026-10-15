# eval

One evaluation suite for both services: retrieval, answers and refusals in the
retrieval service, and multi-step tasks in the agent service. It is a small project
of its own, like [`mcp/`](../mcp/README.md), and talks to both services over HTTP
only, so it measures what actually runs ([ADR 0006](../docs/decisions/0006-evaluation-harness-and-judge.md)).
It replaces the retrieval service's own harness, which has been retired
([last version](https://github.com/AlirezaAbedinii/groundwork-agents/tree/940677e9078e4cd4d8a70aecb74f26c8295435d0/services/rag/eval)).

Every published number can be recomputed for $0 from the records committed under
[`reports/`](reports/): `make eval-score` at the repository root re-scores them,
byte for byte, without a service or a key.

## Results at a glance

**Retrieval** ([full report](reports/rag.md)), 57 answerable questions, top 10 from
`/v1/search`, `text-embedding-3-small` embeddings; relevance is counted per evidence
item, not per chunk:

| mode | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| dense | 0.471 | 0.675 | 0.763 | 0.678 | 0.661 |
| hybrid (BM25 + RRF + cross-encoder) | 0.459 | 0.664 | 0.761 | 0.670 | 0.655 |

**Answers**, all 75 questions through `/v1/ask` (`gpt-4o-mini`, 5 contexts in both
modes), judged by `claude-haiku-4-5`, a model from another provider:

| mode | correct [95 % CI] | faithfulness | fully supported | cost/query | P95 |
|---|---|---|---|---|---|
| dense | 0.693 [0.58, 0.79] | 0.910 | 0.814 | $0.00019 | 5.4 s |
| hybrid | 0.667 [0.55, 0.76] | 0.963 | 0.884 | $0.00020 | 6.2 s |

There is no evidence of a correctness difference between the modes on this set.
Every unanswerable question is refused in both modes (end-to-end recall 1.000 on the
test split), at a precision of 0.647 (dense) and 0.579 (hybrid): some answerable
questions are refused too.

**Refusal gate**: one threshold per mode, chosen on the dev split and measured on
the test split ([ADR 0007](../docs/decisions/0007-refusal-threshold-calibration.md)):
dense 0.617442 (precision 0.615, recall 0.727), hybrid 0.963599 (precision 0.500,
recall 0.545). The previous shared 0.30 refused nothing in dense mode.

**The judge against a person**: 30 judged answers graded pass or fail blind to the
judge: raw agreement 0.933, Cohen's κ 0.830. In one of the two disagreements the
judge was too lenient: q019 rated 4 an answer that never says what plain
`uv cache prune` removes, against its own rule that a missing part rates 3 or below.
In the other, the person's pass was the miss: s024's answer never gives the
reference's reason. Each rater erred once.

**Agent tasks** ([full report](reports/agent.md)), 13 tasks run under both reviewer
policies (`REVIEWER_PROVIDER`): `cross` puts the reviewer on another provider than
the specialists (`claude-sonnet-5` reviewing OpenAI output), `same` keeps it on theirs
(`gpt-4o`):

| | cross | same |
|---|---|---|
| task success | 1/13 | 1/13 |
| completed | 7/13 | 11/13 |
| reviewer mean score (1–5) | 3.30 | 4.36 |
| rework rate | 0.250 | 0.070 |
| escalations | 40 | 13 |
| cost per task | $0.081 | $0.034 |
| wall-clock per task | 143 s | 74 s |

The two policies tie on success, on different tasks. The cross-provider reviewer is
far stricter and costs 2.4 times as much, but most failures start before any
review (see the findings below). The clearest difference: under `same`, `gpt-4o`
reviewed 4 out of 5 a research note that had made up vector databases and star
counts the demo database doesn't hold; under `cross`, the same failure was rejected.

## What is measured, and why

- **Retrieval** from `/v1/search`: recall, MRR and nDCG at k over evidence items. A
  golden question's evidence is a list of facts, each one or more verbatim quotes; a
  hit is relevant to a fact if it contains one of its quotes. Labels are text, not
  chunk ids, so they survive re-chunking, and a fact found twice (overlapping
  windows) counts once.
- **Refusals**: the gate refuses before generating when the top hit scores below its
  mode's threshold. Thresholds are swept on the dev split and reported on the test
  split with Wilson intervals; refusals by the model itself are counted apart.
- **Answers** from `/v1/ask`, with the same 5 contexts in both modes: correctness
  against a reference answer (a rating of 4 or 5 passes) and faithfulness (the share
  of the answer's atomic claims its contexts support, judged without the reference).
  The judge comes from another provider than the generator; its verdicts are
  structured, versioned and cached.
- **Agent tasks** through the agent API, one at a time, each under a fresh user so
  no task's memories reach another: task success from a rubric judge that sees only
  the deliverables, tool-call recall, precision and error rate from the agent's
  invocation log, cost (LLM plus `rag_ask`) and wall-clock, escalations, the
  reviewer's scores and the rework rate ([ADR 0008](../docs/decisions/0008-agent-evaluation.md)).

## The corpus and the golden set

The corpus, `toolchain_docs`, is the English documentation of uv 0.12.21,
Pydantic v2.13.5 and FastAPI 0.142.2 (without its release notes): 253 Markdown files
under permissive licenses, fetched at pinned release commits by
[`scripts/fetch_corpus.py`](../services/rag/scripts/fetch_corpus.py) and never
committed. It indexes to 3,571 chunks (fixed 800/120). Exact tokens (environment
variables, flags, config keys) and vocabulary shared across the three projects
("dependencies" means two things) keep the dense-vs-hybrid question meaningful.

The [golden set](golden/golden_set.jsonl) ([schema and rules](golden/SCHEMA.md)) has
75 questions: 37 lookup, 14 multi-hop, 18 with no answer in the docs, and 6 ambiguous,
split 40/60 into dev and test within each category. Fifty are targeted: written from
the docs to a coverage spec (exact-token lookups, paraphrases, cross-project
multi-hops, near-miss and out-of-scope refusals) and checked by hand. The other 25 are
synthetic: drafted by `gpt-4o-mini` from a single chunk, then verified by hand. Of 40
drafts, 10 were rejected (4 not answerable from their chunk, 3 trivial, 2 meaningful
only in the chunk's own example, 1 partial), and every accepted question and answer
was rewritten, because the drafts copied their chunk's wording or answered beyond
their quote. Synthetic questions still flatter retrieval: R@10 0.800 against 0.733
for targeted ones (dense), 0.840 against 0.699 (hybrid).

The [agent tasks](tasks/agent_tasks.jsonl) are 18 requests: 6 docs answers saved to
files, 3 analyses of the demo database, 2 code tasks, 4 that combine docs with data
or code into a memo, 2 the docs can't answer, and a three-step chain. Each names the
calls it needs (`must_call`, where "consult the docs" accepts `rag_search` or
`rag_ask`), the tools it may also use and the ones it must not, and 3–5 rubric
criteria (70 in all, 63 critical). Two more, in
[`smoke_tasks.jsonl`](tasks/smoke_tasks.jsonl), replay the agent's recorded test
fixtures for a keyless run of the whole harness.

## Reproducing

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
.venv/bin/python -m pytest -q          # the live (paid) tests are deselected unless -m live
make -C .. eval-score                  # re-score every published report: $0, no services

.venv/bin/python -m dataset.validate golden/golden_set.jsonl \
  --corpus ../services/rag/data/raw/toolchain_docs --rag http://localhost:8000
.venv/bin/python -m dataset.validate tasks/agent_tasks.jsonl --tasks
```

**Free, keyless:** seed the corpus with local embeddings (the `.env.example` in
`services/rag` shows the MiniLM settings), then collect and score retrieval; the
MiniLM numbers differ from the published ones, which used `text-embedding-3-small`.
The agent smoke is free too: start the agent API in inline mode with `MOCK_LLM=1`, a
migrated and seeded database and `MCP_RAG_URL` pointing at the RAG stack's MCP
server, then run `make eval-agent-smoke` from the repository root.

```bash
.venv/bin/python -m rag_eval collect --stage retrieval --rag http://localhost:8000 --run-id my-run
.venv/bin/python -m rag_eval score --run-id my-run              # -> runs/my-run/report/
```

**Paid runs** take a spending cap and stop before a call could pass it; collected
runs resume, and a recorded verdict is never paid for twice:

```bash
.venv/bin/python -m rag_eval collect --stage answers --rag http://localhost:8000 \
  --run-id my-run --max-cost-usd 1.00                       # generation + judge
.venv/bin/python -m agent_eval run --label cross --run-id my-run --max-cost-usd 1.50
.venv/bin/python -m agent_eval score --run-id my-run --out reports   # publish
```

The published runs cost $2.31 in all: indexing $0.01, synthetic drafts $0.004, the
judge's sanity checks $0.13, the answer run $0.58, and the agent runs $1.58.

## Findings

**Agents.** The tasks fail mostly for reasons a reviewer can't fix:

- **The planner gives database lookups to the wrong specialist.** "From the demo
  database…" goes to research, which has no `db_query` tool (t07–t09, t12). The
  cross reviewer rejects every attempt, and approving an escalation retries the same
  specialist, so the task loops until the harness stops approving (5 approvals).
  Under `same`, the reviewer approved one of these runs with invented data.
- **Research reads snippets.** It searches (`rag_search` shows each hit's first 300
  characters) and rarely asks (`rag_ask` answers from whole passages), so the notes
  it hands the writer miss the facts the rubrics check.
- **The planner over-splits.** Even two-step tasks become about five subtasks, each
  reviewed, and some are impossible, such as testing `uv` commands in a sandbox
  without `uv` or a network.
- **Outside knowledge leaks in.** Asked for Uvicorn's keep-alive timeout from the
  docs only, both arms offered the `--timeout-keep-alive` option the docs never
  mention, and the cross arm stated a 5-second default as well.
- **The reviewer is most of the cost.** `claude-sonnet-5` writes about 470 billed
  output tokens per review (it reasons before it answers), `gpt-4o` about 130.

**Retrieval and answers.** With `text-embedding-3-small`, hybrid retrieval no longer
beats dense (on the local MiniLM embeddings it led clearly): the reranker's edge on
exact tokens is matched by the stronger embeddings. Ambiguous questions score 0/6 in
both modes, because the generation prompt never asks the model to name the readings.
Hybrid's threshold sits on the steep end of the reranker's scores, so several
answerable questions just under it are refused. MCP `search` keeps 3 hits of 300
characters: 5 of 150 lost 8 points of recall.

## Limits

- **Small samples.** 57 answerable questions, 13 agent tasks per policy: the
  intervals are wide, and 1/13 against 1/13 can't separate the reviewer policies.
- **One run per configuration.** Models sample, and repeated runs would differ: t10
  passed under one policy and failed under the other, where its "run the code"
  subtask failed twelve times before any review.
- **Five agent tasks were not run** (t13, t14, t15, t17 and t18), to keep the agent
  runs within their spending cap: the chain and three of the four combined tasks are
  unmeasured.
- **The judge is checked, not proven.** κ over 30 answers is a coarse check. The
  judge rates a complete answer that adds correct detail 4 rather than 5, so the mean
  rating understates thorough answers (the pass rate is unaffected), and at the
  faithfulness margin it errs both ways on about 2–3 % of claims.
- **The synthetic third** shares its chunks' wording; read retrieval by origin.
- **Thresholds are tied to `text-embedding-3-small` and the reranker** and have to be
  calibrated again when either changes; `/v1/config` shows what a service runs.

## Layout

| Path | What it is |
|---|---|
| `schemas.py` | Golden questions, agent tasks, judge verdicts and metric results |
| `textnorm.py` | How a quote is matched against chunk text, for labels and scoring alike |
| `golden/` | The golden set, its [schema and rules](golden/SCHEMA.md), the human grades and the rejected synthetic drafts |
| `tasks/` | The agent tasks and the two keyless smoke tasks |
| `dataset/validate.py` | Checks the golden set against the schema, the corpus files and the served chunks, and the agent tasks against the agent's tools |
| `dataset/synthesize.py`, `dataset/review.py` | Draft synthetic candidates from single chunks (paid), and accept, edit or reject them by hand |
| `dataset/grade.py` | Grades a sample of judged answers by hand, blind to the judge, for the judge-human κ |
| `llm.py`, `budget.py` | Structured completions from OpenAI or Anthropic, priced, under a spending cap |
| `metrics.py` | Retrieval, refusal, answer, agreement and agent metrics: pure functions over recorded runs |
| `judge.py`, `judge_prompts.py` | Correctness, faithfulness and task-rubric verdicts from another provider's model, cached by prompt version |
| `judge_fake.py` | A scripted judge for tests and keyless runs |
| `rag_eval.py` | Collects searches and answers from the RAG API into `runs/` and scores them |
| `agent_eval.py` | Runs agent tasks through the agent API into `runs/` and scores them |
| `report.py`, `svg.py` | The markdown and JSON reports, the compact records they're re-scored from, and the refusal curves |
| `reports/` | The published reports and their records |
