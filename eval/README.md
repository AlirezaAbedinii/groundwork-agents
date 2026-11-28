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

**Retrieval** ([full report](reports/rag.md)), 156 answerable questions, top 10 from
`/v1/search`, `text-embedding-3-small` embeddings; relevance is counted per evidence
item, not per chunk:

| mode | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| dense | 0.391 | 0.651 | 0.736 | 0.604 | 0.620 |
| hybrid (BM25 + RRF + `gte-reranker-modernbert-base`) | 0.494 | 0.750 | 0.785 | 0.711 | 0.705 |

On the test split, question by question, hybrid's nDCG@10 is +0.096 above dense
[+0.055, +0.140] and its Recall@3, the hits an agent reads whole, +0.124 [+0.063,
+0.189] (95 % paired bootstrap intervals). The configuration was chosen on dev from ten
([ADR 0021](../docs/decisions/0021-hybrid-reranks-with-gte.md)).

**Answers**, all 197 questions through `/v1/ask` (`gpt-4o-mini`, 5 contexts in both
modes), judged by `claude-haiku-4-5`, a model from another provider:

| mode | correct [95 % CI] | faithfulness | fully supported | cost/query | P95 |
|---|---|---|---|---|---|
| dense | 0.685 [0.62, 0.75] | 0.913 | 0.760 | $0.00024 | 6.3 s |
| hybrid | 0.701 [0.63, 0.76] | 0.951 | 0.869 | $0.00025 | 23.1 s |

There is no evidence of a correctness difference between the modes on this set
(exact McNemar p 0.66: 12 questions only hybrid gets right, 9 only dense). Hybrid's
latency is the re-ranker on the service's one CPU. Every unanswerable question is
refused in both modes (end-to-end recall 1.000 on the test split), at a precision of
0.556 (dense) and 0.658 (hybrid): some answerable questions are refused too, 27 of 156
in dense mode and 19 in hybrid, over every split.

**Refusal gate**: one threshold per mode, the highest at which at most 10 % of the dev
split's answerable questions are refused, by the gate or by the model
([ADR 0007](../docs/decisions/0007-refusal-threshold-calibration.md), addendum of
2026-11-28): dense 0.499202, hybrid 0.895311. In the published run that refused 6 of
63 in hybrid mode and 7 of 63 in dense mode, one over the bound: q047's dense score was
0.500183 in the run the threshold was chosen from and 0.491032 in this one, because
query embeddings vary between calls, and the threshold sits on a dev score with no
margin. On the test split the gate itself refuses 5 (dense) and 7 (hybrid) of the 25
unanswerable questions, and the model declines the rest after reading the contexts.

**The judge against a person**: 30 judged answers of the previous answer run (75
questions, the same judge and prompts) graded pass or fail blind to the judge: raw
agreement 0.933, Cohen's κ 0.830. A grade joins only the exact answer it was given
for, so the current report counts the 4 answers that came out word for word the same.
In one of the two disagreements the
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
  The scoring rules are versioned: records collected from now on carry `scoring: 3`
  (a criterion about a file the run never wrote fails without a judge call, a code
  run that exits non-zero is a tool error, and cost includes the memory step, as in
  version 2; and the judge makes one call per subject, so a criterion about the answer
  sees only the final output, one about a file only that file, and one about the work
  as a whole every output), and records without the field score under the rules they
  were published with.
- **Agent diagnostics**, read from the same records with no model call and kept beside
  the run, not in the report: plan size and specialist order, whether the plan starts
  with research, completed subtasks with no tool call, calls that repeat an earlier
  call exactly, calls refused by a rate limit, tool results longer than their tool's
  budget (the characters a specialist is shown, from the agent's `/config`; 2,000 for a
  run whose config names none), critical criteria passed, and whether each
  required file was written at its exact path. They locate where a run's work went
  missing; they don't change how a run is scored. Records collected before a field
  existed show `n/a` for what needs it.
- **RAG statistics**, in the report and printed by `rag_eval stats` from a records
  file with no model call: the answerable questions each mode refused,
  by the gate and by the model, with Wilson intervals; hybrid against dense question
  by question, an exact McNemar test on correctness and a paired bootstrap interval
  (10,000 resamples, seed 0) for the difference in each retrieval metric; and how
  many readings of each ambiguous question each mode's top 5 covers. Each is given
  over all questions and per category, split and origin
  ([ADR 0007](../docs/decisions/0007-refusal-threshold-calibration.md), addendum of
  2026-11-18).

## The corpus and the golden set

The corpus, `toolchain_docs`, is the English documentation of uv 0.12.21,
Pydantic v2.13.5 and FastAPI 0.142.2 (without its release notes): 253 Markdown files
under permissive licenses, fetched at pinned release commits by
[`scripts/fetch_corpus.py`](../services/rag/scripts/fetch_corpus.py) and never
committed. It indexes to 3,571 chunks (fixed 800/120). Exact tokens (environment
variables, flags, config keys) and vocabulary shared across the three projects
("dependencies" means two things) keep the dense-vs-hybrid question meaningful.

The [golden set](golden/golden_set.jsonl) ([schema and rules](golden/SCHEMA.md)) has
197 questions, all in the published RAG results: 98 lookup, 43 multi-hop, 41 with no
answer in the docs, and 15 ambiguous, 79 in the dev split and 118 in test. The first 75
(37 lookup, 14 multi-hop, 18 no-answer, 6 ambiguous, split 40/60 into dev and test
within each category) came first. Fifty of them are targeted: written from
the docs to a coverage spec (exact-token lookups, paraphrases, cross-project
multi-hops, near-miss and out-of-scope refusals) and checked by hand. The other 25 are
synthetic: drafted by `gpt-4o-mini` from a single chunk, then verified by hand. Of 40
drafts, 10 were rejected (4 not answerable from their chunk, 3 trivial, 2 meaningful
only in the chunk's own example, 1 partial), and every accepted question and answer
was rewritten, because the drafts copied their chunk's wording or answered beyond
their quote. Synthetic questions still flatter retrieval: R@10 0.800 against 0.724
for targeted ones (dense), 0.840 against 0.774 (hybrid).

The other 122 are targeted rows added on 2026-11-14: 61 lookup and 29 multi-hop, each
marked exact-token or paraphrase, 9 ambiguous and 23 no-answer, written from the
corpus files without running retrieval, checked by hand, and split 40/60 inside each
category × phrasing stratum
([ADR 0006](../docs/decisions/0006-evaluation-harness-and-judge.md), addendum).

The [agent tasks](tasks/agent_tasks.jsonl) are 28 requests. The first 18 are 6 docs
answers saved to files, 3 analyses of the demo database, 2 code tasks, 4 that combine
docs with data or code into a memo, 2 the docs can't answer, and a three-step chain.
Each names the calls it needs (`must_call`, where "consult the docs" accepts
`rag_search` or `rag_ask`), the tools it may also use and the ones it must not, and
3–5 rubric criteria (70 in all, 63 critical). The other 10 (t19 to t28, added on
2026-11-15) are marked `held_out`: 4 docs, 2 data, 1 code and 3 combined tasks, in
the proportions of those categories above, each with three critical criteria (47 in
all, 30 critical). They draw on corpus pages the first 18 don't use, and their facts
were checked against the corpus files and the demo seed, not against the agent. Five of
the first 18, t13 to t15, t17 and t18, which no published run includes, are marked
`held_out` too. `agent_eval run` leaves all 15 out unless `--include-held-out` asks
for them, so a plain run covers the 13 tasks of the published run
([ADR 0008](../docs/decisions/0008-agent-evaluation.md), second and third addenda). Two more, in
[`smoke_tasks.jsonl`](tasks/smoke_tasks.jsonl), replay the agent's recorded test
fixtures for a keyless run of the whole harness.

## Reproducing

```bash
uv sync --locked --extra dev --python 3.12      # .venv with uv.lock's exact versions
.venv/bin/python -m pytest -q          # the live (paid) tests are deselected unless -m live
make -C .. eval-score                  # re-score every published report: $0, no services

.venv/bin/python -m dataset.validate golden/golden_set.jsonl \
  --corpus ../services/rag/data/raw/toolchain_docs --rag http://localhost:8000
.venv/bin/python -m dataset.validate tasks/agent_tasks.jsonl --tasks
.venv/bin/python -m dataset.validate golden/candidates_targeted.jsonl --candidates \
  --corpus ../services/rag/data/raw/toolchain_docs
.venv/bin/python -m dataset.review --targeted   # check the targeted candidates by hand
.venv/bin/python -m dataset.promote              # move the checked ones into the golden set
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

On real models `agent_eval run` holds `--task-reserve-usd` (default $0.25) against the
cap before each task, and only the agent's own ceiling stops a task there, so it
refuses an agent whose `/config` shows `max_task_cost_usd` at 0 or above the reserve:
start the agent with `MAX_TASK_COST_USD` above 0 and at most the reserve. A task that
times out runs on inside the agent, so it is charged at least the reserve; its record
keeps the cost measured. `rag_ask` spends in the RAG service, under neither limit,
and is counted when the task ends
([ADR 0008](../docs/decisions/0008-agent-evaluation.md), addendum of 2026-11-21).

**Comparing retrieval configurations.** `rag_eval ablate` searches the answerable
questions once per arm of an arms file ([`ablation/arms.json`](ablation/arms.json)):
each arm is a mode on a RAG service started with the arm's settings, and names the
`/v1/config` values it needs, so a service started with other settings is refused
before any search. It searches dev unless `--splits` names test, skips what an arm
has recorded (an arm with nothing left never contacts its service), and writes
`ablation.md`, `ablation.json` and `ablation-records.jsonl` into `runs/<id>/`: per
split and arm, the retrieval metrics, the service's own search latency and its
re-rank stage, and the paired bootstrap difference to the reference arm, overall and
per category and phrasing. `--records` re-scores the records file for $0. Each search
embeds the query (`text-embedding-3-small`: under $0.0001 per arm and split).

```bash
.venv/bin/python -m rag_eval ablate --arms ablation/arms.json --run-id my-ablation --only A0,A1
.venv/bin/python -m rag_eval ablate --records runs/my-ablation/ablation-records.jsonl --out /tmp/t
```

The two ablations that chose hybrid's configuration are published with their records:
[`reports/rag-ablation/`](reports/rag-ablation/ablation.md) (ten arms, chosen on dev, test
searched once) and [`reports/rag-ablation-cores/`](reports/rag-ablation-cores/ablation.md)
(the gte arms on 2 and 4 cores). `make eval-score` re-scores both
([ADR 0021](../docs/decisions/0021-hybrid-reranks-with-gte.md)).

**Calibrating the refusal thresholds.** Answer the dev questions with a service started
with both thresholds at 0, so the model's own refusals are known for every question,
then score the published run with that run beside it: `--calibration` adds its compact
rows to the records and the report chooses each threshold from them, the highest at
which at most 10 % of dev's answerable questions are refused by the gate or the model
([ADR 0007](../docs/decisions/0007-refusal-threshold-calibration.md), addendum of
2026-11-28). The fake judge is enough there, since a refusal needs no verdict:

```bash
.venv/bin/python -m rag_eval collect --stage answers --run-id gate-off --ids "$DEV_IDS" \
  --judge fake --max-cost-usd 0.20                         # generation only
.venv/bin/python -m rag_eval score --run-id my-run --calibration gate-off
```

`agent_eval score --scoring 2` scores a run collected under the first rules by
version 2's, for $0; give it its own `--out` so the published report stays as it was.
`--scoring 3` refuses records collected under 1 or 2: what the judge is shown is
decided when the run is judged, so only a new run can be judged that way.
`rag_eval stats --records reports/rag-records.jsonl` prints the RAG statistics as
markdown and writes nothing ($0, no services). `agent_eval diagnose --run-id my-run` writes `diagnostics.md` and `diagnostics.json`
into `runs/my-run/`, from the run's records alone ($0, no services).

The published runs cost $3.61 in all: indexing $0.01, synthetic drafts $0.004, the
judge's sanity checks $0.13, the retrieval ablations $0.012, the calibration run
$0.043, the answer run $1.83, and the agent runs $1.58.

## Against the Azure deployment

The harness takes the cloud URLs like local ones; `make urls` prints them, and only the
allow-listed IP can reach them. Wake the apps first (`make warm`): a cold app can
outlast the client's 30 s timeout.

```bash
.venv/bin/python -m rag_eval collect --stage retrieval --rag "$RAG_URL" --run-id azure
.venv/bin/python -m agent_eval run --ids t01 --label cross --agent "$AGENT_URL" \
  --run-id azure-agent --max-cost-usd 0.25
```

Checked on 2026-11-08 against a stack seeded with a copy of `toolchain_docs` (the same
rows and vectors as the published runs, checked by a checksum):

- **Retrieval** (150 searches, under $0.0001): 71 of 75 dense and 69 of 75 hybrid top-10
  lists matched the published records, and at the deployed thresholds the refusal gate
  decided the same on all 75 questions in dense mode and on 73 in hybrid. Every
  difference traced to the vector index, not the deployment: for each of the seven
  questions involved, an exact search returned the same top 20 on both databases, and
  one side's approximate (HNSW) index had missed a nearer chunk. The exception was a tie
  between two chunks of equal score, ordered differently because query embeddings vary
  slightly between calls. The two gate changes: q046, answerable, refused in Azure (its
  index missed the five nearest chunks); q037, unanswerable, answered in Azure (its
  index found the nearest chunk, which the local one misses). Dense MRR@10 moved 0.678 →
  0.689 and hybrid R@10 0.761 → 0.749. See
  [ADR 0010](../docs/decisions/0010-azure-topology-and-access.md).
- **One agent task** (t01, cross reviewer): completed for $0.053, the judge included,
  with `rag_search` served through the cloud MCP server and the retrieval API.

The published reports in [`reports/`](reports) stay the local runs; cloud runs go to
the gitignored `runs/`.

## Findings

**Agents.** The tasks fail mostly for reasons a reviewer can't fix:

- **The planner gives database lookups to the wrong specialist.** "From the demo
  database…" goes to research, which has no `db_query` tool (t07–t09, t12). The
  cross reviewer rejects every attempt, and approving an escalation retries the same
  specialist, so the task loops until the harness stops approving (5 approvals).
  Under `same`, the reviewer approved one of these runs with invented data.
- **Research reads snippets.** It searches (`rag_search` showed each hit's first 300
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

**Retrieval and answers.** With `ms-marco-MiniLM-L-6-v2` as its re-ranker, hybrid
retrieval was level with dense on `text-embedding-3-small`; with
`gte-reranker-modernbert-base` it finds more of the needed facts, most of all at the
top 3. BM25 adds nothing at the 0.7/0.3 fusion weights, so the gain is the re-ranker's,
and it costs about 15 s a search on one CPU against 0.2 s dense
([ADR 0021](../docs/decisions/0021-hybrid-reranks-with-gte.md)). Better retrieval has
not shown up in correctness on this set. Ambiguous questions score 1 of 15 in both
modes, even with a prompt rule asking for each reading by name: for most of them the
top 5 hits hold at most one reading. Both thresholds refuse at most about a tenth of
dev's answerable questions, so the model, not the gate, declines most unanswerable
ones. MCP `search` kept 3 hits of 300
characters, since 5 of 150 lost 8 points of recall; it now returns whole passages
within a budget, and at `top_k` 3 it shows as much evidence as whole chunks on the
test split.

## Limits

- **Small samples.** 156 answerable questions (93 on test), 13 agent tasks per policy: the
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
  calibrated again when either changes; `/v1/config` shows what a service runs. A
  threshold set on a dev question's score has no margin for the variation in query
  embeddings between calls, up to 0.009 in cosine similarity across the dev questions.

## Layout

| Path | What it is |
|---|---|
| `schemas.py` | Golden questions, agent tasks, judge verdicts and metric results |
| `textnorm.py` | How a quote is matched against chunk text, for labels and scoring alike |
| `golden/` | The golden set, its [schema and rules](golden/SCHEMA.md), the targeted candidates awaiting review, the human grades and the rejected drafts |
| `tasks/` | The agent tasks and the two keyless smoke tasks |
| `dataset/validate.py` | Checks the golden set against the schema, the corpus files and the served chunks, and the agent tasks against the agent's tools |
| `dataset/synthesize.py`, `dataset/review.py` | Draft synthetic candidates from single chunks (paid), and accept, edit or reject them by hand; `review.py --targeted` checks the targeted candidates in place |
| `dataset/grade.py` | Grades a sample of judged answers by hand, blind to the judge, for the judge-human κ |
| `llm.py`, `budget.py` | Structured completions from OpenAI or Anthropic, priced, under a spending cap |
| `metrics.py` | Retrieval, refusal, answer, agreement, paired-comparison and agent metrics: pure functions over recorded runs |
| `judge.py`, `judge_prompts.py` | Correctness, faithfulness and task-rubric verdicts from another provider's model, cached by prompt version |
| `judge_fake.py` | A scripted judge for tests and keyless runs |
| `rag_eval.py` | Collects searches and answers from the RAG API into `runs/`, scores them, prints the paired statistics, and compares retrieval configurations (`ablate`) |
| `ablation/` | The retrieval configurations `rag_eval ablate` compares, with the settings each must be served with |
| `agent_eval.py` | Runs agent tasks through the agent API into `runs/`, scores them, and diagnoses them |
| `report.py`, `svg.py` | The markdown and JSON reports, the compact records they're re-scored from, the agent diagnostics, and the refusal curves |
| `reports/` | The published reports and their records |
