# 0021. Hybrid search re-ranks with gte-reranker-modernbert-base, on the same chunks

Date: 2026-11-28. Status: accepted.

## Context

Hybrid mode fuses the dense top 20 with the BM25 top 20 by weighted Reciprocal Rank
Fusion (k 60, 0.7 dense / 0.3 BM25) and re-ranks the fused 20 with a cross-encoder. With
`cross-encoder/ms-marco-MiniLM-L-6-v2` as that cross-encoder and `text-embedding-3-small`
embeddings it was no better than dense: Recall@10 0.761 against 0.763 on 57 answerable
questions. Three causes were suspected: a small re-ranker trained on web passages
overriding a good first stage, a BM25 leg too light to add candidates, and fixed
800-character chunks that ignore headings.

The golden set has since grown to 197 questions, 156 of them answerable (63 dev, 93
test; [ADR 0006](0006-evaluation-harness-and-judge.md), addendum). Ten retrieval
configurations were searched on dev to choose between them, and on test once, after the
choice ([`eval/reports/rag-ablation/`](../../eval/reports/rag-ablation/ablation.md)). The
rule was to choose on dev by nDCG@10, then Recall@5, and to report any re-ranker slower
than 2 s a query on the cloud app's CPU.

The constraints are the cloud app's 1 vCPU and 2 GiB, one replica, and the 30 s timeout
on every MCP and agent call to the retrieval service. The agent reads the top 3 hits of a
search whole ([ADR 0004](0004-mcp-as-the-seam.md), addendum of 2026-11-16).

## Decision

Hybrid mode re-ranks with `Alibaba-NLP/gte-reranker-modernbert-base` (Apache-2.0, 149.6 M
parameters, 575 MB of weights), on the same fixed 800/120 chunks, with the same pool of
20 and the same fusion: arm A4a. The model is baked into the image; the app keeps 1 vCPU
and 2 GiB. Dense mode is unchanged.

Every arm, nDCG@10 on dev (where the choice was made) and on test, the test difference to
dense with its 95 % paired bootstrap interval, and the search time on one pinned core
(test, median / p95):

| Arm | Change | dev nDCG@10 | test nDCG@10 | test − dense | search, one core |
|---|---|---|---|---|---|
| A0 | dense | 0.647 | 0.601 | – | 0.2 / 0.2 s |
| A1 | hybrid as it was: MiniLM, pool 20 | 0.660 | 0.641 | +0.039 [−0.002, +0.080] | 1.9 / 2.5 s |
| A2 | A1 without the re-ranker | 0.660 | 0.619 | +0.018 [−0.015, +0.051] | 0.2 / 0.3 s |
| A3 | A1 without BM25 | 0.660 | 0.641 | +0.039 [−0.002, +0.080] | 1.9 / 2.5 s |
| **A4a** | **A1 with gte** | **0.715** | **0.698** | **+0.096 [+0.055, +0.140]** | **8.1 / 14.4 s** |
| A4b | A1 with granite-embedding-reranker-english-r2 | 0.667 | 0.674 | +0.073 [+0.030, +0.118] | 8.2 / 12.9 s |
| A5a | A4a, pool 50, fusion 0.5/0.5 | 0.719 | 0.724 | +0.123 [+0.077, +0.172] | 19.3 / 24.7 s |
| A5b | A4a, pool 50 | 0.746 | 0.728 | +0.127 [+0.079, +0.177] | 18.8 / 26.0 s |
| A6 | A5b on heading-path chunks | 0.768 | 0.728 | +0.127 [+0.072, +0.184] | 18.8 / 25.6 s |
| A6d | dense on the heading-path chunks | 0.621 | 0.589 | −0.013 [−0.068, +0.044] | 0.2 / 0.3 s |

Why A4a:

- **It is what the agent sees that improves.** At the top 3, A1 is level with dense
  (test Recall@3 +0.019 [−0.053, +0.091]) and A4a is not (+0.124 [+0.063, +0.189]).
  A4a beats A1 on test nDCG@10 by +0.057 [+0.010, +0.107] even at the 99.4 % interval
  that allows for the eight comparisons made (27 questions better, 8 worse).
- **Over A6 and A5b it is a latency and cost choice, not a quality one.** A6 and A5b lead
  on dev (nDCG@10 0.768 and 0.746 against 0.715) and are no worse on test. They re-rank 50
  candidates: 18.7 s median and 27 to 31 s at p95 on one core, at the 30 s timeout before
  any generation, and 9.4 / 14.5 s on 4 cores, which means four times the vCPU and memory
  billing. A6 also needs a re-index and makes dense mode slightly worse on its chunks
  (A6d). Their dev lead over A4a is within what the set can resolve: A6 − A4a +0.053,
  99.4 % interval [−0.013, +0.124]; A5b − A4a +0.031 [−0.010, +0.079] at 95 %.
- **No re-index and no added cost.**

**BM25 changes nothing at these settings.** A chunk only BM25 finds scores at most
0.3 / 61 = 0.0049 in the fusion, and the dense chunk at rank p at least 0.7 / (60 + p),
which is more for every pool up to 82. So the fused 20 are exactly the dense top 20, in an
order the re-ranker then discards; A3 equals A1 in every aggregate. The gain is the
re-ranker's. BM25 stays on, so hybrid keeps its keyword leg for other weights and for the
keyless embeddings, at the cost of the in-memory index.

**PK-21's 2 s limit is waived for this arm.** It was a limit to stop at and report, and
gte cannot meet it on any core count measured: 20 pairs take 3.4 s even on 8 cores.
Measured search time with A4a is 8.1 / 14.4 s median / p95 on one pinned core (test; 8.5 /
11.7 s on dev) and 4.7 / 5.6 s on 4 cores ([`rag-ablation-cores`](../../eval/reports/rag-ablation-cores/ablation.md)).
The binding limit is the 30 s timeout on MCP and agent calls. In the published run, with
the service limited to 1 CPU and 2 GiB as in the cloud, a hybrid search took 15.6 s at
the median, 18.6 s at p95 and 24.2 s at most, and an answer 18.3, 23.1 and 28.2 s
([report](../../eval/reports/rag.md)). No call reached 30 s, but the slowest answer came
within 2 s of it.

Two conditions came with the choice:

- **One mapping into [0, 1].** sentence-transformers applies a sigmoid of its own to a
  one-label model whose config names no activation, such as gte, and none to MiniLM, whose
  config says identity. The service's own sigmoid was then a second one for gte, which
  squeezed its top scores into 0.70 to 0.73, below every calibrated threshold. The scorer
  now asks every model for its logits and the re-ranker maps them once. Rankings were
  unaffected, so the ablation stands; its gte and granite records keep the old scores.
- **Torch threads at the CPU limit.** A container limited by quota sees every host core,
  and torch starts a thread per core: MiniLM ran in 6.6 s instead of 1.7 s under a 1-CPU
  quota on 8 cores. The service now sets torch's threads to the CPUs its cgroup quota
  allows. That is not enough on its own: the OpenMP and tokenizer threads started per
  host core still take part of the quota. In the published run the re-ranker took 15.4 s
  at the median for 20 candidates, and the service was throttled in 29 % of its
  scheduling periods; the same re-rank takes 7.4 s on one pinned core. In a benchmark of
  the same image, capping those threads too (`OMP_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false`) brought the quota case to 7.4 to 7.6 s, with 1 to 2 %
  of periods throttled. That cap is left to a later change, so the published latency is
  this image's.

## Consequences

- A hybrid search takes seconds instead of milliseconds. An agent task with two searches
  spends about 31 s in them on one vCPU at the published median, and concurrent calls
  queue behind the one replica.
- The image grows by 0.41 GB, to 2.23 GB of layers; the weights are stored once (a later
  `chown` had copied them into a second layer). Memory charged to the container peaked
  at 862 MiB of its 2 GiB in the published run. The process's peak resident set reads
  2.07 GiB because `/v1/ask` and `/v1/search` each build their own retriever and map the
  model file, and mapped weights are file pages the kernel can reclaim. Sharing one
  retriever per mode is a later change.
- The refusal thresholds were calibrated again for the new scores
  ([ADR 0007](0007-refusal-threshold-calibration.md), addendum of 2026-11-28).
- Going back to A1 is a settings change, but it means calibrating and republishing again.
- Not measured: Azure's CPU speed, quota behaviour and the cold start with a larger
  image. They are measured before any cloud demo.

## Alternatives

- **A1, keeping MiniLM.** Near 2 s a search, but level with dense on what the agent reads,
  and its nDCG@10 gain over dense is not significant on either split.
- **A4b, granite-embedding-reranker-english-r2.** As slow as gte, no better than MiniLM on
  dev.
- **A5a, A5b, A6: a pool of 50, heading-path chunks.** Better on dev, not significantly,
  for 2.3 times the re-rank time; see above.
- **A7, a generated context per chunk.** Not run: it inherits A6's latency and would cost
  up to $2.50 to embed.
- **A8, the keyword leg from Postgres full-text search.** Not run; at 0.7/0.3 the keyword
  leg adds nothing to measure.
- **BM25 off.** Identical results with less memory and no index to build, but hybrid
  would then mean dense plus a re-ranker.
- **ONNX or quantised models, a GPU, a hosted re-ranking API.** A new dependency, a cost,
  or a new service.
