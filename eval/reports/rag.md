# RAG evaluation

Run `2026-10-06-openai`, collected 2026-10-06. Collection `toolchain_docs`: 3571 chunks (fixed, 800 characters, 120 overlap), embedded with `text-embedding-3-small`. Hybrid mode adds BM25 and a cross-encoder reranker to the dense search.

Golden set: 75 questions (sha256 `6411de51afc2`): ambiguous 6, lookup 37, multi_hop 14, no_answer 18; synthetic 25, targeted 50.

Each stage records the service's settings when it starts, and every stage of a run serves the same collection:

|  | retrieval | answers |
|---|---|---|
| collected | 2026-10-06 | 2026-10-06 |
| `top_k` asked for | 10 | 5 |
| refusal thresholds (dense / hybrid) | 0.3 / 0.3 | 0.617442 / 0.963599 |
| generator | `gpt-4o-mini` (openai) | `gpt-4o-mini` (openai) |
| citation verification | off | on |

## Retrieval

The 57 answerable questions (every category but no_answer), top 10 hits from `/v1/search`. A hit is relevant to an evidence item if it contains one of the item's quotes; an item found by several hits counts once, at its first rank, so recall and nDCG measure the facts found, not the chunks.

| mode | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| dense | 57 | 0.471 | 0.622 | 0.675 | 0.763 | 0.678 | 0.661 |
| hybrid | 57 | 0.459 | 0.605 | 0.664 | 0.761 | 0.670 | 0.655 |

### By category

| mode | category | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| dense | ambiguous | 6 | 0.333 | 0.472 | 0.556 | 0.667 | 0.833 | 0.615 |
| dense | lookup | 37 | 0.541 | 0.676 | 0.703 | 0.757 | 0.606 | 0.645 |
| dense | multi_hop | 14 | 0.348 | 0.545 | 0.652 | 0.819 | 0.804 | 0.724 |
| hybrid | ambiguous | 6 | 0.389 | 0.472 | 0.528 | 0.667 | 0.854 | 0.642 |
| hybrid | lookup | 37 | 0.514 | 0.676 | 0.676 | 0.757 | 0.602 | 0.644 |
| hybrid | multi_hop | 14 | 0.345 | 0.476 | 0.690 | 0.812 | 0.769 | 0.691 |

### By origin

Synthetic questions were drafted from one chunk and share its wording, which flatters retrieval: read them against the targeted ones.

| mode | origin | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| dense | synthetic | 25 | 0.560 | 0.720 | 0.760 | 0.800 | 0.639 | 0.678 |
| dense | targeted | 32 | 0.402 | 0.546 | 0.608 | 0.733 | 0.709 | 0.648 |
| hybrid | synthetic | 25 | 0.520 | 0.720 | 0.720 | 0.840 | 0.631 | 0.681 |
| hybrid | targeted | 32 | 0.411 | 0.516 | 0.620 | 0.699 | 0.700 | 0.634 |

### By split

| mode | split | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| dense | dev | 23 | 0.386 | 0.520 | 0.578 | 0.694 | 0.600 | 0.578 |
| dense | test | 34 | 0.529 | 0.691 | 0.740 | 0.809 | 0.731 | 0.717 |
| hybrid | dev | 23 | 0.297 | 0.543 | 0.572 | 0.690 | 0.538 | 0.550 |
| hybrid | test | 34 | 0.569 | 0.647 | 0.725 | 0.809 | 0.759 | 0.726 |

## Snippet size for MCP search

MCP `search` gives an agent each hit cut to its first characters. Evidence recall on the test split (34 answerable questions) with whole chunks and with the snippets an agent sees:

| mode | n | R@3 full | R@3 300 chars | R@5 full | R@5 150 chars |
|---|---|---|---|---|---|
| dense | 34 | 0.691 | 0.272 | 0.740 | 0.120 |
| hybrid | 34 | 0.647 | 0.208 | 0.725 | 0.125 |

Rule: switch MCP from 3 hits of 300 characters to 5 of 150 only if that raises hybrid recall by at least 5 points. Here 5 × 150 is -8.3 points: **keep 3 × 300**.

## Refusal gate

The gate refuses before generating when the top hit's score is below the mode's threshold; a question should be refused if and only if it is no_answer. Each mode's threshold is chosen on the dev split (the highest F1, ties to the lower threshold) and measured on the test split (45 questions, 11 to refuse), with 95 % Wilson intervals; the configured threshold is shown for comparison. A chosen threshold is printed as the value to configure: the dev score it was chosen at, truncated so that every decision stays the same (the gate refuses strictly below it, so rounding up would refuse that question). Gate only: refusals by the model itself come from the answer run.

| mode | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| dense | 0.617442 (chosen on dev) | 8 | 5 | 3 | 29 | 0.615 [0.36, 0.82] | 0.727 [0.43, 0.90] | 0.667 |
| dense | 0.3 (configured) | 0 | 0 | 11 | 34 | n/a | 0.000 [0.00, 0.26] | 0.000 |
| hybrid | 0.963599 (chosen on dev) | 6 | 6 | 5 | 28 | 0.500 [0.25, 0.75] | 0.545 [0.28, 0.79] | 0.522 |
| hybrid | 0.3 (configured) | 4 | 0 | 7 | 34 | 1.000 [0.51, 1.00] | 0.364 [0.15, 0.65] | 0.533 |

### Refusals by no_answer kind

No_answer questions come in kinds: near-miss (the docs cover a neighbouring feature), knows-elsewhere (a fact a model may know from other sources that the pinned docs don't state) and out-of-scope (pricing, roadmaps, benchmarks). How many of each the gate refuses, as counts (each kind has only a few questions); the median is the kind's top score over both splits, and the dev rows are the ones the threshold was chosen on.

| mode | kind | median top score | dev refused (chosen) | test refused (chosen) | test refused (configured) |
|---|---|---|---|---|---|
| dense | near-miss | 0.575 | 2/2 | 3/4 | 0/4 |
| dense | knows-elsewhere | 0.575 | 2/2 | 2/4 | 0/4 |
| dense | out-of-scope | 0.589 | 2/3 | 3/3 | 0/3 |
| hybrid | near-miss | 0.985 | 0/2 | 2/4 | 1/4 |
| hybrid | knows-elsewhere | 0.949 | 2/2 | 2/4 | 1/4 |
| hybrid | out-of-scope | 0.228 | 3/3 | 2/3 | 2/3 |

![Refusal gate, dense: precision and recall against the threshold](refusal_dense.svg)
![Refusal gate, hybrid: precision and recall against the threshold](refusal_hybrid.svg)

## Answers

`POST /v1/ask` with `top_k` 5 in both modes, generated by `gpt-4o-mini` (openai). Judge: `anthropic:claude-haiku-4-5 prompts v1`.
An answerable question the system refused fails correctness without a judge call, and a no_answer question is correct if and only if it was refused; otherwise an answer passes at a rating of 4 or 5. Faithfulness is the share of an answer's claims that its retrieved contexts support, judged without the reference answer.

| mode | n | correct [95 % CI] | mean rating | faithfulness | fully supported | citations (self-check) | cost/query | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense | 75 | 0.693 [0.58, 0.79] | 4.256 | 0.910 | 0.814 [0.67, 0.90] | 0.936 | $0.00019 | 2,276 | 5,429 |
| hybrid | 75 | 0.667 [0.55, 0.76] | 4.233 | 0.963 | 0.884 [0.76, 0.95] | 0.987 | $0.00020 | 2,685 | 6,173 |

The mean rating is over the answers the judge rated (answerable and not refused). The judge rates a complete answer that adds correct detail beyond the reference 4 rather than 5, so the mean understates thorough answers; the pass rate is unaffected. Citations (self-check) is the pipeline's own verdict on its citations, n/a when it doesn't verify them. Faithfulness, fully supported and the citation self-check only cover the answers a mode gave, not its refusals, so a mode that refuses more of its harder questions can score higher on them: they compare how answers are grounded, not which mode does better end to end. Correctness is the end-to-end measure.

### Correct by category

| mode | category | n | correct [95 % CI] | mean rating | faithfulness |
|---|---|---|---|---|---|
| dense | ambiguous | 6 | 0.000 [0.00, 0.39] | 3.000 | 1.000 |
| dense | lookup | 37 | 0.649 [0.49, 0.78] | 4.481 | 0.910 |
| dense | multi_hop | 14 | 0.714 [0.45, 0.88] | 4.077 | 0.888 |
| dense | no_answer | 18 | 1.000 [0.82, 1.00] | n/a | n/a |
| hybrid | ambiguous | 6 | 0.000 [0.00, 0.39] | 2.800 | 1.000 |
| hybrid | lookup | 37 | 0.622 [0.46, 0.76] | 4.654 | 0.968 |
| hybrid | multi_hop | 14 | 0.643 [0.39, 0.84] | 3.917 | 0.938 |
| hybrid | no_answer | 18 | 1.000 [0.82, 1.00] | n/a | n/a |

### Refusals end to end (test split)

The gate and the model together: a refusal by either counts.

| mode | n | TP | FP | FN | TN | precision | recall | F1 | by gate | by model |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 45 | 11 | 6 | 0 | 28 | 0.647 [0.41, 0.83] | 1.000 [0.74, 1.00] | 0.786 | 13 | 4 |
| hybrid | 45 | 11 | 8 | 0 | 26 | 0.579 [0.36, 0.77] | 1.000 [0.74, 1.00] | 0.733 | 12 | 7 |

### No_answer questions by kind

No_answer questions come in kinds: near-miss (the docs cover a neighbouring feature), knows-elsewhere (a fact a model may know from other sources that the pinned docs don't state) and out-of-scope (pricing, roadmaps, benchmarks). Who declined each one: the gate (before generating), the model (after reading the contexts), or nobody (it was answered).

| mode | kind | split | n | refused by gate | refused by model | answered |
|---|---|---|---|---|---|---|
| dense | near-miss | dev | 2 | 2 | 0 | 0 |
| dense | near-miss | test | 4 | 3 | 1 | 0 |
| dense | knows-elsewhere | dev | 2 | 2 | 0 | 0 |
| dense | knows-elsewhere | test | 4 | 2 | 2 | 0 |
| dense | out-of-scope | dev | 3 | 2 | 1 | 0 |
| dense | out-of-scope | test | 3 | 3 | 0 | 0 |
| hybrid | near-miss | dev | 2 | 0 | 2 | 0 |
| hybrid | near-miss | test | 4 | 2 | 2 | 0 |
| hybrid | knows-elsewhere | dev | 2 | 2 | 0 | 0 |
| hybrid | knows-elsewhere | test | 4 | 2 | 2 | 0 |
| hybrid | out-of-scope | dev | 3 | 3 | 0 | 0 |
| hybrid | out-of-scope | test | 3 | 2 | 1 | 0 |

Spend in this run: RAG $0.0297, judge $0.5533 (a cached verdict costs nothing).

### Judge agreement with a human

No human grades for this run yet.
