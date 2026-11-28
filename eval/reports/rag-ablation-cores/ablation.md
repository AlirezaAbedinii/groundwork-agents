# Retrieval ablation

Run `2026-11-20-ablation-cores`. Each arm searched the answerable questions of the golden set (197 rows, sha256 `7be5db05284c`) with `top_k` 10. The metrics are the RAG report's: evidence items found in the hits, each counted once at its first rank. Differences are taken per question to A0, over the questions both arms searched, with 95 % paired bootstrap percentile intervals (10,000 resamples, seed 0). Latency is the service's own timing of the search (query embedding, dense, BM25, fusion, re-rank), so it leaves out HTTP and model loading; the re-rank columns are that stage alone.

## Arms

| arm | change | mode | collection (chunking, size/overlap) | BM25 | re-ranker | pool | RRF weights |
|---|---|---|---|---|---|---|---|
| A0 | dense, as published | dense | toolchain_docs (fixed, 800/120) | – | – | – | – |
| A4a-2c | A4a on 2 pinned cores | hybrid | toolchain_docs (fixed, 800/120) | on | `Alibaba-NLP/gte-reranker-modernbert-base` | 20 | 0.7/0.3 |
| A4a-4c | A4a on 4 pinned cores | hybrid | toolchain_docs (fixed, 800/120) | on | `Alibaba-NLP/gte-reranker-modernbert-base` | 20 | 0.7/0.3 |
| A6-2c | A6 on 2 pinned cores | hybrid | toolchain_docs_headings (recursive+heading_path, 800/0) | on | `Alibaba-NLP/gte-reranker-modernbert-base` | 50 | 0.7/0.3 |
| A6-4c | A6 on 4 pinned cores | hybrid | toolchain_docs_headings (recursive+heading_path, 800/0) | on | `Alibaba-NLP/gte-reranker-modernbert-base` | 50 | 0.7/0.3 |

## dev: 63 answerable questions

| arm | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | search ms p50 | p95 | re-rank ms p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 63 | 0.433 | 0.617 | 0.683 | 0.743 | 0.642 | 0.647 | 199 | 640 | – | – |
| A4a-2c | 63 | 0.489 | 0.703 | 0.781 | 0.818 | 0.700 | 0.715 | 8570 | 10360 | 8312 | 9851 |
| A4a-4c | 63 | 0.489 | 0.703 | 0.781 | 0.818 | 0.700 | 0.715 | 4660 | 5550 | 4391 | 5204 |
| A6-2c | 63 | 0.532 | 0.725 | 0.793 | 0.870 | 0.767 | 0.768 | 16925 | 23527 | 16379 | 23200 |
| A6-4c | 63 | 0.532 | 0.725 | 0.793 | 0.870 | 0.767 | 0.768 | 9723 | 14893 | 9376 | 14528 |

Difference to A0, arm − A0 per question:

| arm | pairs | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| A4a-2c | 63 | +0.056 [-0.060, +0.168] | +0.086 [+0.004, +0.173] | +0.098 [+0.037, +0.164] | +0.075 [+0.015, +0.139] | +0.058 [-0.034, +0.150] | +0.068 [+0.001, +0.136] |
| A4a-4c | 63 | +0.056 [-0.060, +0.168] | +0.086 [+0.004, +0.173] | +0.098 [+0.037, +0.164] | +0.075 [+0.015, +0.139] | +0.058 [-0.034, +0.150] | +0.068 [+0.001, +0.136] |
| A6-2c | 63 | +0.099 [+0.003, +0.192] | +0.108 [+0.028, +0.198] | +0.110 [+0.032, +0.196] | +0.127 [+0.053, +0.206] | +0.126 [+0.042, +0.214] | +0.121 [+0.059, +0.190] |
| A6-4c | 63 | +0.099 [+0.003, +0.192] | +0.108 [+0.028, +0.198] | +0.110 [+0.032, +0.196] | +0.127 [+0.053, +0.206] | +0.126 [+0.042, +0.214] | +0.121 [+0.059, +0.190] |

nDCG@10, arm − A0, per stratum (questions in brackets):

| arm | all (63) | category ambiguous (6) | category lookup (39) | category multi_hop (18) | phrasing exact_token (18) | phrasing paraphrase (18) | phrasing unlabelled (27) |
|---|---|---|---|---|---|---|---|
| A4a-2c | +0.068 [+0.001, +0.136] | -0.016 [-0.279, +0.248] | +0.065 [-0.023, +0.159] | +0.101 [+0.013, +0.199] | +0.021 [-0.062, +0.097] | +0.095 [-0.034, +0.224] | +0.081 [-0.029, +0.202] |
| A4a-4c | +0.068 [+0.001, +0.136] | -0.016 [-0.279, +0.248] | +0.065 [-0.023, +0.159] | +0.101 [+0.013, +0.199] | +0.021 [-0.062, +0.097] | +0.095 [-0.034, +0.224] | +0.081 [-0.029, +0.202] |
| A6-2c | +0.121 [+0.059, +0.190] | +0.084 [-0.119, +0.306] | +0.110 [+0.030, +0.199] | +0.157 [+0.054, +0.267] | +0.071 [-0.023, +0.170] | +0.091 [-0.014, +0.203] | +0.175 [+0.069, +0.290] |
| A6-4c | +0.121 [+0.059, +0.190] | +0.084 [-0.119, +0.306] | +0.110 [+0.030, +0.199] | +0.157 [+0.054, +0.267] | +0.071 [-0.023, +0.170] | +0.091 [-0.014, +0.203] | +0.175 [+0.069, +0.290] |

R@5, arm − A0, per stratum (questions in brackets):

| arm | all (63) | category ambiguous (6) | category lookup (39) | category multi_hop (18) | phrasing exact_token (18) | phrasing paraphrase (18) | phrasing unlabelled (27) |
|---|---|---|---|---|---|---|---|
| A4a-2c | +0.098 [+0.037, +0.164] | +0.056 [-0.194, +0.306] | +0.085 [+0.009, +0.179] | +0.139 [+0.046, +0.241] | +0.065 [+0.000, +0.139] | +0.046 [+0.000, +0.120] | +0.154 [+0.031, +0.290] |
| A4a-4c | +0.098 [+0.037, +0.164] | +0.056 [-0.194, +0.306] | +0.085 [+0.009, +0.179] | +0.139 [+0.046, +0.241] | +0.065 [+0.000, +0.139] | +0.046 [+0.000, +0.120] | +0.154 [+0.031, +0.290] |
| A6-2c | +0.110 [+0.032, +0.196] | +0.000 [-0.250, +0.250] | +0.128 [+0.026, +0.239] | +0.106 [-0.037, +0.250] | +0.028 [-0.083, +0.139] | +0.014 [-0.065, +0.102] | +0.228 [+0.074, +0.389] |
| A6-4c | +0.110 [+0.032, +0.196] | +0.000 [-0.250, +0.250] | +0.128 [+0.026, +0.239] | +0.106 [-0.037, +0.250] | +0.028 [-0.083, +0.139] | +0.014 [-0.065, +0.102] | +0.228 [+0.074, +0.389] |
