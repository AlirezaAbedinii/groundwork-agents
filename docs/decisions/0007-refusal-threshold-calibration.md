# 0007. Refusal thresholds calibrated per retrieval mode on held-out questions

Date: 2026-10-06. Status: accepted.

## Context

The retrieval service refuses before generating when the top retrieved chunk's score
is below a threshold. One threshold, 0.30, served both retrieval modes, although
their scores mean different things: dense mode scores a chunk by cosine similarity,
hybrid mode by the sigmoid of a cross-encoder's logit. On the original demo corpus
that gap showed as 0 of 11 unanswerable questions refused in dense mode against 8 of
11 in hybrid mode. On the public docs corpus with `text-embedding-3-small`, 0.30
still refuses none of the 11 held-out unanswerable questions in dense mode, because
no top cosine similarity falls that low, and catches 4 of 11 in hybrid mode.

## Decision

Each mode has its own threshold. It is chosen on the golden set's dev split (30
questions, 7 of them unanswerable) as the one with the highest refusal F1, ties going
to the lower threshold (fewer refusals), and measured on the test split (45
questions, 11 unanswerable) with 95 % Wilson intervals. The rule was fixed before the
run, and the values are used as chosen, without adjusting them to the test results:

| mode | threshold | test precision | test recall | test F1 | at the old 0.30 |
|---|---|---|---|---|---|
| dense | 0.617442 | 0.615 [0.36, 0.82] | 0.727 [0.43, 0.90] | 0.667 | F1 0.000 (refuses nothing) |
| hybrid | 0.963599 | 0.500 [0.25, 0.75] | 0.545 [0.28, 0.79] | 0.522 | F1 0.533 (precision 1.000, recall 0.364) |

The values hold for `text-embedding-3-small` with the
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, and are the defaults in the
service's settings (truncated to six decimals, so the question each was chosen at
stays on the same side of it). The evaluation reports refusals by the gate alone, from
recorded searches, and end to end, where a refusal by the model after reading the
contexts counts too.

The same run settles the snippet size for MCP `search`: 5 hits of 150 characters
recall 8.3 points less evidence than 3 hits of 300 in hybrid mode on the test split
(the rule was to switch only for a gain of at least 5), so MCP keeps 3 × 300.

## Consequences

- Dense mode refuses at all now: on held-out questions it catches 8 of 11
  unanswerable ones, at the price of 5 false refusals among 34 answerable ones.
- In hybrid mode the calibrated threshold does no better than the old one on test (F1
  0.522 against 0.533) and trades precision for recall: it refuses 6 answerable
  questions where 0.30 refused none. The reranker gives 93 % of answerable questions a
  top score of 0.9 or more, and half of the unanswerable ones too (they share the
  corpus's vocabulary), so the threshold sits on a steep part of the curve, and with 7
  unanswerable dev questions one question moves F1 by several points. The intervals
  overlap widely; a larger dev split is the remedy, not a threshold picked on test.
- The thresholds are tied to the embedding model and the reranker and must be
  recalibrated when either changes. `/v1/config` publishes the thresholds and the
  served collection's embedding model, and every report records both, so a mismatch
  is visible. The keyless MiniLM embeddings used in development need their own.
- Each threshold equals the score of the lowest-scoring dev question the gate kept,
  so that question passes with no margin.

## Alternatives

- **One threshold for both modes.** The two scores aren't on a common scale; that is
  what produced 0 of 11 in dense mode.
- **Calibrating on the whole golden set.** It leaves no held-out questions, so the
  reported precision and recall would be the optimistic ones the threshold was fitted
  to.
- **The highest recall with precision at least 0.9.** With 7 unanswerable dev
  questions it means no false refusal at all until 10 are caught, so it collapses to
  "never refuse wrongly" and jumps on a single question.
- **Platt scaling or isotonic calibration of the scores.** A gate only needs the scores'
  order, which a monotone calibration doesn't change; it would matter if the scores
  were shown to users as probabilities.
- **Keeping 0.30 for hybrid because it did slightly better on test.** That would make
  the published test numbers a fit to the test split, the bias the split exists to
  prevent.
