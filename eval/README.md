# eval

Cross-service evaluation lands here in Phase 3: retrieval metrics (Recall@k,
MRR@10, nDCG@10) over a real corpus in pgvector, answer correctness and
faithfulness scored by a separate judge model, refusal precision/recall with a
calibrated threshold, and rubric-based agent tasks scored on success, tool-call
correctness, cost and wall-clock. It replaces the retrieval service's own
harness, which has been retired
([last version](https://github.com/AlirezaAbedinii/groundwork-agents/tree/940677e9078e4cd4d8a70aecb74f26c8295435d0/services/rag/eval)).
