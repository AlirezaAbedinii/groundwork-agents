# eval

Cross-service evaluation lands here in Phase 3: retrieval metrics (Recall@k,
MRR@10, nDCG@10) over a real corpus in pgvector, answer correctness and
faithfulness scored by a separate judge model, refusal precision/recall with a
calibrated threshold, and rubric-based agent tasks scored on success, tool-call
correctness, cost and wall-clock. Until then the RAG service's own harness —
golden set, judge and `run_eval.py` — stays in [`services/rag/eval/`](../services/rag/eval/).
