# eval

Cross-service evaluation lands here in Phase 3: retrieval metrics (Recall@k,
MRR@10, nDCG@10) over a real corpus in pgvector, answer correctness and
faithfulness scored by a separate judge model, refusal precision/recall with a
calibrated threshold, and rubric-based agent tasks scored on success, tool-call
correctness, cost and wall-clock. It replaces the retrieval service's own
harness, which has been retired
([last version](https://github.com/AlirezaAbedinii/groundwork-agents/tree/940677e9078e4cd4d8a70aecb74f26c8295435d0/services/rag/eval)).

It is a small project of its own, like [`mcp/`](../mcp/README.md): it talks to
both services over HTTP only, so it measures what actually runs.

## So far

| Path | What it is |
|---|---|
| `schemas.py` | Golden questions, agent tasks, judge verdicts and metric results |
| `textnorm.py` | How a quote is matched against chunk text, for labels and scoring alike |
| `golden/` | The golden set and its [schema and rules](golden/SCHEMA.md) |
| `dataset/validate.py` | Checks the golden set against the schema, the corpus files and the served chunks |
| `llm.py`, `budget.py` | Structured completions from OpenAI or Anthropic, priced, under a spending cap |
| `dataset/synthesize.py` | Drafts synthetic golden candidates, one chunk each (paid; `--max-cost-usd` required) |
| `dataset/review.py` | Accepts, edits or rejects those drafts by hand |
| `tasks/` | Agent evaluation tasks |

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
.venv/bin/python -m pytest -q -m "not live"
.venv/bin/python -m dataset.validate golden/golden_set.jsonl \
  --corpus ../services/rag/data/raw/toolchain_docs --rag http://localhost:8000
```
