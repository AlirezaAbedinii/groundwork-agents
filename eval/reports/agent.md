# Agent evaluation

Run `2026-10-14-agents`. Each task runs once per label through the agent API, under a user of its own so that no task's memories reach another's plan. Tool calls are scored from the agent's invocation log; the outcome comes from a rubric judge that sees only the outputs: the final answer, the files the run wrote and, when a criterion asks about the work as a whole, what each subtask reported.

## Provenance

|  | cross | same |
|---|---|---|
| collected | 2026-10-14 | 2026-10-15 |
| supervisor | `openai:gpt-4o` | `openai:gpt-4o` |
| specialists | `openai:gpt-4o-mini` | `openai:gpt-4o-mini` |
| reviewer, as routed | `anthropic:claude-sonnet-5` | `openai:gpt-4o` |
| reviewer policy | `cross` | `same` |
| memory | `openai:gpt-4o-mini` | `openai:gpt-4o-mini` |
| MOCK_LLM | no | no |
| rag_* tools | yes | yes |
| tasks | 13 from `agent_tasks.jsonl` (sha256 `d461d3e1cd0e`) | 13 from `agent_tasks.jsonl` (sha256 `d461d3e1cd0e`) |
| task judge | `anthropic:claude-haiku-4-5 prompts v2` | `anthropic:claude-haiku-4-5 prompts v2` |
| spend: agent, judge | $1.05864, $0.03473 | $0.43891, $0.05003 |

## Results

|  | cross | same |
|---|---|---|
| task success | 0.077 [0.01, 0.33] (1/13) | 0.077 [0.01, 0.33] (1/13) |
| completed | 7/13 | 11/13 |
| tool-call must-call recall | 0.795 | 0.744 |
| tool-call precision | 0.652 | 0.685 |
| tool-call error rate | 0.233 | 0.200 |
| tool calls per task | 22.846 | 15.923 |
| forbidden calls | 0 (0 tasks) | 0 (0 tasks) |
| cost per task | $0.08143 | $0.03376 |
| cost per completed task | $0.07792 | $0.03283 |
| wall-clock per task | 142.8 s | 74.4 s |
| wall-clock per completed task | 92.4 s | 47.6 s |
| escalations | 40 (8 tasks) | 13 (3 tasks) |
| approvals given | 34 | 11 |
| reviewer mean score | 3.30 (53 subtasks) | 4.36 (67 subtasks) |
| rework rate | 0.250 (68 subtasks) | 0.070 (71 subtasks) |
| non-critical criteria passed | 0.500 [0.09, 0.91] (1/2) | 0.750 [0.30, 0.95] (3/4) |

Task success is over every task run, with a 95 % Wilson interval: a task that didn't complete fails without a judge call, and one that did succeeds when every critical criterion passed. Tool-call ratios are means over tasks, each task counting once: must-call recall is the share of required calls that a successful call made; precision the share of calls to tools the task names; the error rate the share that failed, were rejected or were rate limited. Cost is the agent's LLM spend plus what `rag_ask` spent in the RAG service; the judge's spend is not in it. "Per completed task" means are over the tasks that completed. Escalations are the approvals the agent asked for; the harness approves each one (at most 5 per task). The reviewer's mean score is over the subtasks it scored, and the rework rate is the share of subtasks run more than once.

## Cross vs same reviewer

On the 13 tasks judged under both labels: 0 succeeded under both, 11 under neither; only with the cross-provider reviewer: t10; only with the same-provider reviewer: t06.

## Task success by category

| category | cross | same |
|---|---|---|
| code | 0.500 [0.09, 0.91] (1/2) | 0.000 [0.00, 0.66] (0/2) |
| data | 0.000 [0.00, 0.56] (0/3) | 0.000 [0.00, 0.56] (0/3) |
| docs | 0.000 [0.00, 0.39] (0/6) | 0.167 [0.03, 0.56] (1/6) |
| multi | 0.000 [0.00, 0.79] (0/1) | 0.000 [0.00, 0.79] (0/1) |
| no_answer | 0.000 [0.00, 0.79] (0/1) | 0.000 [0.00, 0.79] (0/1) |

## Per task: cross

| task | category | outcome | success | must-call recall | precision | error rate | forbidden | cost | wall-clock | escalations | agent task |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t01 | docs | completed | fail | 1.000 | 1.000 | 0.000 | 0 | $0.03725 | 66.9 s | 0 | `166c0f3941f2412ab2b43b9c92493032` |
| t02 | docs | approval_limit | fail | 1.000 | 0.636 | 0.000 | 0 | $0.17462 | 287.5 s | 6 | `fe4770a4510a4f73920d544d9f5260b0` |
| t03 | docs | completed | fail | 1.000 | 0.909 | 0.000 | 0 | $0.11938 | 189.8 s | 3 | `1d97a298c2954361837a92e94189e464` |
| t04 | docs | completed | fail | 1.000 | 0.600 | 0.000 | 0 | $0.13238 | 99.1 s | 0 | `5dad7501e3024f5a9ba2680f8dbf3b26` |
| t05 | docs | completed | fail | 1.000 | 0.800 | 0.000 | 0 | $0.06141 | 70.1 s | 0 | `1bbf456033ad4d4782a0a83244e0bca1` |
| t06 | docs | completed | fail | 1.000 | 0.769 | 0.000 | 0 | $0.08290 | 107.2 s | 1 | `10149cc4838e42e3899f859a6352f79a` |
| t07 | data | approval_limit | fail | 0.000 | 0.000 | 0.475 | 0 | $0.04753 | 131.9 s | 6 | `82f88eed6993439cbe1c207d2a1dd1ab` |
| t08 | data | approval_limit | fail | 0.000 | 0.000 | 0.439 | 0 | $0.05535 | 115.7 s | 6 | `a6f3ae2652874fa881d3497e9d61175f` |
| t09 | data | approval_limit | fail | 1.000 | 0.807 | 0.754 | 0 | $0.05573 | 137.8 s | 6 | `913302e74a394c7a9f01a1f1736b84e6` |
| t10 | code | completed | pass | 1.000 | 0.333 | 0.000 | 0 | $0.07493 | 80.4 s | 0 | `91ca77fdca64469cbf0990780db8845f` |
| t11 | code | approval_limit | fail | 1.000 | 0.968 | 0.887 | 0 | $0.12201 | 433.0 s | 6 | `3b0c058e6770440ba0b175aa4e7ee8bf` |
| t12 | multi | approval_limit | fail | 0.333 | 0.659 | 0.477 | 0 | $0.05795 | 103.6 s | 6 | `0d4ee02967774468acafaca35c011389` |
| t16 | no_answer | completed | fail | 1.000 | 1.000 | 0.000 | 0 | $0.03720 | 33.6 s | 0 | `3c0f8f8852784d56968c55decddfbbbd` |

## Per task: same

| task | category | outcome | success | must-call recall | precision | error rate | forbidden | cost | wall-clock | escalations | agent task |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t01 | docs | completed | fail | 1.000 | 1.000 | 0.000 | 0 | $0.02732 | 53.5 s | 0 | `9fd38b78861c4b3a86987e75e7950bc9` |
| t02 | docs | completed | fail | 1.000 | 0.667 | 0.000 | 0 | $0.03464 | 45.3 s | 0 | `0f93c3ec222c43b0a29c8042e9b98fac` |
| t03 | docs | completed | fail | 1.000 | 0.667 | 0.000 | 0 | $0.04158 | 61.9 s | 0 | `e54343b1cbf14b52a52e0fe052ce2dee` |
| t04 | docs | completed | fail | 0.500 | 0.750 | 0.000 | 0 | $0.04994 | 34.6 s | 0 | `a89c605705a14fe7b0a9afe39b0615c2` |
| t05 | docs | completed | fail | 0.500 | 0.333 | 0.000 | 0 | $0.03194 | 48.3 s | 0 | `708804790ca74b08afa0d10c3042a7b3` |
| t06 | docs | completed | pass | 1.000 | 0.273 | 0.000 | 0 | $0.05961 | 105.9 s | 0 | `1816f0150a5148c6a77f27cfc17f7081` |
| t07 | data | completed | fail | 0.000 | 0.250 | 0.250 | 0 | $0.01988 | 36.2 s | 0 | `0c96c2226d504f2cb48d6369c9ea1776` |
| t08 | data | completed | fail | 0.000 | 0.714 | 0.357 | 0 | $0.03088 | 49.6 s | 0 | `21707d2947884a3aad91933655eb6af1` |
| t09 | data | completed | fail | 1.000 | 0.750 | 0.250 | 0 | $0.02119 | 30.0 s | 0 | `c60e664091ee48bcb6ed761f4a97cfbe` |
| t10 | code | approval_limit | fail | 1.000 | 0.938 | 0.859 | 0 | $0.02421 | 127.8 s | 6 | `c58d5676a0554b31b787cffa02183746` |
| t11 | code | approval_limit | fail | 1.000 | 0.968 | 0.887 | 0 | $0.05357 | 315.8 s | 6 | `ad0d37ba3e464cabb2f3319d929a6d60` |
| t12 | multi | completed | fail | 0.667 | 0.600 | 0.000 | 0 | $0.02316 | 35.6 s | 1 | `1b8990b8cb664a82a8d20d1f3d590642` |
| t16 | no_answer | completed | fail | 1.000 | 1.000 | 0.000 | 0 | $0.02100 | 23.1 s | 0 | `896d26112cfe4e23adbf07f710e1e6b3` |
