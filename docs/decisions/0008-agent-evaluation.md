# 0008. Agent tasks scored on their deliverables and their invocation log

Date: 2026-10-15. Status: accepted.

## Context

The agent service had tests but no measurement of how well it does real work. Its
runs are long, nondeterministic and multi-step: a supervisor plans subtasks, four
specialists call tools, a reviewer scores each deliverable and can send it back, and
some steps pause for a human's approval. A task can go wrong in its outcome (the
deliverable is wrong) or in its trajectory (the right answer reached with the wrong
tools, or none at all), and the two need separate measures. Long-term memory is per
user and feeds planning, so one task's run can change the next one's plan. And the
reviewer's provider policy (another provider than the specialists, or the same one)
is a design choice no one had measured.

## Decision

`eval/agent_eval.py` runs each task through the agent API, one at a time, and scores
two things separately:

- **The trajectory, from the invocation log** (`GET /traces/{id}/tools`): each task
  names the calls it needs (`must_call`, with argument subsets such as a file's path,
  and alternatives such as "`rag_search` or `rag_ask`" where either consults the
  docs), the tools it may also use, and the ones it must not. Must-call recall counts
  only successful calls; precision counts calls to named tools whatever their
  status; the error rate counts the rest. Tools that reach outside services are
  never required.
- **The outcome, from a rubric judge that sees only the deliverables.** A task's
  criteria name what they are about (a file the task writes, or the final answer),
  and the judge is given the final answer and the files' contents. The subtasks'
  own reports are shown only for a criterion that names neither: in a live check,
  a judge that could read a research note stating a fact credited a file that left
  it out. A task succeeds only if it completed and every critical criterion passed.

Runs are made comparable and bounded: every task runs under a fresh user per run,
policy and task, so no memories cross; the harness approves every approval the agent
asks for and counts it, up to 5 per task; it stops waiting at 600 s; a per-task
reserve is checked against a mandatory spending cap. The reviewer policy is one
setting (`REVIEWER_PROVIDER=cross|same`) and one restart, with the same tasks, judge
and harness for both.

## Consequences

- Outcome and trajectory disagree usefully: in the first runs most tasks failed on
  the outcome for reasons the log explains (a lookup planned for a specialist
  without the database tool, research reading snippets instead of asking).
- Approving every escalation measures the agent without a human in the loop, but an
  approved subtask escalation means "retry the same specialist", which can't succeed
  when the specialist lacks the tool. The cap turns those loops into a recorded
  `approval_limit` outcome at bounded cost.
- A task that times out keeps running inside the agent (there is no cancel
  endpoint), so its spend after the timeout isn't recorded.
- Rubrics that make every requested part critical are strict: a near miss fails.
  Non-critical criteria are reported as a pass rate.
- The records hold each task's definition, so the agent report re-scores for $0
  without the run or the task file, like the retrieval one.
- With one run per policy and a few dozen tasks, success rates can't separate the
  policies; the reviewer's scores, rework and escalation rates, and cost can.

## Alternatives

- **Judging the trajectory with an LLM as well.** The log already says which tools
  ran and how; a judge would add cost and noise to a question that has an exact
  answer.
- **Showing the judge every output.** Simpler, and measured worse: the judge credited
  deliverables with facts only an intermediate note stated.
- **A human approving escalations.** Faithful to production, but it makes runs slow,
  unrepeatable and dependent on who approves.
- **An isolated reviewer benchmark with planted flaws.** A cleaner test of the
  reviewer alone, but it doesn't show what the reviewer does to whole tasks; left as
  a follow-up.
