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

## Addendum, 2026-11-13: scoring rules version 2

The scoring rules are now versioned. A run stamps each record with `scoring: 2`; a
record without the field is scored by version 1, the rules above, so a published
report re-scores exactly as it was published. `agent_eval score --scoring 2` scores
older records under the new rules for comparison, without a judge call. Version 2
adds three rules:

- **A criterion about a named file fails when the run never wrote that file at its
  exact path**, with the reasoning "the file was not written", and is not sent to
  the judge; when every criterion is decided this way, no judge call is made. This
  is stricter. A criterion that begins with a file's name is about that deliverable,
  and a judge shown the final answer or a file under another name can still credit
  it. Checking the invocation log for a successful `file_write` at that path is
  exact and free, so the judge is left only the questions that need reading.
- **A `code_exec` run that exits non-zero is a tool error.** This is stricter. The
  registry calls a code run successful whenever the sandbox ran it, so a script that
  raised counted as a success in the error rate. A call with no recorded exit code
  keeps the status rule.
- **A task's cost is recorded once its LLM-call count holds still**: equal on two
  polls 2 s apart, or after 10 s. This is more complete. The agent marks a task
  completed before it extracts memories from it, so costs fetched at that moment
  left out the memory step's call. The poll is a heuristic: a memory call slower
  than the poll interval can still be missed, and the 10 s bound keeps a busy agent
  from holding the run.

Version 2 changes no rubric and no judge prompt, and task success still requires
every critical criterion to pass.

## Addendum, 2026-11-15: held-out tasks

Ten tasks, t19 to t28, are marked `held_out: true`: 4 docs, 2 data, 1 code and 3
combined, the proportions of those categories among the first 18. They exist to test
changes made while iterating on the other tasks. A change that only fits the tasks it
was tuned on shows up as a gap between the two sets, and it can only show up if
nobody looked at the held-out tasks' runs while making it.

- **How they were written.** From the corpus files and the demo database's seed, never
  by running the agent or a model. Each fact was checked in the corpus file it comes
  from, and the data answers against the seed applied to PostgreSQL 16. The docs
  facts come from corpus pages the first 18 tasks don't use, and no request repeats
  their wording or file names. The demo schema has one table, so the data tasks query
  the same table as before, with other questions: the catalog's description of the
  table, and figures relative to each row instead of to the total.
- **Rubrics.** Three critical criteria per task, each a fact that can be checked in
  the deliverable; everything else asked for is non-critical and reported as a pass
  rate. Fewer critical parts make a near miss less likely to fail the whole task.
- **The run refuses them.** `agent_eval run` leaves held-out tasks out and says so,
  unless `--include-held-out` is given; naming one in `--ids` without the flag is an
  error, not a warning. A refusal fits because a held-out task run once during
  iteration is no longer held out, and nothing can undo that.
- **Records keep the flag.** It is part of each task's definition in the records, so
  a report can show the held-out tasks apart from the rest.

The first 18 tasks, their rubrics and the scoring rules are unchanged.

## Addendum, 2026-11-15: the never-run tasks are held out as well

t13 to t15, t17 and t18 are now marked `held_out: true` too. No published run includes
them, and they were already to be kept out of the runs that change the agent, but only
by each such run naming its tasks with `--ids`: one plain run would have used them up.
With the flag, a plain `agent_eval run` covers exactly the 13 tasks of the published
run, and the held-out set is 15 tasks: these five and t19 to t28, which keep their own
ids as the ten written to be held out. Only the flag changed; their requests and
rubrics are as before.

## Addendum, 2026-11-21: the agent's ceiling fits in the reserve

`--max-cost-usd` is checked against a
reserve (`--task-reserve-usd`, default $0.25) before each task starts, but nothing held
the task to the reserve: the agent's own per-task ceiling (`MAX_TASK_COST_USD`, ADR
0015) defaults to $0.50, so the last task admitted could take spending past the cap by
the difference. The agent's `/config` now reports `max_task_llm_calls` and
`max_task_cost_usd`, and on real models `agent_eval run` refuses a label whose
`max_task_cost_usd` is 0 (no ceiling) or above the reserve, before it records the
label. A task that times out is charged the larger of its recorded cost and the
reserve: there is no cancel endpoint, so it may go on spending up to its ceiling after
the harness moves on. Its record keeps the cost that was measured.

The run's limits, and which of them are hard:

- **The judge: hard.** Each judge call is checked at its worst case, the input estimate
  plus the whole output allowance, before it is made.
- **The agent: its per-task ceiling, at most the reserve. Not quite hard.** The agent
  checks a task's spend before each model call, so a call already in flight when the
  ceiling is reached can pass it by its own cost, and parallel subtasks can have
  several in flight. A task that ends is charged what it spent, so an overshoot counts
  before the next task's reserve is checked. Spending after a timeout is not measured;
  the reserve stands in for it.
- **`rag_ask`: not capped.** It spends in the RAG service, which has no per-task limit;
  the agent allows it 5 calls per task. Each call's `cost_usd` is added from the
  invocation log when the task ends.

So a run can pass `--max-cost-usd` only by calls in flight when a task reached its
ceiling and by `rag_ask` calls, those of the last task and of any task that timed out.

Since `/config` gained fields, a label begun before this change cannot be resumed: its
recorded config no longer matches.
