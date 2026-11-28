# 0018. A rules version on every task

Date: 2026-11-16. Status: accepted.

## Context

A strict replay re-runs a past task through the current graph, serving the model's
recorded responses in place of new calls. A resumed task continues from its checkpoint
under the current code. ADR 0017 changed which model calls a run makes, and what it
delivers, without changing anything that was recorded:

- The plan lint can send a plan back for a retry. A plan recorded before the lint has no
  `tools`, so it fails the lint, and the retry it asks for is not in the recording. The
  replay fails.
- A one-subtask plan's reviewed output became the deliverable, with no synthesis call. A
  recording from before that change has a synthesis call, which the replay skipped. The
  replay returned the specialist's text where the original delivered the synthesis, and
  nothing reported the difference.

A task paused before the change and resumed after it would meet the same rules halfway
through its run.

## Decision

- **Every task records a rules version.** It is stored in `tasks.rules_version`, added by
  migration 0009. A new task gets the current version, 2. NULL, which is every task
  created before the column existed, means version 1.
- **One function maps a version to a small frozen set of switches.** That function is
  `rules_for(version)` in `orchestrator/rules.py`. Graph nodes and the approvals API read
  the switches and never compare version numbers; a test checks that no other module
  does.
- **Version 1 switches both rules off:**
  - `lint_plans` is off: no plan lint, neither on the planner's plan nor on a plan a
    human modified at the plan gate.
  - `deliver_single_output` is off: every plan's outputs go through synthesis.
- **Version 2 switches both on:**
  - `lint_plans`: the lint checks the planner's plan and a modified plan, and
    `PLAN_LINT` decides whether its findings reject the plan.
  - `deliver_single_output`: a one-subtask plan's reviewed output is the deliverable,
    with no synthesis call.
- **Nothing else is switched.**
- **Where a run gets its version:**
  - A replay or a fork runs under its original's version.
  - A resumed run takes the version from its graph state. A checkpoint from before
    versions has none, so it is version 1.
- **The version is on the task.** `GET /tasks/{id}` returns `rules_version`. The agent
  evaluation's run records keep that bundle, so each run's version stays with its record.
- **A strict replay must use every recorded call.** A strict replay that finishes with a
  recorded call it never served fails with `ReplayDivergenceError`, naming the first
  unused call. This leaves out calls to the memory agent, which a replay never makes
  because it does not use long-term memory, and a replay paused for an approval, which
  has not finished.
- **Limits from earlier packets apply to every replay, whatever its version.** These are
  the attempt allowance and its hard cap, and the per-task ceilings on model calls and
  spend (all ADR 0015). The same holds for every setting, which a replay reads from the
  deployment it runs in.
- **When a later packet bumps the version.** Only when it changes which model calls
  happen or what is delivered. It does not bump the version for a change to prompt text,
  because a replay serves a call whose prompt has drifted by its order among that
  agent's calls.

## Consequences

- A recording from before ADR 0017 replays the way it ran: no lint, and synthesis for
  every plan.
- When a replay finishes without reproducing its original, it fails and names the call
  where the divergence shows. It no longer returns a different output silently. A
  replay that one of today's limits ends sooner than its original ran fails with that
  limit's error, or pauses at the escalation the limit raises.
- The approvals API refuses a modified plan that fails the lint with a 422, and the
  approval stays pending. A modified plan that reaches the graph another way ends the
  run, with the lint errors as its error.
- Each version's behaviour stays in the code for as long as old tasks are replayed.
  Adding a version means adding one entry to `rules_for` and reading the new switch
  where the behaviour changes.
- A version that this build does not know, for example from a newer deployment sharing
  the database, is refused before a run, a resume or a replay starts. A resume reads the
  version from the checkpoint, not the task row. The task fails with the reason, and
  nothing runs under the wrong rules.

## Alternatives

- **Infer the rules from the saved plan,** where a subtask without a `tools` key marks
  an old plan. That marker covers only this change, so every later change would need a
  marker of its own.
- **Record the rules on the trace,** with replay reading the original's spans. That
  makes replay depend on the span export having succeeded.
- **Compare version numbers where behaviour differs.** The checks would spread through
  the graph, and each later version would have to find and extend them. Named switches
  keep the meaning of a version in one place.
- **Leave old replays broken and say so.** Replay is meant to reproduce any past run,
  and the failure was silent in one of the two cases.

## Addendum (2026-11-17): version 3

Version 3 adds one switch, `check_deliverables`, and keeps both of version 2's on. With
it, the plan lint checks the plan's deliverables, and a verify step after synthesis
checks that each was written (ADR 0019). A new task gets version 3; a task from version
2 replays and resumes without either check.

## Addendum (2026-11-17): version 4

Version 4 adds one switch, `require_tool_results`, and keeps version 3's on. With it, a
subtask that lists tools, none of which returned a result in an attempt, goes back for
rework before any review (ADR 0020). A new task gets version 4; a task from version 3
replays and resumes without the check. The dispatch payload now carries the task's rules
version, so the execute node can read the switch; a payload from before that reads as
version 1, which has the switch off, as every version before 4 does.

## Addendum (2026-11-26): version 5

Version 5 adds one switch, `only_requested_files`, and keeps version 4's on. With it,
the plan lint refuses a deliverable whose path the request does not name. It also
refuses `file_write` in a plan that has no deliverable the request names (ADR 0019,
addendum of 2026-11-26). Both checks apply to the planner's plan, to a plan modified at
the plan gate, and in the approvals API. A new task gets version 5. A task from version
4 replays and resumes without the checks. The planner's rule on deliverables and the
synthesis rule for a plan without files are prompt text, so they need no version.

## Addendum (2026-11-28): version 6

Version 6 adds one switch, `one_specialist_one_subtask`, and keeps version 5's on. With
it, the plan lint refuses a plan of more than one subtask when one specialist owns every
tool its subtasks list (ADR 0017, addendum of 2026-11-28). The check applies to the
planner's plan, to a plan modified at the plan gate, and in the approvals API. A new
task gets version 6. A task from version 5 replays and resumes without the check. The
planner's rule 2, which asks for the same plan, is prompt text and is unchanged.
