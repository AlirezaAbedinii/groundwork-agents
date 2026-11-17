# 0019. Required files are checked before delivery, and synthesis adds nothing

Date: 2026-11-17. Status: accepted.

## Context

A request that asks for a file names it in prose ("Save it as `notes.md`"). The plan
carried that name only inside a subtask's description, and nothing after the subtask
checked that the file existed. A task could complete with the file missing, saved under
another name, or only described in the final answer, and the run reported success.

Synthesis had the opposite problem. Its prompt passed the subtask outputs and asked for
the final deliverable, with nothing to stop the model adding its own material. A subtask
that found the documentation silent on a question could be followed by a final answer
that answered it anyway, with a value or a command no tool had returned.

## Decision

- **Deliverables in the plan.** `ExecutionPlan.deliverables` lists each file the task
  must deliver as `{path, producer}`: the path exactly as the request gives it, and the
  id of the subtask that writes it. Validation requires unique paths and a producer that
  is a subtask of the plan. The planner's prompt asks for every file the task asks to be
  saved or written.
- **The plan lint checks them.** Each producer's specialist must own `file_write`, and
  the producer must list it in its tools. Every file the request asks to save must be
  listed. The lint finds those with a conservative scan of the request
  (`requested_files`): a back-quoted name with a short extension, after "save", "saves",
  "saved" or "saving", either at once or after a short object ending in "as", "to", "in"
  or "into", within one sentence. A bare "as" does not count, so an input file mentioned
  as "such as `pyproject.toml`" is not taken for a deliverable. A test runs the scan on
  every request in the agent evaluation's task file: each yields exactly the file the
  task requires, or nothing.
- **A verify step after synthesis.** For each deliverable, the task's invocation log
  must hold a successful `file_write` whose `path` argument is exactly the deliverable's
  path, character for character, as the evaluation's own file check requires. Any
  subtask's write counts, and files written under other names do not matter. If one is
  missing, its producer runs once more with feedback naming the path ("Required file not
  written: `notes.md`. Call file_write with exactly that path …"), and the outputs are
  synthesized again. That attempt is granted the way an approval grants one (ADR 0015):
  the allowance rises by exactly one and the counters stay as they are. It happens only
  when the producer is below `MAX_SUBTASK_ATTEMPTS`, has not been sent back for a file
  before, and its output is not a human's take-over. Otherwise, or if the file is still
  missing after that attempt, the task ends `failed` with an error naming every missing
  file and why its producer was not run again. Nothing is delivered then.
- **Synthesis from a manifest.** The synthesis prompt lists each subtask with its
  specialist and description, the outcome of each tool call it made (read from the
  invocation log, with `file_write`'s path), and what it reported, followed by the
  deliverables, each marked written or not. Its rules are to use only those reports and
  add no fact, number, command, setting, example or recommendation; to say "the
  documentation does not cover this" wherever a subtask reports something as
  undocumented, and give no other answer to it; to name each deliverable by its exact
  path and call it written only if it is; and to say which part of the task the reports
  leave undone.
- **Switches.** The lint's deliverable checks and the verify step are rules version 3
  (`check_deliverables`, ADR 0018), so a task created earlier replays and resumes
  without them. `VERIFY_DELIVERABLES=false` turns the verify step off for a deployment
  without touching the lint.

## Consequences

- With the check on, no task whose plan lists a deliverable completes without a
  successful write at that exact path.
- The check covers only the files the plan lists. A planner can still leave out a file
  the request names without back-quotes or without "save"; the scan is a cross-check
  that catches the common phrasing, not a guarantee.
- The check is for presence, not content. Whether the file says the right things is
  still the reviewer's and the evaluation's to judge.
- A corrective attempt costs one specialist run, one review and one more synthesis
  call. It happens at most once per producer, inside the attempt cap and the per-task
  ceilings.
- Subtasks that depend on the producer are not run again; they already completed with
  its earlier output.
- A human who takes over a producer supplies text, not files. If that producer's file
  was not written by an earlier attempt, the task fails naming it.
- Grounding is by instruction: nothing compares the final text against the reports. The
  manifest does mark a missing file as not written, so the answer has no reason to
  claim it.
- A one-subtask plan still skips synthesis; the verify step runs after it the same way,
  and after a corrective attempt the new output is the deliverable.
- The verify step is a new span (`verify`, kind `verification`) with the deliverables
  and the missing paths.

## Alternatives

- **Check files only in the evaluation.** The evaluation would score the miss, but the
  user would still receive a completed task without the file.
- **Have the verify step write the file itself,** from the final answer. The file would
  then hold text no reviewer saw, and the verify step would become a producer.
- **Normalise paths,** so that `./notes.md` or `docs/notes.md` counts for `notes.md`.
  The request and the evaluation ask for the exact name, and one corrective attempt
  fixes the name cheaply.
- **Fail at once, without a corrective attempt.** A wrong name or a forgotten write is
  a common slip, and the feedback that fixes it is precise.
- **A broader scan:** any back-quoted file name, or one after a bare "as". It picks up
  input files, and a false positive makes the planner promise a file nobody asked for,
  which the verify step would then require.
- **A model check of the final answer against the reports.** It costs a call per task
  and is not deterministic; it is left out of this change.
