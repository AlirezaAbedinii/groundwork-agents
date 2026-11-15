# 0015. An approval buys one attempt, and each subtask has a cap

Date: 2026-11-15. Status: accepted.

## Context

A subtask escalates to a human after two failed attempts, or when its rework cycles
(`MAX_SPECIALIST_RETRIES`) end with a review score still below the threshold. Approving
that escalation reset both counters to zero, so each approval bought two or three more
attempts, and nothing bounded how many times one subtask could run. Each result recorded
its attempt count, but nothing read it. With `APPROVAL_LEVEL_OVERRIDES` mapping a
subtask trigger to `notify`, the same reset happened with no human asked, and the run
ended only at LangGraph's recursion limit. The evaluation harness approves every
escalation up to a limit, so a subtask that could not succeed kept spending model calls
until that limit stopped the task. The phase that this change belongs to (ADR 0014)
makes termination its first requirement, and a cost guard here has to stop spending,
not report it (ADR 0013).

## Decision

An approval raises an allowance and never resets a counter. Each subtask result carries
`attempt_allowance`, at first `MAX_SPECIALIST_RETRIES + 1` (3). A subtask escalates on
the existing conditions, or when it has not completed and has made `attempt_allowance`
attempts. An approve, a modify or a notify sets the allowance to `attempts + 1`: exactly
one more attempt.

`MAX_SUBTASK_ATTEMPTS` (default 5) caps one subtask's attempts. At the cap an approve, a
modify or a notify no longer grants an attempt: the task ends `failed` with "Attempt
budget exhausted for subtask <sid> after <n> attempts". Take-over and reject work as
they do below the cap.

## Consequences

- No subtask runs more than `MAX_SUBTASK_ATTEMPTS` times under any sequence of
  decisions, and an unattended run (every subtask trigger at `notify`) ends.
- The counters stay as they were, so after a grant the next attempt that does not
  complete escalates at once. The allowance covers the mixed case where neither counter
  has reached its limit, such as two weak drafts and then one error.
- A reviewer who wants a sixth attempt cannot have one; they can take over with their
  own output, reject, or raise the setting. The approval at the cap proposes `fail_task`
  and its reasoning says what approving will do.
- A subtask that has made every allowed attempt is held out of the next wave until its
  own escalation is decided. `gather` raises one escalation at a time, so when two
  subtasks reach their limits after the same wave, the second one waits instead of
  running on, which without the hold could take it past the cap. If nothing else can
  run, `schedule` goes back to `gather` instead of synthesizing with a subtask missing.
- A second subtask that reached a counter limit in that same wave but still has
  attempts left in its allowance uses them before its own escalation, as it did before.
  Its allowance bounds it.
- The first allowance is never larger than the cap, whatever `MAX_SPECIALIST_RETRIES`
  is set to.
- Tasks paused before this change have no allowance on their results. The default
  stands in, so they resume under the new rules.
- The escalation triggers are unchanged: an exhausted allowance raises
  `specialist_double_failure` after a failed attempt and `low_review_score` after a
  weak one, so level overrides written for those triggers still apply.
- Rollback is by revert only. The old behaviour is removed, not kept behind a setting.

## Alternatives

- **Keep the reset and cap approvals per task.** It ends the loop, but each approval
  would still buy two or three attempts, and the subtasks of one task would share one
  limit, so one hopeless subtask could use up the approvals of all the others.
- **A cap without an allowance.** Each approval would still buy a fresh set of attempts
  until the cap. With the allowance, "approve" always means one more attempt, which a
  reviewer can predict.
- **A sixth trigger for a spent allowance.** It would need its own approval level, and
  overrides written for the two subtask triggers would not cover it.
- **Let an escalation change the specialist or the subtask.** That is a larger change to
  what a human can decide at the gate. It is not needed to make a task terminate, so it
  waits for evidence that it would help.
