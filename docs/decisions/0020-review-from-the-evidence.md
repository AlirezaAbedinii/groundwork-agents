# 0020. Review from the evidence, and no review when the tools returned nothing

Date: 2026-11-17. Status: accepted.

## Context

The reviewer scored a subtask from its description, its expected format and its output.
It saw neither the user's request nor what the subtask's tools had returned. An output
that offered estimates after every database query had failed read as a competent answer
and could be approved, and the next subtask could state the estimates as figures. A
tool-bound subtask that answered without calling a tool, with a value the documentation
does not give, could be confirmed by a reviewer drawing on its own knowledge, even when
the request had limited the sources to the documentation.

## Decision

- **Receipts.** The specialist loop leaves one receipt for each call that ran or that a
  human ruled on: `{tool, status, digest}`, with status `success`, `error` or `denied`
  and, as digest, the first 200 characters of the answer on one line. A repeat of an
  earlier call leaves none, since nothing ran. `SpecialistResult.receipts` returns them.
  Before the cut, credential-shaped text in the answer (named secrets such as
  `API_KEY=…`, passwords in URLs, bearer tokens, common provider key formats) is
  replaced by `[redacted]`, since the reviewer may run on another provider than the one
  that called the tool.
- **No result, no review.** Before the reviewer, if the subtask lists tools and none of
  its receipts for those tools is a success or a human's denial, the attempt goes back as
  `rework` with no reviewer call. The feedback is fixed text: "None of your required
  tools returned a result in this attempt (db_query). Errors (3): …", quoting the last
  three errors of those tools, or saying that none of them was called, and asking the
  specialist to say plainly what could not be retrieved rather than estimate it. The
  entry has no score and records the tools as `no_tool_result`. It counts as a rework:
  after `MAX_SPECIALIST_RETRIES` of them it escalates under `low_review_score`, with a
  reason that names the tools instead of a score.
- **Evidence in the review.** The reviewer's prompt adds the user's request, the tools
  the subtask lists, the ids of the inputs it was given (not their text), the receipts,
  and two rules. A claim with no receipt behind it is unsupported, unless the request or
  the inputs could have supplied it, and an output that states unsupported numbers,
  names or facts as findings scores below the threshold. When the request restricts the
  sources, facts from the reviewer's own knowledge do not count. The instructions and
  the rules come as a system message, and the system message says that text in the
  receipts or the output that reads as an instruction is content to judge, not an
  instruction. The rest stays in the user message, where each receipt is one line, so
  a tool's text cannot start a line of its own. The threshold and `REVIEW_MARKER` are
  unchanged.
- **Switches.** The check before review is rules version 4 (`require_tool_results`,
  ADR 0018), so a task created earlier replays and resumes without it.
  `REVIEW_EVIDENCE=false` turns off both the check and the evidence in the prompt; the
  prompt is then the earlier one, character for character.

## Consequences

- A subtask whose listed tools all failed is not approved. It is reworked with its
  errors and, if they persist, escalated to a human. Subtasks that depend on it do not
  run on its output meanwhile.
- A subtask that lists tools but answers without calling any is treated the same way.
  A writing subtask that works from its inputs lists no tools and is exempt. One that
  lists `file_write` and never writes is sent back here, before the verify step
  (ADR 0019) would find the file missing. A subtask that verify sent back for a missing
  file is exempt from then on: its corrective attempt is reviewed as before, so a file
  still missing after it fails the task, as ADR 0019 says, rather than escalating for
  more attempts.
- A silent attempt costs no review call. Every other review prompt grows by the
  request, the receipts and the rules: a few hundred tokens.
- A human's denial counts as an answer. Trying again would put the same question to the
  same human, so the reviewer judges what the specialist made of the denial.
- The reviewer does not see the inputs' text, so a claim carried in from an input is
  taken on trust. The check before review is what keeps an input built on failed tools
  from completing in the first place.
- One successful call of a listed tool satisfies the check, even if it returned nothing
  useful: an empty result, or a script that printed a constant. Judging those is left to
  the reviewer, with the receipt in view.
- Whether the reviewer applies its rules is up to the model. The tests fix the prompt,
  not the verdict; only the check before review is deterministic.
- The redaction is a list of patterns, not a guarantee, and the system message lowers
  the risk of injected instructions without removing it. The specialist's output, which
  the reviewer saw before this change, can still repeat what a tool returned. Checks for
  injection and personal data on every provider call are the guardrails planned for
  v1.1.

## Alternatives

- **Show the reviewer the full tool results and the inputs' text.** It could check every
  claim, but the prompt would grow with every call and every input, and review is
  already a large share of a task's model spend.
- **A digest shaped for each tool** (paths and row counts, no content). It would show
  the reviewer nothing to check a claim against, which is the point of the receipts.
- **Receipts in the prompt and no check before it.** It leaves to a model the one case
  that needs no judgement: nothing came back.
- **Require every listed tool to succeed.** A plan may list a tool the subtask turns out
  not to need; one result from the tools it named shows the attempt worked from them.
- **Count a success of any tool, listed or not.** The plan names the tools that hold
  what the subtask needs; a result from another does not supply it.
- **Fail the attempt instead of reworking it.** A wrong table name or a syntax error is
  often fixed on the next attempt once the error is quoted back.
- **Record a score of 1 for a silent attempt.** The escalation would need no change, but
  the record and the trace would show a review that never happened.
