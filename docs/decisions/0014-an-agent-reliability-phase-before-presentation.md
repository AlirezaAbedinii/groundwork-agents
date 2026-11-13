# 0014. An agent reliability phase before presentation

Date: 2026-11-12. Status: accepted.

## Context

The plan was to go from the Azure deployment straight to presentation. The first agent
evaluation (ADR 0008) and a review of the code showed that the agent's control policy,
not its model or its infrastructure, decides how well it does a task, and that the
policy has gaps a presentation should not be built on:

- **Plans.** The planner's prompt describes the specialists in a line each and names no
  tool, so it cannot know which specialist can query the database or write a file. It
  is asked to decompose, not to use the fewest steps that do the work.
- **What a specialist sees.** A specialist is given its subtask's description and the
  outputs of the subtasks it depends on, never the user's request.
- **Tool contracts.** Tool descriptions do not state the schema a query runs against,
  what a successful result looks like, or the per-task call limit.
- **Retries.** A failed call can be repeated unchanged, and an approval that sends a
  subtask back starts its error and rework counters again, so nothing in the agent
  bounds the number of attempts.
- **Review.** The reviewer is given the subtask's description, the expected format and
  the output text. It sees no tool result and no file, so it cannot tell a grounded
  answer from a fluent one.

The retrieval side has open questions of the same kind: the hybrid mode's re-ranker,
fusion settings and chunking have not been compared arm by arm on this corpus, and the
golden set is too
small to resolve differences of a few points between configurations.

## Decision

A phase, 4b, goes between the Azure deployment and presentation. It hardens the agent's
control policy, tunes hybrid retrieval by measured ablation, and then measures both
again. It adds no service and no dependency.

The work is ordered so that nothing is paid for before it can be checked for free:

1. **Diagnostics.** The evaluation harness records what is needed to say, for any run,
   where a task went wrong: plan shape, tool use per subtask, repeated and refused calls.
2. **Termination.** A fixed number of attempts per subtask that approvals do not reset,
   and ceilings per task on model calls and spend.
3. **Information.** Tool descriptions that state schema, outcome and limits; tool
   results cut at record boundaries; repeated identical calls answered without running
   them; the user's request in every specialist's prompt.
4. **Plans and review.** A planner that knows which specialist holds which tool and
   plans only when a request needs more than one, with a check that rejects a plan no
   specialist can carry out; a reviewer that sees what the tools returned.
5. **Retrieval.** A larger golden set, written without running retrieval and checked by
   hand; an ablation over re-ranker, fusion and chunking, chosen on the dev split.
6. **Measurement.** A small paid pilot, changes driven by its records, then the full
   run. Tasks held out from that iteration are reported separately.

Steps 1 to 4 run on mock models and recorded data. Every paid run has a spending cap set
before it starts and needs its own approval. Each change that alters a design decision
gets its own ADR when it is made; this one records only that the phase exists, why, and
in what order it proceeds.

## Consequences

- Presentation moves back by a few weeks of part-time work.
- Phases 1 to 3 are re-opened in part: the specialist loop, the tool registry's
  descriptions, the planner, the retrieval defaults and the evaluation harness change.
  Later measurements must stay comparable with earlier ones: same task texts, rubrics
  and judge; scoring rules carry a version.
- Changes made after the pilot are driven by the tasks' own records, which invites
  fitting the agent to its test set. Held-out tasks, and a rule that no prompt or tool
  description may name a task's content, are the guard.
- A change of chunking or re-ranker means re-indexing the collection and calibrating
  the refusal thresholds again (ADR 0007).

## Alternatives

- **Go straight to presentation.** The least work, and it would present a design with
  known gaps in how the agent is controlled.
- **A stronger model first.** The gaps are in what the model is shown and what happens
  after a failed call. A stronger model would still plan without knowing the tools, at a
  higher price per task. Model choice is measured after the policy is fixed, not before.
- **More agents, or another agent framework.** None of the gaps belongs to the
  framework, and more hand-offs lose more context.
- **Raise the rate limits.** That hides repeated calls instead of stopping them, with
  no bound on spend.
- **Fix and re-run everything at once.** One large change cannot say which fix mattered.
