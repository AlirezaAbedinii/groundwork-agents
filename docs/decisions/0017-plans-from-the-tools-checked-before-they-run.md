# 0017. Plans from the tools, checked before they run

Date: 2026-11-16. Status: accepted.

## Context

The planner's prompt described the four specialists in a phrase each and asked it to
decompose the task. It named no tool, so the planner assigned work by role name. A
database lookup could go to research, which has no database tool. A request to
summarise documentation could become a research step, an analysis step and a writing
step, each a hand-off.

A misassigned subtask does not fail. A specialist is offered only the tools it owns, so
a research subtask that needs `db_query` never sees it: the model answers from what it
already knows, the reviewer reads fluent text, and nothing in the run reports the
mismatch. Every extra step is also another specialist call, another review and another
hand-off where detail can be lost.

A request to turn documentation into a file always needed two specialists, because the
only specialist with the retrieval tools could not write files.

## Decision

- **A capability manifest.** `planning/capabilities.py` reads the live tool registry:
  for each specialist, each tool it owns, the first sentence of the tool's description,
  and its per-task call limit. The planner's prompt shows it as a table. A tool this
  deployment does not register (`code_exec` without a sandbox, `web_search` without a
  search provider, the `rag_*` tools without MCP) is not in it.
- **Rules in the prompt.** Use the fewest subtasks, never more than `max_plan_subtasks`
  (default 4). If one specialist owns every tool the task needs, plan one subtask. Add a
  subtask only for work that needs another specialist's tools, never one that only
  reviews, checks, finalises or analyses another subtask's output. List each subtask's
  tools and give it only to a specialist that owns them. Copy file names, input values
  and constraints from the request word for word.
- **`Subtask.tools`.** Each subtask names the tools it will call.
- **A plan lint.** `lint(plan, registry, max_subtasks)` is a pure function that reports:
  more subtasks than the limit; a tool this deployment does not have; a tool the
  subtask's specialist does not own; a research, analysis or code subtask with no tools.
  A writing subtask may list none and work from its inputs. The lint runs after the plan
  parses. Its errors go into the planner's existing single retry, the way schema errors
  already did, and a second failure fails the task with them. With `plan_lint=off` the
  prompt stays the same and the findings are recorded without rejecting anything. The
  plan span records `lint_errors` and `subtask_count`.
- **Research reads and writes workspace files.** `file_read` and `file_write` gain
  research as an owner, so documentation can become a file in one subtask.
- **The routing rule: plan only when the request needs more than one specialist.** A
  plan with one subtask is the normal case for a request one specialist can carry out.
  Its reviewed output is the final deliverable, and the synthesis call is skipped.

Each specialist is also shown the user's request verbatim above its subtask (a change
that landed just before this one), so a one-subtask plan loses nothing to the planner's
paraphrase.

## Consequences

- The manifest, the lint and the tools a specialist is offered all read the same
  registry, so they cannot disagree about who owns what.
- Fewer subtasks mean fewer specialist calls, reviews and hand-offs per task.
- The lint checks ownership and size, not intent. A subtask that lists a tool it owns
  but does not need passes, and the rule against review-only steps is in the prompt,
  not in code.
- The limit is a setting. The showcase scenario (`make demo`) plans five subtasks by
  design, three research steps in parallel, then analysis, then the memo, so the compose
  stack sets `MAX_PLAN_SUBTASKS=5`, and the end-to-end tests that play it raise the limit
  the same way. Elsewhere the default of 4 applies.
- Research reads untrusted text and can now write files. Its writes stay inside the
  task's workspace, under the same path check as the other specialists' writes.
- With one subtask, the deliverable is the specialist's reviewed text in the subtask's
  expected format, and no second model rewrites it. With several, synthesis runs as
  before.
- The planning prompt is longer by one table row per specialist and tool.

## Alternatives

- **Ask a model to check the plan.** It costs a call and is not deterministic.
  Ownership is a lookup.
- **No planner: route each request to one specialist.** Tasks that need tools owned by
  different specialists, such as a database query and a web search, still need a small
  plan. The planner stays, and the rules make one subtask its default.
- **Give every specialist every tool.** Misassignment would disappear, but so would the
  ownership boundary that keeps `api_call` and `code_exec` with the roles meant to use
  them.
- **Keep file writing out of research.** That keeps the hand-off from research to
  writing for every documentation task that ends in a file.
- **Reject a failing plan without a retry.** A lint error names the subtask and the
  field to change, as a schema error does, and the retry already exists for those.
