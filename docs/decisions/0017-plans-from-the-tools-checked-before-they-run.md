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
  the same way. Elsewhere the default of 4 applies. Both demo plans also run code in the
  sandbox, which the Azure deployment leaves out by design (ADR 0009), so the demo runs on
  the compose stack only; `run_demo.py` reads `/config` and stops with the reason
  elsewhere.
- Research reads untrusted text and can now write files. Its writes stay inside the
  task's workspace, under the same path check as the other specialists' writes.
- With one subtask, the deliverable is the specialist's reviewed text in the subtask's
  expected format, and no second model rewrites it. With several, synthesis runs as
  before.
- The planning prompt is longer by one table row per specialist and tool.
- The lint and the one-subtask delivery are switched by the task's rules version
  (ADR 0018), so a run recorded before them replays the way it ran.

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

## Addendum (2026-11-28): one specialist's work is one subtask

### Context

The routing rule above, plan only when the request needs more than one specialist, was
prompt text: the planner's rule 2 says that if one specialist owns every tool the task
needs, the plan has one subtask for it. The lint checked ownership and size, not this,
and plans still split one specialist's work. One analysis subtask queried the database
and a second, also analysis, was to rank the rows with code. One specialist wrote code
and another that owns the same tool ran it. Research gathered documentation and a
writing subtask saved the file that research can write itself. Each split adds a
hand-off, a review and a synthesis call. A later subtask can also list a tool it does
not need, and the evidence check (ADR 0020) then sends it back for not calling that
tool, even when the answer is already in the earlier subtask's output.

### Decision

- **The lint refuses a split plan.** A plan of more than one subtask is an error when
  one specialist owns every tool its subtasks list. A subtask that lists no tools adds
  none, so a writing subtask that works from another's output is no reason to split.
  Rule 3 already allows a subtask only for work that needs another specialist's tools.
- **The message names the specialist.** It asks for exactly one subtask for that
  specialist, listing those tools, doing the whole task, and writing every deliverable
  when the plan has any. The specialist named is the first in the plan that owns every
  tool, or else the first in the registry that does. The message goes into the
  planner's single retry like any other lint error. A plan modified at the plan gate and
  one sent to the approvals API get the same check.
- **It stands aside while a research, analysis or code subtask lists no tools.** That is
  already an error, and a specialist named from an incomplete list of tools can be the
  wrong one: a plan whose analysis subtask has not yet listed its database tool would be
  told to give all the work to research.
- **Ownership comes from the registry**, as for the rest of the lint. A tool this
  deployment does not have belongs to no one, so a plan that lists it is never refused
  on this rule's account.
- **Switch.** Rules version 6 (`one_specialist_one_subtask`, ADR 0018). The planner's
  prompt is unchanged.

### Consequences

- A request that one specialist can carry out runs as one subtask. Its reviewed output
  is the answer, with no synthesis call, so the routing rule now holds in code.
- A writing subtask appears only in a plan that already needs two other specialists:
  writing owns only the file tools, and every specialist owns those.
- A single-specialist task with distinct steps runs them in one subtask's tool loop,
  within that subtask's attempts and turns (ADR 0015), not as separately reviewed steps.
- The rule reads the tools a plan lists, not what the request needs. A plan that lists
  too few tools can be refused when the request needs two specialists. The lint has no
  other source to check against.
- Checked against the recorded plans with the current registry, before the change.
  Eight plans list their subtasks' tools; the rule refuses six of them, each of which
  split one specialist's work, and names that specialist correctly. Twenty-six plans
  were recorded before subtasks listed tools. Each has a research, analysis or code
  subtask with no tools, so the rule stands aside for all of them, and they test
  nothing.

### Alternatives

- **Leave it to the prompt.** Rule 2 was already there, and plans kept splitting the
  work.
- **Merge the subtasks in code.** The lint would rewrite the plan: descriptions,
  dependencies and the deliverables' producers combined without the model, into a plan
  the planner never wrote. The retry already exists for this.
- **Count specialists instead of tools.** A plan that gives `code_exec` to code and then
  to analysis has two specialists and needs one. What decides is who owns the tools.
- **Run the check whatever the other errors.** Its message would sometimes name the
  wrong specialist, next to the error that asks for the missing tools.
