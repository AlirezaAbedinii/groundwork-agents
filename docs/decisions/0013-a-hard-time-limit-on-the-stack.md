# 0013. A hard time limit on the stack instead of a spending cap

Date: 2026-11-08. Status: accepted.

## Context

The stack costs nothing within the free amounts and about $0.26 an hour with all three
apps active past them; Postgres alone would be about $17 a month if it ran all the time
outside its free offer. The plan was to end every session with `make down`, but a stack
can be forgotten: a session that stalls, a machine that sleeps. One did stay up for 41
hours, harmlessly only because Postgres was on its free meter. The cost guard had to stop
spending, not just report it. Azure offers no such cap: a budget only sends email, often
a day late; the Free Trial's spending limit stops only when the whole credit is gone; a
pay-as-you-go subscription has no spending limit to set. What can be capped is time,
because the stack's cost per hour is bounded (one replica per app at most).

## Decision

A GitHub workflow, `stack-guard.yml`, runs hourly from `main`. It signs in over OIDC
with the deploy identity and deletes `rg-groundwork` once its oldest resource is older
than `STACK_MAX_HOURS` (a repository variable, 8 by default). A deletion fails the run,
and so does any check it can't complete, because GitHub emails a failed run: the
failure is the alert. `make up` refuses to start unless the workflow is active. A manual
run with `max_hours=0` ends a stack at once.

## Consequences

- A forgotten stack ends without anyone's machine or session being up. Each stack's worst
  case is bounded: at list price with all three apps active the whole time, about $4.
- The identity's role lives on the group it guards, so while the stack is down the check
  is refused, and the guard reports "down" and passes. A refusal for any other reason,
  such as a wrong subscription id, fails the run instead of passing as "down".
- After the guard deletes a stack, `make down` still has work: Terraform's state lists
  the deleted resources, and the database secrets and role grants in the shared Key
  Vault outlive the group. In the cloud check the guard deleted a 109-minute-old stack
  with `max_hours=0`; the group was gone 25 minutes later, and `make down` removed the
  rest in 101 s.
- **GitHub's schedule is loose.** The first scheduled run came about ten hours after the
  workflow reached `main`, and later ones 4.5 to 7 hours apart, not hourly. The real
  bound is the limit plus GitHub's delay: about 15 hours at worst so far, still a stop.
  Schedules run only from the default branch, and GitHub turns them off after 60 days
  without repository activity, after which `make up` refuses.
- A stack whose first apply failed before granting the role would look "down" to the
  guard. The `main` credential skips the `azure` environment's reviewer, so any workflow
  on `main` could sign in as this identity (ADR 0011).

## Alternatives

- **Budget alerts alone.** They report, a day late, and stop nothing.
- **A runbook triggered by a budget.** A new service with broader rights, acting on cost
  data that arrives a day late.
- **A timer on the operator's machine.** It dies with the machine or the login, which is
  exactly when it's needed.
- **Scale to zero alone.** It stops the apps' compute but not the database's.
- **A schedule on another service.** It would add something outside the roadmap to do
  what GitHub, already holding the deploy identity, can do.
