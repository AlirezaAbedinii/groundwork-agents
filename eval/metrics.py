"""The evaluation's metrics: pure functions over what the harnesses recorded.

Stdlib only (plus the contract types in ``schemas``), no I/O. The harnesses turn raw
runs into the arguments below and the reports print the results, so a metric can be
fixed and every run re-scored for free.

Conventions every function follows:

- **The unit of relevance is the evidence item**, not the chunk. A ranked hit is the
  set of items it covers; an item covered by several hits counts once, at its first
  rank, and ``n_items`` is the number of items, never the number of relevant chunks.
- **Undefined is None, not 0.0**: a ratio with a zero denominator is None, which the
  reports print as "n/a" and leave out of means.
- **Bad input raises ValueError** instead of producing a number that looks valid.
- **Fail closed**: a criterion the judge's verdict leaves out has failed.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence

from schemas import (
    PRF,
    AgentTask,
    CorrectnessVerdict,
    CriterionVerdict,
    ExpectedCall,
    FaithfulnessVerdict,
    FalseRefusals,
    McNemar,
    PairedDifference,
    PlannedSubtask,
    Share,
    SweepPoint,
    TaskVerdict,
    ToolCall,
    ToolCallScores,
)

# One entry per ranked hit, best first: the evidence items (indexes into the question's
# evidence list) that the hit covers.
Ranked = Sequence[frozenset[int]]


# --- retrieval ----------------------------------------------------------------------


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _check_items(ranked: Ranked, n_items: int) -> None:
    if n_items <= 0:
        raise ValueError(f"n_items must be >= 1, got {n_items} (no_answer rows have no ranking)")
    for rank, hit in enumerate(ranked, start=1):
        if bad := sorted(i for i in hit if not 0 <= i < n_items):
            raise ValueError(f"hit {rank} covers items {bad}, outside range({n_items})")


def _first_ranks(ranked: Ranked, k: int) -> list[int]:
    """The 1-based rank at which each distinct item is first covered in the top k, in order.

    One entry per item, so an item seen again lower down adds nothing, and a hit that
    covers two new items adds its rank twice.
    """
    seen: set[int] = set()
    ranks: list[int] = []
    for rank, hit in enumerate(ranked[:k], start=1):
        new = hit - seen
        ranks.extend([rank] * len(new))
        seen |= new
    return ranks


def _discount(rank: int) -> float:
    return 1 / math.log2(rank + 1)


def recall_at_k(ranked: Ranked, n_items: int, k: int) -> float:
    """The share of the question's evidence items covered anywhere in the top k.

    Rewards finding everything the answer needs and ignores order within the top k,
    so it measures what the generator is given, not how well it's ranked. Duplicates
    don't help: two chunks holding the same item count as one item found.
    """
    _check_items(ranked, n_items)
    _check_k(k)
    return len(_first_ranks(ranked, k)) / n_items


def mrr_at_k(ranked: Ranked, k: int = 10) -> float:
    """1 / rank of the first hit that covers any evidence item, or 0.0 if none is in the top k.

    Rewards putting something relevant at the very top and nothing else: it can't
    tell whether the other items were found, so on multi-item questions it is a
    "first useful hit" score, not a completeness score.
    """
    _check_k(k)
    for rank, hit in enumerate(ranked[:k], start=1):
        if hit:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: Ranked, n_items: int, k: int = 10) -> float:
    """DCG of the items found in the top k over the DCG of the best possible ranking.

    Rewards finding every item and finding it early: each item is worth
    1 / log2(rank + 1), so a late item still counts, but less. The j-th distinct item
    found is credited at rank max(r_j, j), where r_j is the rank of the hit that first
    covered it: a hit covering two new items is worth what two hits at the top would
    be, never more, so the score stays in [0, 1]. The ideal puts one new item at each
    of ranks 1..min(n_items, k), and at most that many items are credited.
    """
    _check_items(ranked, n_items)
    _check_k(k)
    creditable = min(n_items, k)
    found = _first_ranks(ranked, k)[:creditable]
    dcg = sum(_discount(max(rank, j)) for j, rank in enumerate(found, start=1))
    ideal = sum(_discount(j) for j in range(1, creditable + 1))
    return dcg / ideal


# --- refusals -----------------------------------------------------------------------
# The positive class is "refused": a question should be refused iff it is no_answer.


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def refusal_prf(refused: Sequence[bool], should_refuse: Sequence[bool]) -> PRF:
    """Precision, recall and F1 of refusing, with the confusion counts.

    Precision rewards refusing only what has no answer (a false refusal costs a user
    an answer the docs hold); recall rewards refusing everything that has none (a
    missed refusal is where hallucination starts). F1 = 2tp / (2tp + fp + fn) weighs
    the two mistakes equally. Each is None when its denominator is 0.
    """
    if len(refused) != len(should_refuse):
        raise ValueError(f"{len(refused)} predictions for {len(should_refuse)} labels")
    pairs = list(zip(refused, should_refuse, strict=True))
    tp = sum(r and s for r, s in pairs)
    fp = sum(r and not s for r, s in pairs)
    fn = sum(s and not r for r, s in pairs)
    tn = len(pairs) - tp - fp - fn
    return PRF(
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def refusal_sweep(scores: Sequence[float], should_refuse: Sequence[bool]) -> list[SweepPoint]:
    """The refusal gate's PRF at every threshold that changes its decisions, ascending.

    The gate refuses iff score < threshold, as the RAG service's does. Thresholds are
    the distinct scores and +inf: the lowest refuses nothing, +inf refuses everything,
    and any threshold between two adjacent scores decides exactly like the higher one.
    Raising the threshold only adds refusals, so recall never falls along the sweep.
    """
    if len(scores) != len(should_refuse):
        raise ValueError(f"{len(scores)} scores for {len(should_refuse)} labels")
    if not all(math.isfinite(s) for s in scores):
        # NaN compares False with everything, and +inf would survive the +inf threshold.
        raise ValueError("scores must be finite numbers")
    thresholds = sorted(set(scores) | {math.inf})
    return [SweepPoint(t, refusal_prf([s < t for s in scores], should_refuse)) for t in thresholds]


def choose_threshold(points: Sequence[SweepPoint]) -> float:
    """The swept threshold with the highest refusal F1; ties go to the lower threshold.

    Rule: maximize F1 on the dev split, so false refusals and missed refusals cost the
    same, then among equal F1s refuse less (a lower threshold refuses a subset). The
    result is one of the swept thresholds, i.e. the score of the lowest-scoring
    question the gate keeps. Raises ValueError when the labels have nothing to refuse:
    F1 is then None or 0.0 everywhere, and the "best" pick would be arbitrary.
    """
    # tp + fn is the number of questions that should be refused, the same at every point.
    if not points or any(p.prf.tp + p.prf.fn == 0 for p in points):
        raise ValueError("the labels have nothing to refuse, so no threshold is better")
    # With something to refuse, 2tp + fp + fn > 0, so every F1 is a number.
    return max(points, key=lambda p: (p.prf.f1, -p.threshold)).threshold


# The two refusers of the RAG service, in the order a question meets them.
REFUSERS = ("gate", "model")


def false_refusals(refused_by: Sequence[str | None]) -> FalseRefusals:
    """Of answerable questions, the share refused, in all and by each refuser.

    One entry per answerable question: None when it was answered, "gate" when the score
    gate refused it before generation, "model" when the model refused after reading the
    contexts. Each of these refusals withholds an answer the docs hold. Refusal F1 counts
    them only on the split it is measured on and adds them to the missed refusals; this
    counts them alone, on every answerable question. The split by refuser matters because
    each is tuned on its own: the gate by its threshold, the model by its prompt.
    """
    if bad := sorted({str(r) for r in refused_by if r is not None and r not in REFUSERS}):
        raise ValueError(f"unknown refusers {bad}; expected None or one of {REFUSERS}")
    n = len(refused_by)
    return FalseRefusals(
        refused=Share(k=sum(r is not None for r in refused_by), n=n),
        gate=Share(k=sum(r == "gate" for r in refused_by), n=n),
        model=Share(k=sum(r == "model" for r in refused_by), n=n),
    )


def per_stratum[R](strata: Sequence[str], fn: Callable[..., R], *columns: Sequence) -> dict[str, R]:
    """``fn`` of each stratum's rows, by stratum in sorted order.

    Row i of every column belongs to stratum ``strata[i]``, and ``fn`` gets the columns
    cut down to one stratum's rows, in their order, as positional arguments: so
    ``per_stratum(categories, mcnemar_exact, dense, hybrid)`` keeps each question's two
    results paired. Strata with no rows don't appear.
    """
    if any(len(c) != len(strata) for c in columns):
        raise ValueError(f"columns of {[len(c) for c in columns]} rows for {len(strata)} strata")
    out: dict[str, R] = {}
    for stratum in sorted(set(strata)):
        rows = [i for i, s in enumerate(strata) if s == stratum]
        out[stratum] = fn(*([c[i] for i in rows] for c in columns))
    return out


# --- uncertainty and agreement ------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a proportion (95 % at the default z).

    Unlike p ± z·sqrt(p(1-p)/n), it stays inside [0, 1] and isn't zero-width at 0/n
    or n/n, which is where small eval sets put many results. With no data (n = 0) it
    is (0.0, 1.0): every proportion is possible.
    """
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes} of {n}")
    if z <= 0:
        raise ValueError(f"z must be > 0, got {z}")
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # The interval is inside [0, 1] exactly; clamping only removes rounding at 0/n and n/n.
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Agreement between two raters beyond what their pass rates give by chance.

    kappa = (p_o - p_e) / (1 - p_e): p_o is the share of items they agree on, p_e the
    share two independent raters with the same pass rates would agree on. 1 is perfect
    agreement, 0 is chance level, below 0 is worse than chance. Convention: when
    p_e = 1 (both raters constant and equal) kappa is 0/0 and the result is None, not
    1.0: two raters who pass everything haven't shown they can tell anything apart.
    """
    if len(a) != len(b) or not a:
        raise ValueError(f"need two equally long, non-empty ratings, got {len(a)} and {len(b)}")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    if p_e == 1:  # exact: only reachable with pa = pb in {0, 1}
        return None
    return (p_o - p_e) / (1 - p_e)


# --- paired comparisons -------------------------------------------------------------
# Two configurations run on the same questions are compared question by question: a hard
# question lowers both results and drops out of their difference, so a paired test can
# see a gap that two independent intervals, each as wide as the questions vary, hide.

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """The exact McNemar test: do a and b pass at different rates on the same items?

    Items both pass or both fail say nothing about which is better, so only the
    discordant ones count: ``only_a`` passed in a alone, ``only_b`` in b alone. If the
    two were equally good, each of the d = only_a + only_b discordant items would go
    either way with probability 1/2, so the two-sided p-value is twice the smaller
    binomial tail, p = min(1, 2 · Σ_{i ≤ min(only_a, only_b)} C(d, i) / 2^d), and 1.0
    when nothing is discordant. Exact, because the chi-square form needs more
    discordant items than a golden set of this size gives.
    """
    if len(a) != len(b) or not a:
        raise ValueError(f"need two equally long, non-empty results, got {len(a)} and {len(b)}")
    only_a = sum(bool(x) and not y for x, y in zip(a, b, strict=True))
    only_b = sum(bool(y) and not x for x, y in zip(a, b, strict=True))
    d = only_a + only_b
    tail = sum(math.comb(d, i) for i in range(min(only_a, only_b) + 1))
    return McNemar(only_a=only_a, only_b=only_b, n=len(a), p_value=min(1.0, 2 * tail / 2**d))


def _nearest_rank(ordered: Sequence[float], p: float) -> float:
    # The smallest value with at least a share p of them at or below it. Rounded before
    # the ceiling, so that 0.975 · 10,000 picks the 9,750th, not the 9,751st.
    return ordered[max(1, math.ceil(round(p * len(ordered), 9))) - 1]


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedDifference:
    """The mean of b - a over paired items, with a paired bootstrap percentile interval.

    Each resample draws n items with replacement, each keeping its two scores together,
    and takes the mean difference. The interval is the nearest-rank (1 - c)/2 and
    (1 + c)/2 percentiles of the resampled means: at 10,000 resamples and 95 %, the
    250th and 9,750th smallest. The percentile interval assumes nothing about the
    differences' distribution, which for R@k and MRR on a few dozen questions is a few
    spikes, not a bell. The seed is fixed so the interval can be recomputed from the
    records; it is a constant, not a choice made after seeing results. ``difference``
    is the observed mean, not the mean of the resamples.
    """
    if len(a) != len(b) or not a:
        raise ValueError(f"need two equally long, non-empty scores, got {len(a)} and {len(b)}")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    if not all(math.isfinite(d) for d in diffs):
        raise ValueError("scores must be finite numbers")
    n, rng = len(diffs), random.Random(seed)
    means = sorted(math.fsum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    return PairedDifference(
        difference=math.fsum(diffs) / n,
        low=_nearest_rank(means, tail),
        high=_nearest_rank(means, 1 - tail),
        n=n,
        resamples=resamples,
        seed=seed,
    )


# --- answers ------------------------------------------------------------------------

# Correctness is judged on 1-5; an answer passes at this rating or above.
CORRECTNESS_PASS_RATING = 4


def correctness_pass(verdict: CorrectnessVerdict) -> bool:
    """True for a rating of 4 or 5: correct in substance, at most minor omissions.

    A 3 (partly right) fails, since a user acting on it would miss something the
    reference answer says. The same cut binarizes the human grades, so the judge-human
    kappa measures agreement on this pass/fail decision.
    """
    return verdict.rating >= CORRECTNESS_PASS_RATING


def faithfulness_score(verdict: FaithfulnessVerdict) -> float | None:
    """The share of the answer's claims that the retrieved contexts support.

    Rewards saying only what the contexts state, regardless of whether it's correct.
    None for an answer with no claims (a refusal, say), which has nothing to be
    faithful about: counting it as 1.0 would let refusals inflate the mean.
    """
    return _ratio(sum(c.supported for c in verdict.claims), len(verdict.claims))


# --- agents -------------------------------------------------------------------------
# Tool calls come from the agent's invocation log (the trajectory); task verdicts come
# from the rubric judge, which sees only the outputs (the outcome).
#
# The scoring rules are versioned, so a record is always scored by the rules it was
# collected under. Version 1 (records without a "scoring" field) is what the first
# published run used. Version 2 adds two rules: a code_exec run that exits non-zero is a
# tool error, and a criterion about a file the run never wrote fails without a judge.

# The reasoning recorded for a criterion decided by the file rule instead of the judge.
UNWRITTEN_FILE = "the file was not written"


def _same(expected: object, actual: object) -> bool:
    # Equal and of the same type: "3" isn't 3, 3.0 isn't 3, and True isn't 1 (a bool is
    # an int to ==), so a task's args mean exactly what they say.
    return type(expected) is type(actual) and expected == actual


def _meets(call: ToolCall, entry: ExpectedCall) -> bool:
    """A successful call to one of the entry's tools, with every one of its args."""
    return (
        call.status == "success"
        and call.tool_name in entry.tools
        and all(k in call.arguments and _same(v, call.arguments[k]) for k, v in entry.args.items())
    )


def _errored(call: ToolCall, exit_codes: bool) -> bool:
    if call.status != "success":
        return True
    # A code run that raised or exited non-zero did not do its job, though the sandbox ran.
    return exit_codes and call.tool_name == "code_exec" and call.exit_code not in (None, 0)


def tool_call_scores(
    calls: Sequence[ToolCall], task: AgentTask, *, exit_codes: bool = False
) -> ToolCallScores:
    """How the run used its tools, scored from the invocation log.

    - must_call_recall: the share of the task's ``must_call`` entries that some
      successful call met: a call to the entry's tool (or one of its alternatives)
      whose arguments contain the entry's ``args``. A failed, rejected or rate-limited
      call meets nothing: the work it stands for didn't happen. None when the task
      requires no call.
    - precision: the share of calls that went to a tool the task names, in
      ``must_call`` (every alternative counts) or ``may_call``, whatever their status.
      It rewards choosing the right tools and penalizes detours; how the calls went is
      the error rate's business.
    - error_rate: the share of calls that didn't succeed (failure, rejected or
      rate_limited). With ``exit_codes`` (scoring version 2), a ``code_exec`` call that
      ran but exited non-zero is an error too; a call without a recorded exit code
      keeps the status rule.
    - forbidden_calls: calls to a forbidden tool at any status, since attempting one is
      the mistake.

    Precision and the error rate are None for a run that made no calls.
    """
    named = {tool for entry in task.must_call for tool in entry.tools} | set(task.may_call)
    met = sum(any(_meets(call, entry) for call in calls) for entry in task.must_call)
    return ToolCallScores(
        must_call_recall=_ratio(met, len(task.must_call)),
        precision=_ratio(sum(c.tool_name in named for c in calls), len(calls)),
        error_rate=_ratio(sum(_errored(c, exit_codes) for c in calls), len(calls)),
        forbidden_calls=sum(c.tool_name in task.forbidden for c in calls),
        n_calls=len(calls),
    )


def criteria_passed(verdict: TaskVerdict, task: AgentTask) -> dict[str, bool]:
    """Whether each of the task's rubric criteria passed, by id, failing closed.

    A criterion the verdict leaves out failed, and one it lists more than once passed
    only if every entry for it did. Entries for ids that aren't in the rubric are
    ignored. The report's per-criterion pass rates come from here too, so they follow
    the same rule as ``task_success``.
    """
    entries: dict[str, list[bool]] = {}
    for c in verdict.criteria:
        entries.setdefault(c.id, []).append(c.passed)
    return {c.id: bool(entries.get(c.id)) and all(entries[c.id]) for c in task.rubric}


def unwritten_file_criteria(
    subjects: Mapping[str, str], calls: Sequence[ToolCall]
) -> list[CriterionVerdict]:
    """A failed verdict for each criterion whose file the run never wrote (version 2).

    ``subjects`` maps each criterion that is about a file to that file's path (the
    judge's ``file_subjects``). A file counts as written only if a ``file_write``
    succeeded at exactly that path: what is under another name, or nowhere, can't meet a
    criterion about the named file, and a judge shown the final answer or another file
    can be talked into crediting it. Deciding these without the judge makes the rule
    exact and costs nothing.
    """
    written = {
        c.arguments.get("path")
        for c in calls
        if c.tool_name == "file_write" and c.status == "success"
    }
    return [
        CriterionVerdict(id=cid, reasoning=UNWRITTEN_FILE, passed=False)
        for cid, path in subjects.items()
        if path not in written
    ]


def overrule(verdict: TaskVerdict, decided: Sequence[CriterionVerdict]) -> TaskVerdict:
    """The verdict with every entry for a decided criterion replaced by its decision."""
    ids = {d.id for d in decided}
    return TaskVerdict(criteria=[c for c in verdict.criteria if c.id not in ids] + list(decided))


def task_success(verdict: TaskVerdict, task: AgentTask) -> bool:
    """True iff every critical criterion passed: the task's outcome, judged from its outputs.

    Non-critical criteria never fail a task; the report shows their pass rate instead. A
    critical criterion missing from the verdict counts as failed, so a judge that skips
    one can't pass the task. Raises ValueError for a task without a critical criterion:
    "every critical criterion passed" would then be true of any run, so it couldn't fail.
    """
    critical = [c.id for c in task.rubric if c.critical]
    if not critical:
        raise ValueError(f"task {task.id} has no critical criterion, so it can't be scored")
    passed = criteria_passed(verdict, task)
    return all(passed[cid] for cid in critical)


# --- agent diagnostics --------------------------------------------------------------
# Counts read from the records alone, never from a model. They don't score a run; they
# say where in it the work went missing, so a change upstream of the outcome (a plan, a
# tool result, a retry) can be checked without a paid run. Each returns k of n as a
# Share, or None when the record predates the field it needs: "not recorded" is never
# reported as 0.

# What the specialist's tool loop shows the model of one result: json.dumps(output)[:2000].
RESULT_BUDGET_CHARS = 2000


def plan_size(plan: Sequence[PlannedSubtask] | None) -> int | None:
    """The number of subtasks in the supervisor's plan; None when the record has no plan."""
    return None if plan is None else len(plan)


def starts_with_research(plan: Sequence[PlannedSubtask] | None) -> bool | None:
    """Whether the plan's first subtask goes to the research specialist.

    A plan that opens with research for a task that needs no documents (a query, a
    calculation) spends its first subtask on work the request didn't ask for. An empty
    plan doesn't start with research; None when the record has no plan.
    """
    if plan is None:
        return None
    return bool(plan) and plan[0].specialist == "research"


def toolless_completed(subtasks: Sequence[Mapping], calls: Sequence[ToolCall]) -> Share | None:
    """Of the subtasks that completed, those with no tool call at any status.

    ``subtasks`` are the run's subtasks with their ``sid`` and final ``status``. Such a
    subtask finished on the model's text alone. That is right for some (a writing step
    working from its inputs) and wrong for any that had to look something up, so the
    share is a pointer to read the subtasks, not a count of failures. A rate-limited or
    failed call still counts as a call: the specialist tried. None when a call has no
    ``sid`` (an older record), because the calls can't be attributed to subtasks.
    """
    if any(c.sid is None for c in calls):
        return None
    called = {c.sid for c in calls}
    completed = [s["sid"] for s in subtasks if s["status"] == "completed"]
    return Share(k=sum(sid not in called for sid in completed), n=len(completed))


def repeat_call_share(calls: Sequence[ToolCall]) -> Share:
    """Of the run's calls, those that repeat an earlier call of the same run exactly.

    Same tool and equal arguments, at any status: a repeat after a rate limit is still
    a call that could not tell the model anything new. Arguments are compared as the
    records hold them, where a long string is cut to its first 200 characters plus its
    length, so two calls that agree on both count as one call made twice.
    """
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for call in calls:
        key = (call.tool_name, json.dumps(call.arguments, sort_keys=True, default=str))
        repeats += key in seen
        seen.add(key)
    return Share(k=repeats, n=len(calls))


def rate_limited_share(calls: Sequence[ToolCall]) -> Share:
    """Of the run's calls, those the registry refused because the tool's per-task limit
    was spent. Such a call returns an error, not a result."""
    return Share(k=sum(c.status == "rate_limited" for c in calls), n=len(calls))


def cut_result_share(calls: Sequence[ToolCall], budget: int = RESULT_BUDGET_CHARS) -> Share:
    """Of the successful calls with a recorded result size, those whose result is longer
    than ``budget`` characters: the model saw only the first ``budget`` of them.

    The loop cuts the serialized JSON without a marker, so a result over the budget
    reaches the model as invalid JSON missing its end. Calls without ``result_chars``
    (failed calls, older records) are left out; with none recorded, n is 0.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    sizes = [c.result_chars for c in calls if c.result_chars is not None]
    return Share(k=sum(size > budget for size in sizes), n=len(sizes))


def critical_pass_share(verdict: TaskVerdict, task: AgentTask) -> Share:
    """How many of the task's critical criteria passed, failing closed as ``task_success``
    does. Success needs all of them; this shows how close a failed run came."""
    passed = criteria_passed(verdict, task)
    critical = [c.id for c in task.rubric if c.critical]
    return Share(k=sum(passed[cid] for cid in critical), n=len(critical))


def artifact_at_path(calls: Sequence[ToolCall], task: AgentTask) -> dict[str, bool]:
    """Each file the task requires, and whether a successful ``file_write`` wrote it at
    exactly that path.

    A required file is the ``path`` of a ``must_call`` entry that ``file_write`` can
    satisfy. The path must match character for character: the same content under
    another name or in a subdirectory is not the file the request asked for. Empty
    when the task requires no file.
    """
    written = {
        c.arguments.get("path")
        for c in calls
        if c.tool_name == "file_write" and c.status == "success"
    }
    required = [
        path
        for entry in task.must_call
        if "file_write" in entry.tools and isinstance(path := entry.args.get("path"), str)
    ]
    return {path: path in written for path in required}
