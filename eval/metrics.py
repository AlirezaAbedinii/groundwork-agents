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
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from schemas import PRF, CorrectnessVerdict, FaithfulnessVerdict, SweepPoint

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
